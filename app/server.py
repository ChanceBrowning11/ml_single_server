# app/server.py
import os, json, time, threading
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

# ---- Hard-coded config (simple, explicit) ----
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# MODEL_NAME          = "iris-classifier"
# MODEL_VERSION       = "1"

# MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
# model = mlflow.pyfunc.load_model(MODEL_URI)

# ----- Config -----
MODEL_VERSION       = "2"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000") # update ip when running MLFlow
MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
DEFAULT_MODEL_VERSION = int(os.getenv("MODEL_VERSION", "2")) # optional: seed a default on first boot, otherwise get the latest
STATE_PATH = os.getenv("MODEL_STATE_PATH", "./model_state.json")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

class ModelState(BaseModel):
    model_name: str
    version: int
    uri: str
    loaded_at: float
    
class SelectVersionRequest(BaseModel):
    version: int = Field(..., gt=0)

class VersionedModelManager:
    def __init__(self, model_name: str, state_path: str, default_version: int):
        self.model_name = model_name
        self.state_path = state_path
        self.default_version = default_version
        self._lock = threading.RLock()
        self._model = None
        self._state: Optional[ModelState] = None
        self._cold_start()
        
    # --- helpers ---
    def _persist(self, state: ModelState):
        with open(self.state_path, "w") as f:
            json.dump(state.model_dump(), f)
    
    def _load_state_file(self) -> Optional[ModelState]:
        if not os.path.exists(self.state_path):
            return None
        with open(self.state_path, "r") as f:
            return ModelState(**json.load(f))
    
    # --- mlflow helpers ---
    def _latest_version(self) -> int:
        versions = client.search_model_versions(f"name = '{self.model_name}'")
        if not versions:
            raise RuntimeError(f"No versions found for model '{self.model_name}'")
        return max(int(v.version) for v in versions)
    
    def _load_model(self, version: int):
        uri = f"models:/{self.model_name}/{version}"
        model = mlflow.pyfunc.load_model(uri) # raises if not found/bad
        return uri, model
    
    # --- boot logic ---
    def _cold_start(self):
        with self._lock:
            candidates = []
            persisted = self._load_state_file()
            if persisted:
                candidates.append(persisted.version)
            # default from env (yours was hard-coded "1")
            if self.default_version not in candidates:
                candidates.append(self.default_version)
            # latest in registry
            try:
                latest = self._latest_version()
                if latest not in candidates:
                    candidates.append(latest)
            except Exception:
                pass
            
            last_err = None
            for v in candidates:
                try:
                    uri, model = self._load_model(v)
                    self._model = model
                    self._state = ModelState(
                        model_name=self.model_name, version=int(v), uri=uri, loaded_at=time.time()    
                    )
                    self._persist(self._state)
                    return
                except Exception as e:
                    last_err = e
            raise RuntimeError(f"Failed to load any candidate version for {self.model_name}: {last_err}")
            
    # --- public API ---
    def current(self) -> ModelState:
        with self._lock:
            return self._state
    
    def select_version(self, version: int) -> ModelState:
        # validate that version exists in registry
        try:
            client.get_model_version(self.model_name, str(version))
        except Exception:
            raise HTTPException(status_code=404, detail=f"Version {version} not found for {self.model_name}")
        
        # Load new model first; then swap under lock (so we don't drop live traffic)
        uri, new_model = self._load_model(version)
        with self._lock:
            prev_model, prev_state = self._model, self._state
            try:
                self._model = new_model
                self._state = ModelState(
                    model_name = self.model_name, version=int(version), uri=uri, loaded_at=time.time()    
                )
                self._persist(self._state)
                return self._state
            except Exception:
                # rollback on any error
                self._model, self._state = prev_model, prev_state
                raise HTTPException(status_code=500, detail=f"Failed to switch to version {version}")
    
    def predict(self, records: List[dict]):
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded")
            df = pd.DataFrame.from_records(records)
            y = self._model.predict(df)
            # normalize to list of ints (common for iris)
            class_ids = [int(v) for v in y]
            labels = [IRIS_LABELS.get(i, str(i)) for i in class_ids]
            return class_ids, labels
            
# one global manager
manager = VersionedModelManager(MODEL_NAME, STATE_PATH, DEFAULT_MODEL_VERSION)

# ----- FastAPI -----
app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    st = manager.current()
    return {"status": "ok", "model_uri": st.uri, "model_version": st.version}

# TODO Add endpoint to get the current model serving version
@app.get("/model", tags=["model"], summary="Get current model version")
def get_model():
    return manager.current().model_dump()

# TODO Add endpoint to update the serving version
@app.post("/model/select", tags=["model"], summary="Switch to a specific model version")
def select_model_version(body: SelectVersionRequest):
    st = manager.select_version(body.version)
    return st.model_dump()

# TODO Predict using the correct served version
@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        # convert pydantic models to dicts
        records = [s.model_dump() for s in req.samples]
        class_ids, labels = manager.predict(records)
        return PredictResponse(class_id=class_ids, class_label=labels)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
