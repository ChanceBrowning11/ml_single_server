#!/usr/bin/env bash
set -euo pipefail

export AIRFLOW_HOME="$(pwd)/.airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
# DA Gs live in ./dags
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"

source venv/bin/activate

airflow scheduler
