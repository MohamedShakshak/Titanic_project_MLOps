# Titanic Survival MLOps Pipeline

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/Cookiecutter-Data%20Science-328F97?logo=cookiecutter" alt="Cookiecutter Data Science badge" />
</a>

## Overview

This repository contains an end-to-end MLOps workflow for the Titanic survival
prediction task. It uses DVC to reproduce the data and training pipeline, Hydra
and `params.yaml` for configuration, MLflow for experiment tracking and model
registry management, Docker-based services for online inference, and a Prefect
batch prediction flow backed by MotherDuck.

The default workflow trains a scikit-learn pipeline that includes feature
engineering, preprocessing, model fitting, evaluation, local artifact storage,
and MLflow model registration.

## Features

- Reproducible Python environment managed with `uv`
- DVC pipeline with `download` and `train` stages
- Hydra configuration under `conf/`
- DVC-tracked runtime parameters in `params.yaml`
- Logistic Regression and Random Forest model options
- Feature engineering with custom scikit-learn transformers
- MLflow experiment tracking and model registry integration
- Best-model promotion to the `Production` alias
- Dockerized MLflow registry-backed inference server
- FastAPI no-registry online inference service that loads a local model file
- Prefect batch prediction flow using MotherDuck as input and output storage
- Pytest, Ruff, and MkDocs project tooling

## Tech Stack

- Python 3.11
- uv
- DVC
- Hydra
- MLflow
- scikit-learn
- pandas / NumPy
- joblib
- FastAPI / Uvicorn / Pydantic
- Prefect
- DuckDB / MotherDuck
- Docker / Docker Compose
- pytest and Ruff

## Project Structure

```text
Titanic_project_MLOps/
|-- conf/
|   |-- config.yaml
|   |-- data/default.yaml
|   |-- mlflow/default.yaml
|   |-- model/
|   |   |-- logistic.yaml
|   |   `-- random_forest.yaml
|   `-- training/default.yaml
|-- data/
|   |-- processed/
|   `-- raw/
|-- docs/
|   |-- README.md
|   |-- mkdocs.yml
|   `-- docs/
|-- models/
|-- reports/
|-- scripts/
|   |-- load_to_motherduck.py
|   |-- predict.py
|   `-- serve_model.sh
|-- server.py
|-- src/
|   |-- deployment/
|   |   |-- batch/
|   |   |   |-- extract.py
|   |   |   |-- flow.py
|   |   |   |-- load.py
|   |   |   |-- predict.py
|   |   |   `-- transform.py
|   |   `-- online/
|   |       |-- api.py
|   |       |-- request.py
|   |       `-- response.py
|   |-- prediction/
|   |   `-- predict.py
|   `-- training/
|       |-- data/
|       |   |-- download_data.py
|       |   `-- load.py
|       |-- features/
|       |   |-- preprocess.py
|       |   `-- transformers.py
|       |-- evaluate.py
|       |-- pipeline.py
|       |-- promote_model.py
|       `-- train.py
|-- tests/
|-- docker-compose.inference.yaml
|-- Dockerfile.inference
|-- Dockerfile.online.no_registry
|-- Dockerfile.online.no_registry.dockerignore
|-- dvc.yaml
|-- params.yaml
|-- pyproject.toml
|-- trainer.py
`-- README.md
```

## Setup

Create and activate the local environment:

```bash
make create_environment
```

Windows:

```bash
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
make requirements
```

Or run `uv` directly:

```bash
uv sync
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values required for your MLflow
setup:

```bash
# Kaggle
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_key

# DagsHub / MLflow
MLFLOW_TRACKING_URI=https://dagshub.com/your_username/your_repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# MotherDuck
MOTHERDUCK_TOKEN=your_motherduck_token

# Prefect Cloud, optional
PREFECT_API_KEY=your_prefect_api_key
PREFECT_API_URL=https://api.prefect.io
```

For local MLflow runs, `MLFLOW_TRACKING_URI` can point to a local file store or
database. The Hydra MLflow config defaults to `mlruns` when no environment
variable is set.

The serving containers also use runtime settings such as `MODEL_ALIAS`,
`MODEL_STAGE`, `MODEL_REGISTRY_NAME`, `PORT`, and `MODEL_PATH`. The Compose file
sets the MLflow serving defaults, and the no-registry Dockerfile defaults to
`MODEL_PATH=models/logistic_pipeline.pkl`.

## Data Access

The `download` stage uses Kaggle/KaggleHub to download the Titanic competition
dataset into:

```text
data/raw/train.csv
data/raw/test.csv
```

Make sure your Kaggle credentials are available before running the pipeline.
Common options are:

- Place `kaggle.json` in the project root.
- Place `kaggle.json` in your default Kaggle credentials directory.
- Configure Kaggle credentials through environment variables.

## DVC Pipeline

The DVC pipeline is defined in `dvc.yaml`.

Run the full pipeline:

```bash
dvc repro
```

Run only the download stage:

```bash
dvc repro download
```

Run only the training stage:

```bash
dvc repro train
```

### `download` Stage

Runs:

```bash
uv run python trainer.py --config-name config stage=download
```

Tracks these parameters from `params.yaml`:

- `data.competition_name`
- `data.raw_dir`
- `data.overwrite`

Produces:

- `data/raw/train.csv`
- `data/raw/test.csv`

### `train` Stage

Runs:

```bash
uv run python trainer.py --config-name config stage=train
```

Tracks these parameters from `params.yaml`:

- `training.test_size`
- `training.random_state`
- `model.name`
- `model.params`

Produces:

- `data/processed/`
- `models/`
- `reports/metrics.json`

## Batch Prediction

Batch prediction code lives under `src/deployment/batch/`. The flow is defined
in `src/deployment/batch/flow.py` and orchestrated with Prefect.

The batch flow:

- Extracts passenger rows from MotherDuck table `titanic.test_passengers`
- Cleans and validates the raw Titanic-style columns
- Loads the `titanic-classifier` production model from the MLflow registry
- Runs predictions and survival probabilities
- Appends results to MotherDuck table `titanic.predictions`

Before running the batch flow, load the Titanic test data into MotherDuck:

```bash
uv run python scripts/load_to_motherduck.py
```

Then run the flow directly:

```bash
uv run python src/deployment/batch/flow.py
```

Default flow parameters:

```text
database: titanic
input_table: test_passengers
output_table: predictions
model_name: titanic-classifier
model_stage: Production
```

Required environment variables:

- `MOTHERDUCK_TOKEN`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

## Configuration

Hydra loads the base config from `conf/config.yaml`. At runtime, `trainer.py`
merges values from `params.yaml` on top of the Hydra config so DVC-tracked
parameters control reproducible experiments.

The current model selection is controlled by:

```yaml
model:
  name: random_forest
```

Supported model names:

- `logistic`
- `random_forest`

Hyperparameters live under `model.params.<model_name>` in `params.yaml`.

## Training and Tracking

Training is orchestrated by `trainer.py`.

Run all stages directly with Hydra:

```bash
uv run python trainer.py stage=all
```

Run a single stage directly:

```bash
uv run python trainer.py stage=download
uv run python trainer.py stage=train
```

During training, the project:

- Loads `data/raw/train.csv`
- Splits train and validation data
- Builds feature engineering and preprocessing steps
- Trains the selected model
- Computes accuracy, ROC-AUC, and F1
- Saves processed splits to `data/processed/`
- Saves a joblib pipeline to `models/<model_name>_pipeline.pkl`
- Writes latest metrics to `reports/metrics.json`
- Logs params, metrics, signature, input example, and model artifact to MLflow
- Registers the model as `titanic-classifier`

## Model Promotion

After training one or more runs, promote the best registered model version by
ROC-AUC:

```bash
uv run python src/training/promote_model.py
```

The promotion script searches the `titanic-training` MLflow experiment, finds
the best finished run by `roc_auc`, locates the matching registered model
version, and assigns the `Production` alias to that version.

## Inference

The project currently has two online inference options:

- MLflow registry-backed serving through `Dockerfile.inference` and
  `docker-compose.inference.yaml`
- No-registry FastAPI serving through `Dockerfile.online.no_registry` and
  `server.py`

### MLflow Registry-backed Serving

After a model has been trained, registered, and promoted to `Production`, start
the inference service:

```bash
docker compose -f docker-compose.inference.yaml up --build
```

The service exposes the MLflow model server on:

```text
http://localhost:5001
```

Health check:

```bash
curl http://localhost:5001/ping
```

Run the example inference client:

```bash
uv run python scripts/predict.py
```

There is also a direct registry-loading example at:

```bash
uv run python src/prediction/predict.py
```

Prediction inputs should use the raw Titanic-style columns expected by the
training pipeline. The saved model includes the preprocessing and feature
engineering steps.

### No-registry FastAPI Serving

The no-registry service loads a local `joblib` pipeline directly from
`MODEL_PATH`. By default, the Dockerfile uses:

```text
models/logistic_pipeline.pkl
```

Build the image:

```bash
docker build -f Dockerfile.online.no_registry -t titanic-online-no-registry .
```

Run it with the default model path:

```bash
docker run --rm -p 8000:8000 titanic-online-no-registry
```

Or serve the Random Forest artifact:

```bash
docker run --rm -p 8000:8000 -e MODEL_PATH=models/random_forest_pipeline.pkl titanic-online-no-registry
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction endpoint:

```text
POST http://localhost:8000/api/v1/predict
```

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"Pclass\":1,\"Sex\":\"female\",\"Age\":29.0,\"SibSp\":0,\"Parch\":0,\"Fare\":211.3,\"Embarked\":\"S\",\"Name\":\"Cumings, Mrs. John Bradley\",\"Ticket\":\"PC 17599\",\"Cabin\":\"C85\"}"
```

## Development Commands

Install dependencies:

```bash
make requirements
```

Run tests:

```bash
make test
```

Run lint checks:

```bash
make lint
```

Format code:

```bash
make format
```

Run the docs site locally from the project root:

```bash
mkdocs serve -f docs/mkdocs.yml
```

## Key Artifacts

- `data/raw/train.csv`: downloaded Titanic training data
- `data/raw/test.csv`: downloaded Titanic test data for batch scoring
- `data/processed/`: train and validation splits
- `models/<model_name>_pipeline.pkl`: local joblib copy of the fitted pipeline
- `reports/metrics.json`: latest DVC-tracked evaluation metrics
- MLflow experiment: `titanic-training`
- MLflow registered model: `titanic-classifier`
- Production model URI for serving: `models:/titanic-classifier@Production`
- No-registry API health endpoint: `GET /health`
- No-registry API prediction endpoint: `POST /api/v1/predict`
- Batch input table: `titanic.test_passengers`
- Batch output table: `titanic.predictions`

## Notes

- Use `dvc repro` as the default way to reproduce the project pipeline.
- Use `params.yaml` for experiment changes that DVC should track.
- Use MLflow to compare runs, inspect artifacts, and manage production aliases.
- Use the batch flow when predictions should be written back to MotherDuck.
- Use the MLflow Docker Compose path only after a model version has been promoted to
  `Production`.
- Use the no-registry FastAPI image when you want to serve a local model file
  directly without depending on the MLflow model registry at runtime.
