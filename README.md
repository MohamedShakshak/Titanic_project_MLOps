# Titanic Survival MLOps Pipeline

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/Cookiecutter-Data%20Science-328F97?logo=cookiecutter" alt="Cookiecutter Data Science badge" />
</a>

## Overview

This repository contains an end-to-end MLOps workflow for the Titanic survival
prediction task. It uses DVC to reproduce the data and training pipeline, Hydra
and `params.yaml` for configuration, MLflow for experiment tracking and model
registry management, and Docker Compose for serving the promoted model.

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
- Dockerized MLflow inference server
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
|   |-- predict.py
|   `-- serve_model.sh
|-- src/
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
MLFLOW_TRACKING_URI=""
MLFLOW_TRACKING_USERNAME=""
MLFLOW_TRACKING_PASSWORD=""

MODEL_ALIAS="Production"
MODEL_STAGE="Production"
MODEL_REGISTRY_NAME="titanic-classifier"
PORT="5001"
```

For local MLflow runs, `MLFLOW_TRACKING_URI` can point to a local file store or
database. The Hydra MLflow config defaults to `mlruns` when no environment
variable is set.

## Data Access

The `download` stage uses Kaggle/KaggleHub to download the Titanic competition
dataset into:

```text
data/raw/train.csv
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
- `data/processed/`: train and validation splits
- `models/<model_name>_pipeline.pkl`: local joblib copy of the fitted pipeline
- `reports/metrics.json`: latest DVC-tracked evaluation metrics
- MLflow experiment: `titanic-training`
- MLflow registered model: `titanic-classifier`
- Production model URI for serving: `models:/titanic-classifier@Production`

## Notes

- Use `dvc repro` as the default way to reproduce the project pipeline.
- Use `params.yaml` for experiment changes that DVC should track.
- Use MLflow to compare runs, inspect artifacts, and manage production aliases.
- Use Docker Compose only after a model version has been promoted to
  `Production`.
