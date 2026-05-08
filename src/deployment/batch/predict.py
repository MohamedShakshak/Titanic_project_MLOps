# src/deployment/batch/predict.py
import logging
import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import task

logger = logging.getLogger(__name__)


@task(
    name="load-model-from-registry",
    description="Load Production model from DagsHub MLflow registry",
    retries=2,
    retry_delay_seconds=60,
)
def load_model(
    model_name: str = "titanic-classifier",
    model_stage: str = "Production",
):
    """Load the registered model from DagsHub MLflow registry."""
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{model_stage}"
    logger.info("Loading model from: %s", model_uri)

    pipeline = mlflow.sklearn.load_model(model_uri)

    # Get version info for logging
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[model_stage])
    version = versions[0].version if versions else "unknown"

    logger.info("Loaded model version: %s", version)
    return pipeline, str(version)


@task(
    name="run-predictions",
    description="Run batch predictions on all passengers",
)
def run_predictions(
    pipeline,
    model_version: str,
    features: pd.DataFrame,
    passenger_ids: pd.Series,
) -> pd.DataFrame:
    """
    Run predictions on the entire feature DataFrame.
    Returns a DataFrame with PassengerId, predictions, and probabilities.
    """
    logger.info("Running predictions on %d passengers...", len(features))

    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    results = pd.DataFrame({
        "PassengerId": passenger_ids.values if passenger_ids is not None else range(len(features)),
        "Survived": predictions.astype(int),
        "SurvivalProbability": probabilities.round(4),
        "ModelVersion": model_version,
        "PredictedAt": datetime.utcnow().isoformat(),
    })

    survived_count = int(predictions.sum())
    logger.info(
        "Predictions complete: %d/%d survived (%.1f%%)",
        survived_count,
        len(predictions),
        100 * survived_count / len(predictions),
    )

    return results