# src/deployment/batch/transform.py
import logging

import pandas as pd
from prefect import task

logger = logging.getLogger(__name__)


@task(
    name="transform-passengers",
    description="Validate and prepare passenger data for prediction",
)
def transform_passengers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean passenger data before prediction.

    The sklearn pipeline handles most transformations internally
    (imputation, scaling, encoding). This task handles:
    - Removing columns not needed by the pipeline
    - Logging data quality issues
    - Type coercions that DuckDB may have changed
    """
    logger.info("Input shape: %s", df.shape)
    logger.info("Input columns: %s", list(df.columns))

    # Store PassengerId separately — it is not a feature
    passenger_ids = df["PassengerId"].copy() if "PassengerId" in df.columns else None

    # Drop PassengerId from features — the pipeline does not use it
    feature_cols = [c for c in df.columns if c != "PassengerId"]
    features = df[feature_cols].copy()

    # Log missing values
    missing = features.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        logger.info("Missing values found:\n%s", missing_cols.to_string())
        logger.info("The sklearn pipeline will handle imputation.")

    # Ensure string columns are strings — DuckDB may return None for nulls
    str_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
    for col in str_cols:
        if col in features.columns:
            features[col] = features[col].fillna("").astype(str)

    # Ensure numeric columns are numeric
    num_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    for col in num_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    logger.info("Transformed shape: %s", features.shape)
    return features, passenger_ids