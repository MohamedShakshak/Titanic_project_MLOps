# src/deployment/batch/flow.py
"""
Prefect flow for batch predictions.

This flow:
1. Extracts test passengers from MotherDuck
2. Transforms and validates the data
3. Loads the Production model from DagsHub MLflow registry
4. Runs predictions on all passengers
5. Saves predictions back to MotherDuck

Usage:
    # Run directly
    uv run python src/batch/flow.py

    # Run via Prefect CLI
    uv run prefect run -p src/batch/flow.py -n titanic-batch-predictions
"""

import logging
import os

from dotenv import load_dotenv
from prefect import flow
from prefect.logging import get_run_logger

from src.deployment.batch.extract import extract_passengers
from src.deployment.batch.transform import transform_passengers
from src.deployment.batch.predict import load_model, run_predictions
from src.deployment.batch.load import save_predictions

load_dotenv()


@flow(
    name="titanic-batch-predictions",
    description="Extract passengers from MotherDuck, predict survival, save results",
    log_prints=True,
)
def titanic_batch_flow(
    # MotherDuck settings
    database: str = "titanic",
    input_table: str = "test_passengers",
    output_table: str = "predictions",
    # MLflow registry settings
    model_name: str = "titanic-classifier",
    model_stage: str = "Production",
) -> dict:
    """
    Main batch prediction flow.
    All parameters can be overridden at runtime from the Prefect UI or CLI.
    """
    logger = get_run_logger()
    logger.info("Starting titanic batch prediction flow")
    logger.info("Input  : %s.%s", database, input_table)
    logger.info("Output : %s.%s", database, output_table)
    logger.info("Model  : %s (%s)", model_name, model_stage)

    # Step 1 — Extract
    raw_df = extract_passengers(database=database, table=input_table)

    # Step 2 — Transform
    features, passenger_ids = transform_passengers(raw_df)

    # Step 3 — Load model
    pipeline, model_version = load_model(
        model_name=model_name,
        model_stage=model_stage,
    )

    # Step 4 — Predict
    results = run_predictions(
        pipeline=pipeline,
        model_version=model_version,
        features=features,
        passenger_ids=passenger_ids,
    )

    # Step 5 — Save
    save_predictions(
        results=results,
        database=database,
        table=output_table,
    )

    summary = {
        "rows_processed": len(results),
        "survived": int(results["Survived"].sum()),
        "did_not_survive": int((results["Survived"] == 0).sum()),
        "model_version": model_version,
        "output_table": f"{database}.{output_table}",
    }

    logger.info("Flow complete: %s", summary)
    return summary


if __name__ == "__main__":
    # Run the flow directly
    result = titanic_batch_flow()
    print(result)