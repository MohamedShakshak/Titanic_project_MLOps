# src/load/batch/load.py
import logging
import os

import duckdb
import pandas as pd
from prefect import task

logger = logging.getLogger(__name__)


@task(
    name="save-predictions-to-motherduck",
    description="Save prediction results back to MotherDuck",
    retries=3,
    retry_delay_seconds=30,
)
def save_predictions(
    results: pd.DataFrame,
    database: str = "titanic",
    table: str = "predictions",
) -> None:
    """
    Save prediction results to MotherDuck.

    Creates the predictions table if it doesn't exist.
    Appends new predictions — does not overwrite.
    """
    token = os.environ["MOTHERDUCK_TOKEN"]
    conn_str = f"md:{database}?motherduck_token={token}"

    logger.info("Connecting to MotherDuck to save predictions...")
    conn = duckdb.connect(conn_str)

    # Create predictions table if it doesn't exist
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            PassengerId   INTEGER,
            Survived      INTEGER,
            SurvivalProbability DOUBLE,
            ModelVersion  VARCHAR,
            PredictedAt   VARCHAR,
        )
    """)

    # Register the DataFrame as a DuckDB relation and insert
    conn.register("results_df", results)
    conn.execute(f"""
        INSERT INTO {table}
        SELECT * FROM results_df
    """)

    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    logger.info(
        "Saved %d new predictions. Total rows in %s.%s: %d",
        len(results),
        database,
        table,
        count,
    )

    # Show a preview of what was saved
    preview = conn.execute(
        f"SELECT * FROM {table} ORDER BY PredictedAt DESC LIMIT 5"
    ).df()
    logger.info("Latest predictions:\n%s", preview.to_string())

    conn.close()