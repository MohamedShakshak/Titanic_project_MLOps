# src/deployment/batch/extract.py
import logging
import os

import duckdb
import pandas as pd
from prefect import task

logger = logging.getLogger(__name__)


@task(
    name="extract-from-motherduck",
    description="Extract test passengers from MotherDuck",
    retries=3,
    retry_delay_seconds=30,
)
def extract_passengers(
    database: str = "titanic",
    table: str = "test_passengers",
) -> pd.DataFrame:
    """
    Extract test passenger data from MotherDuck.
    Returns a DataFrame with all rows from the table.
    """
    token = os.environ["MOTHERDUCK_TOKEN"]
    conn_str = f"md:{database}?motherduck_token={token}"

    logger.info("Connecting to MotherDuck database: %s", database)
    conn = duckdb.connect(conn_str)

    query = f"""
        SELECT
            PassengerId,
            Pclass,
            Name,
            Sex,
            Age,
            SibSp,
            Parch,
            Ticket,
            Fare,
            Cabin,
            Embarked
        FROM {table}
        ORDER BY PassengerId
    """

    logger.info("Extracting from table: %s", table)
    df = conn.execute(query).df()
    conn.close()

    logger.info("Extracted %d rows from %s.%s", len(df), database, table)
    return df