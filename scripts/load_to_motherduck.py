import logging
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOTHERDUCK_TOKEN = os.environ["MOTHERDUCK_TOKEN"]

TEST_CSV_PATH = Path("data/raw/test.csv")
DATABASE_NAME = "titanic"
TABLE_NAME = "test_passengers"


def load_test_data() -> None:
    logger.info("Connecting to MotherDuck...")

    # 1. connect WITHOUT database
    conn = duckdb.connect("md:")

    # 2. ensure database exists
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")

    # 3. switch to database
    conn.execute(f"USE {DATABASE_NAME}")

    logger.info("Loading CSV into MotherDuck...")

    # 4. use absolute path (important!)
    csv_path = TEST_CSV_PATH.resolve()

    conn.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT * FROM read_csv_auto('{csv_path}')
    """)

    count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    logger.info("Loaded %d rows into %s.%s", count, DATABASE_NAME, TABLE_NAME)

    preview = conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 3").df()
    logger.info("Preview:\n%s", preview.to_string())

    conn.close()
    logger.info("Done.")
    
if __name__ == "__main__":
    load_test_data()