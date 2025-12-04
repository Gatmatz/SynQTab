import os
from typing import Optional, Tuple, Any

from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pandas as pd

import logging
from typing import Optional

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger to be used across the project.
    - `name`: typically `__name__` from the caller.
    - `level`: logging level (default INFO).
    Ensures a single StreamHandler is added only once to avoid duplicate logs.
    """
    logger_name = name or "SynQTab"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # If no handlers attached, add a StreamHandler with a standard formatter.
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent double logging if root logger is also configured.
        logger.propagate = False

    return logger


def connect_to_db() -> Tuple[psycopg2.extensions.connection, Any]:
    """
    Return a psycopg2 connection and cursor. Loads env from .env.
    Raises the original psycopg2 exception on failure.
    """
    load_dotenv()
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            port=os.getenv("POSTGRES_MAPPED_PORT", "5432"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DB", "postgres"),
        )
        return conn, conn.cursor()
    except psycopg2.OperationalError:
        logging.exception("Error connecting to the database.")
        raise


def create_db_engine(echo: bool = False) -> Engine:
    """
    Create and return a SQLAlchemy Engine using environment variables.
    """
    load_dotenv()
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_MAPPED_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, echo=echo, pool_pre_ping=True)


def write_dataframe_to_db(
    df: pd.DataFrame,
    table_name: str,
    schema: str = "public",
    engine: Optional[Engine] = None,
    if_exists: str = "replace",
    index: bool = False,
    chunksize: int = 1000,
    method: Optional[str] = "multi",
) -> None:
    """
    Write a pandas DataFrame to Postgres using SQLAlchemy engine.

    - If `engine` is None, one is created from environment variables.
    - `if_exists` can be 'replace', 'append', or 'fail'.
    - `method='multi'` and `chunksize` speed up inserts for many rows.
    """
    if engine is None:
        engine = create_db_engine()

    try:
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            method=method,
            chunksize=chunksize,
        )
        logging.info("Wrote %d rows to %s.%s", len(df), schema, table_name)
    except Exception:
        logging.exception("Failed to write DataFrame to %s.%s", schema, table_name)
        raise
