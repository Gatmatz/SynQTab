import os
import logging
from typing import Optional, Tuple, Any

from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pandas as pd


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
