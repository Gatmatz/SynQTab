import os
from typing import Optional, Tuple, Any

from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pandas as pd

from synqtab.utils.logging_utils import get_logger

LOG = get_logger(__file__)


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
        LOG.exception("Error connecting to the database.")
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
        LOG.info(f"Wrote {len(df)} rows to {schema}.{table_name}")
    except Exception:
        LOG.exception(f"Failed to write DataFrame to {schema}.{table_name}")
        raise


def read_table_from_db(
    table_name: str,
    schema: str = "public",
    engine: Optional[Engine] = None,
    columns: Optional[list] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a table from Postgres and return it as a pandas DataFrame.

    - If `engine` is None, creates one using `create_db_engine`.
    - `columns` can be a list of column names to read.
    - `index_col` can be set to a column name to use as the DataFrame index.
    """
    if engine is None:
        engine = create_db_engine()

    try:
        if columns is None:
            df = pd.read_sql_table(table_name, con=engine, schema=schema, index_col=index_col)
        else:
            df = pd.read_sql_table(table_name, con=engine, schema=schema, columns=columns, index_col=index_col)

        LOG.info(f"Read {len(df)} rows from {schema}.{table_name}")
        return df
    except Exception:
        LOG.exception(f"Failed to read table {schema}.{table_name}")
        raise

