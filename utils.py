import logging


def connect_to_db() -> tuple:
    """
    Connects to the PostgreSQL database and returns a tuple
    (connection, connection_cursor).

    example: conn, cursor = connect_to_db()
    """
    import psycopg2
    import os
    from dotenv import load_dotenv

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
        raise "Error connecting to the database. Please check your connection settings."


def configure_logging() -> logging.Logger:
    """
    Configures the logging for the application.

    This function sets up the basic configuration for logging,
    specifying the log level, format, and other settings.
    It then returns a logger instance that can be used throughout the application.

    Returns:
        logging.Logger: A logger instance configured with the specified settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s - '
    )
    return logging.getLogger(__name__)