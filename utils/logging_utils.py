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
            "%(asctime)s %(levelname)s %(name)s:%(lineno)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent double logging if root logger is also configured.
        logger.propagate = False

    return logger
