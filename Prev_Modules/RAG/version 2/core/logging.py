import logging
import sys
from loguru import logger
from .config import settings
from langchain_core.globals import set_verbose, set_debug


def setup_logging():
    """Configure logging for the application."""
    # Remove default handlers
    logger.remove()

    # Add stdout handler
    logger.add(
        sys.stdout,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True
    )

    # Add file handler
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT
    )

    # Set langchain verbosity
    set_verbose(settings.VERBOSE)
    if settings.DEBUG:
        set_debug(True)
