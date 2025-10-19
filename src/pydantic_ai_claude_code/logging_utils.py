"""Utilities for standardized logging across the package."""

import logging


def log_section_separator(logger: logging.Logger, title: str | None = None) -> None:
    """Log a visual section separator with optional title.

    Args:
        logger: Logger instance to use
        title: Optional title to display in the separator
    """
    logger.info("=" * 80)
    if title:
        logger.info(title)


def log_section_end(logger: logging.Logger) -> None:
    """Log a visual section end separator.

    Args:
        logger: Logger instance to use
    """
    logger.info("=" * 80)
