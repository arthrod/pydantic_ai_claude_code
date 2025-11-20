"""Retry logic for Claude Code model.

This module provides retry logic for handling rate limits and transient
infrastructure failures that the Claude Agent SDK doesn't provide.
"""

import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def detect_rate_limit(error_output: str) -> tuple[bool, str | None]:
    """
    Detect whether the CLI output indicates a rate limit and extract the suggested reset time.
    
    Parameters:
        error_output (str): Combined stdout and stderr from the Claude CLI.
    
    Returns:
        tuple: (is_rate_limited, reset_time_str) where `is_rate_limited` is `True` if a rate-limit pattern was found, `False` otherwise, and `reset_time_str` is the extracted reset time (e.g., "3PM") or `None` if not present.
    """
    # Pattern matches: "limit reached.*resets 3PM" or similar
    rate_limit_match = re.search(
        r"limit reached.*resets\s+(\d{1,2}[AP]M)", error_output, re.IGNORECASE
    )

    if rate_limit_match:
        reset_time_str = rate_limit_match.group(1).strip()
        logger.info("Rate limit detected, reset time: %s", reset_time_str)
        return True, reset_time_str

    return False, None


def calculate_wait_time(reset_time_str: str) -> int:
    """
    Compute the number of seconds to wait until the given 12-hour reset time, including a 1-minute buffer.
    
    Parameters:
        reset_time_str (str): Reset time in 12-hour format (e.g., "3PM", "11AM"). If the parsed time is earlier than the current time, it is treated as occurring on the next day.
    
    Returns:
        int: Non-negative number of seconds to wait until the reset time plus a 1-minute buffer. If the input cannot be parsed, returns 300 (5 minutes) as a fallback.
    """
    try:
        now = datetime.now()
        # Parse time like "3PM" or "11AM"
        reset_time_obj = datetime.strptime(reset_time_str, "%I%p")
        reset_datetime = now.replace(
            hour=reset_time_obj.hour,
            minute=reset_time_obj.minute,
            second=0,
            microsecond=0,
        )

        # If reset time is in the past, add a day
        if reset_datetime < now:
            reset_datetime += timedelta(days=1)

        # Add 1-minute buffer
        wait_until = reset_datetime + timedelta(minutes=1)
        wait_seconds = int((wait_until - now).total_seconds())

        logger.info(
            "Rate limit resets at %s, waiting until %s (%d seconds)",
            reset_datetime.strftime("%I:%M%p"),
            wait_until.strftime("%I:%M%p"),
            wait_seconds,
        )

        return max(0, wait_seconds)

    except ValueError as e:
        # Fallback: wait 5 minutes if we can't parse the time
        logger.warning(
            "Could not parse reset time '%s': %s. Defaulting to 5-minute wait.",
            reset_time_str,
            e,
        )
        return 300


def detect_cli_infrastructure_failure(stderr: str) -> bool:
    """
    Detects transient Claude CLI infrastructure failures that should trigger a retry.
    
    Parameters:
        stderr (str): Standard error output from the Claude CLI.
    
    Returns:
        `true` if the stderr indicates a retryable infrastructure failure, `false` otherwise.
    """
    # Node.js module loading errors (e.g., missing yoga.wasm)
    if "Cannot find module" in stderr:
        return True

    # Node.js module resolution errors
    if "MODULE_NOT_FOUND" in stderr:
        return True

    # Other transient errors
    return bool("ENOENT" in stderr or "EACCES" in stderr)