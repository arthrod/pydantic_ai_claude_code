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
    Detects a rate-limit condition in combined Claude CLI output and extracts the reported reset time.
    
    Parameters:
        error_output (str): Combined stdout and stderr text from the Claude CLI.
    
    Returns:
        tuple[bool, str | None]: (True, reset_time_str) when a reset time like '3PM' is found; (False, None) otherwise.
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
    Compute how many seconds to wait until the provided 12-hour reset time, including a 1-minute buffer.
    
    Parameters:
        reset_time_str (str): Reset time in 12-hour format with AM/PM (examples: "3PM", "11AM").
    
    Returns:
        int: Number of seconds to wait until the reset time plus a 1-minute buffer. If the parsed reset time is earlier than now, the function treats it as occurring the next day. If parsing fails, returns 300 (5 minutes) as a fallback.
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
    Detects whether Claude CLI stderr contains a retryable infrastructure failure.
    
    Parameters:
        stderr (str): Error output from the Claude CLI.
    
    Returns:
        True if the stderr indicates a retryable infrastructure failure (e.g., missing or unresolved Node modules, filesystem access errors), False otherwise.
    """
    # Node.js module loading errors (e.g., missing yoga.wasm)
    if "Cannot find module" in stderr:
        return True

    # Node.js module resolution errors
    if "MODULE_NOT_FOUND" in stderr:
        return True

    # Other transient errors
    return bool("ENOENT" in stderr or "EACCES" in stderr)