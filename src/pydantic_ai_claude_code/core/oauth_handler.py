"""OAuth error detection for Claude Code model.

This module provides OAuth error detection functionality that the Claude Agent SDK
doesn't have. It detects when OAuth tokens have expired or been revoked and
provides appropriate error messages.
"""

import json
import logging

logger = logging.getLogger(__name__)


def detect_oauth_error(stdout: str, stderr: str) -> tuple[bool, str | None]:
    """Detect OAuth authentication errors from Claude CLI output.

    The CLI returns OAuth errors in the JSON response (stdout), not stderr.
    Example response:
    {
        "type": "result",
        "subtype": "success",
        "is_error": true,
        "result": "OAuth token revoked - Please run /login"
    }

    Args:
        stdout: Standard output from Claude CLI (may contain JSON response)
        stderr: Standard error output (typically empty for OAuth errors)

    Returns:
        Tuple of (is_oauth_error, error_message)
        - is_oauth_error: True if this is an OAuth/authentication error
        - error_message: The error message from the CLI, or None if not an OAuth error
    """
    if not stdout:
        return False, None

    try:
        # Try to parse JSON from first line of stdout
        first_line = stdout.strip().split("\n")[0]
        response = json.loads(first_line)

        # Check if this is an error response
        if not isinstance(response, dict):
            return False, None

        if not response.get("is_error"):
            return False, None

        # Check the error message for OAuth/authentication indicators
        result_msg = response.get("result", "")
        error_msg = response.get("error", "")
        combined_msg = f"{result_msg} {error_msg}".lower()

        # OAuth token errors
        oauth_indicators = [
            "oauth token",
            "oauth_token",
            "/login",
            "authentication",
            "auth expired",
            "auth failed",
            "token expired",
            "token revoked",
            "please login",
            "please log in",
        ]

        for indicator in oauth_indicators:
            if indicator in combined_msg:
                # Return the actual error message from the result field
                actual_message = result_msg or error_msg or "Authentication error"
                logger.info("Detected OAuth error: %s", actual_message)
                return True, actual_message

    except (json.JSONDecodeError, KeyError, IndexError):
        # Not a valid JSON response, continue to check stderr
        pass

    return False, None
