"""SDK error types for Claude Code integration.

These exceptions provide a consistent error hierarchy for SDK operations.
"""

from __future__ import annotations


class ClaudeSDKError(Exception):
    """Base exception for all SDK errors."""

    pass


class CLIConnectionError(ClaudeSDKError):
    """Error connecting to or communicating with the CLI."""

    pass


class CLINotFoundError(CLIConnectionError):
    """Claude CLI executable not found."""

    def __init__(self, message: str = "Claude CLI not found in PATH"):
        super().__init__(message)


class ProcessError(ClaudeSDKError):
    """Error during subprocess execution."""

    def __init__(
        self,
        message: str,
        *,
        return_code: int | None = None,
        stderr: str | None = None,
    ):
        super().__init__(message)
        self.return_code = return_code
        self.stderr = stderr


class TimeoutError(ClaudeSDKError):
    """Operation timed out."""

    def __init__(self, message: str = "Operation timed out", *, timeout_seconds: int | None = None):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class AuthenticationError(ClaudeSDKError):
    """Authentication failed (e.g., OAuth token expired)."""

    pass


class RateLimitError(ClaudeSDKError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after_seconds: int | None = None,
    ):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
