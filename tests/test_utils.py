"""Tests for utility functions."""

import os
import shutil
from unittest import mock

import pytest

from pydantic_ai_claude_code.exceptions import ClaudeOAuthError
from pydantic_ai_claude_code.types import ClaudeCodeSettings
from pydantic_ai_claude_code.utils import (
    build_claude_command,
    detect_oauth_error,
    parse_stream_json_line,
    resolve_claude_cli_path,
)


def test_build_claude_command_basic():
    """Test building basic Claude command."""
    # Mock shutil.which to provide a predictable path
    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        cmd = build_claude_command()

        # First element should be the resolved claude path
        assert cmd[0] == "/usr/bin/claude"
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        # Note: Prompts are now passed via stdin, not command-line arguments


def test_build_claude_command_with_settings():
    """Test building command with settings."""
    settings: ClaudeCodeSettings = {
        "model": "sonnet",
        "allowed_tools": ["Read", "Edit"],
        "permission_mode": "acceptEdits",
    }

    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        cmd = build_claude_command(settings=settings)

        assert "--model" in cmd
        assert "sonnet" in cmd
        assert "--allowed-tools" in cmd
        assert "Read" in cmd
        assert "Edit" in cmd
        assert "--permission-mode" in cmd
        assert "acceptEdits" in cmd


def test_build_claude_command_stream_json():
    """Test building command with stream-json output."""
    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        cmd = build_claude_command(output_format="stream-json")

        assert "--output-format" in cmd
        assert "stream-json" in cmd
        # Verbose is no longer always included
        assert "--include-partial-messages" in cmd


def test_parse_stream_json_line_valid():
    """Test parsing valid stream JSON line."""
    line = '{"type":"result","subtype":"success","result":"test"}'
    event = parse_stream_json_line(line)

    assert event is not None
    assert event["type"] == "result"
    assert event["subtype"] == "success"
    assert event["result"] == "test"


def test_parse_stream_json_line_empty():
    """Test parsing empty line."""
    event = parse_stream_json_line("")
    assert event is None


def test_parse_stream_json_line_whitespace():
    """Test parsing line with only whitespace."""
    event = parse_stream_json_line("   \n  ")
    assert event is None


def test_parse_stream_json_line_invalid():
    """Test parsing invalid JSON."""
    event = parse_stream_json_line("not json")
    assert event is None


def test_resolve_claude_cli_path_from_settings():
    """Test CLI path resolution from settings (highest priority)."""
    settings: ClaudeCodeSettings = {
        "claude_cli_path": "/custom/path/to/claude",
    }

    # Mock environment and shutil.which to ensure settings takes precedence
    with (
        mock.patch.dict(os.environ, {"CLAUDE_CLI_PATH": "/env/path/claude"}),
        mock.patch("shutil.which", return_value="/usr/bin/claude"),
    ):
        path = resolve_claude_cli_path(settings)
        assert path == "/custom/path/to/claude"


def test_resolve_claude_cli_path_from_env():
    """Test CLI path resolution from environment variable (second priority)."""
    # Don't provide settings with claude_cli_path
    settings: ClaudeCodeSettings = {}

    with (
        mock.patch.dict(os.environ, {"CLAUDE_CLI_PATH": "/env/path/claude"}),
        mock.patch("shutil.which", return_value="/usr/bin/claude"),
    ):
        path = resolve_claude_cli_path(settings)
        assert path == "/env/path/claude"


def test_resolve_claude_cli_path_from_which():
    """Test CLI path resolution from PATH via shutil.which (third priority)."""
    # Mock shutil.which to return a path
    with (
        mock.patch.dict(os.environ, {}, clear=True),  # Clear environment
        mock.patch("shutil.which", return_value="/usr/local/bin/claude"),
    ):
        path = resolve_claude_cli_path()
        assert path == "/usr/local/bin/claude"


def test_resolve_claude_cli_path_not_found():
    """Test CLI path resolution when claude is not found."""
    with (
        mock.patch.dict(os.environ, {}, clear=True),  # Clear environment
        mock.patch("shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="Could not find claude CLI binary"),
    ):
        resolve_claude_cli_path()


def test_resolve_claude_cli_path_real_which():
    """Test CLI path resolution with real shutil.which (integration-style test)."""
    # This test will pass if 'claude' is actually in the PATH
    # Otherwise it should raise RuntimeError
    actual_path = shutil.which("claude")

    if actual_path:
        # claude is in PATH - should resolve successfully
        path = resolve_claude_cli_path()
        assert path == actual_path
    else:
        # claude not in PATH - should raise RuntimeError
        with pytest.raises(RuntimeError, match="Could not find claude CLI binary"):
            resolve_claude_cli_path()


def test_build_claude_command_with_custom_cli_path():
    """Test building command with custom CLI path from settings."""
    settings: ClaudeCodeSettings = {
        "claude_cli_path": "/custom/claude",
    }

    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        cmd = build_claude_command(settings=settings)

        # First element should be the custom path
        assert cmd[0] == "/custom/claude"
        assert "--print" in cmd


def test_build_claude_command_with_extra_cli_args():
    """Test building command with extra CLI arguments."""
    settings: ClaudeCodeSettings = {
        "extra_cli_args": ["--debug", "api", "--mcp-config", "config.json"],
    }

    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        cmd = build_claude_command(settings=settings)

        # Extra args should be present in command
        assert "--debug" in cmd
        assert "api" in cmd
        assert "--mcp-config" in cmd
        assert "config.json" in cmd

        # Note: Prompts are now passed via stdin, not command-line arguments


def test_build_claude_command_with_multiple_extra_args():
    """Test building command with various extra CLI arguments."""
    settings: ClaudeCodeSettings = {
        "model": "sonnet",
        "extra_cli_args": [
            "--verbose",
            "--add-dir", "/custom/path",
            "--agents", '{"reviewer": {"description": "Reviews code"}}',
        ],
    }

    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        cmd = build_claude_command(settings=settings)

        # Model arg should still be present
        assert "--model" in cmd
        assert "sonnet" in cmd

        # Extra args should be present
        assert "--verbose" in cmd
        assert "--add-dir" in cmd
        assert "/custom/path" in cmd
        assert "--agents" in cmd


def test_build_claude_command_without_extra_args():
    """Test building command without extra CLI arguments works normally."""
    settings: ClaudeCodeSettings = {
        "model": "opus",
    }

    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        cmd = build_claude_command(settings=settings)

        # Should build normally without errors
        assert cmd[0] == "/usr/bin/claude"
        assert "--model" in cmd
        assert "opus" in cmd
        # Note: Prompts are now passed via stdin, not command-line arguments


def test_detect_oauth_error_oauth_token_revoked():
    """Test detecting OAuth token revoked error."""
    stdout = '{"type":"result","subtype":"success","is_error":true,"result":"OAuth token revoked · Please run /login"}'
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is True
    assert message == "OAuth token revoked · Please run /login"


def test_detect_oauth_error_oauth_token_expired():
    """Test detecting OAuth token expired error."""
    stdout = '{"type":"result","is_error":true,"result":"OAuth token expired. Please login again."}'
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is True
    assert message == "OAuth token expired. Please login again."


def test_detect_oauth_error_login_required():
    """Test detecting /login required error."""
    stdout = '{"type":"result","is_error":true,"result":"Authentication failed. Please run /login"}'
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is True
    assert message == "Authentication failed. Please run /login"


def test_detect_oauth_error_no_error():
    """Test that successful responses are not detected as OAuth errors."""
    stdout = '{"type":"result","subtype":"success","is_error":false,"result":"Success!"}'
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is False
    assert message is None


def test_detect_oauth_error_other_error():
    """Test that non-OAuth errors are not detected as OAuth errors."""
    stdout = '{"type":"result","is_error":true,"result":"Some other error occurred"}'
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is False
    assert message is None


def test_detect_oauth_error_empty_stdout():
    """Test handling of empty stdout."""
    stdout = ""
    stderr = "Some error"

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is False
    assert message is None


def test_detect_oauth_error_invalid_json():
    """Test handling of invalid JSON in stdout."""
    stdout = "Not a JSON response"
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is False
    assert message is None


def test_detect_oauth_error_multiline_output():
    """Test handling of multiline stdout (only first line parsed as JSON)."""
    stdout = (
        '{"type":"result","is_error":true,"result":"OAuth token revoked · Please run /login"}\n'
        "Additional output line 1\n"
        "Additional output line 2"
    )
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is True
    assert message == "OAuth token revoked · Please run /login"


def test_detect_oauth_error_error_field():
    """Test detecting OAuth error in the 'error' field instead of 'result'."""
    stdout = '{"type":"result","is_error":true,"error":"OAuth token authentication failed. Please run /login"}'
    stderr = ""

    is_oauth_error, message = detect_oauth_error(stdout, stderr)

    assert is_oauth_error is True
    assert message == "OAuth token authentication failed. Please run /login"


def test_oauth_error_exception_attributes():
    """Test ClaudeOAuthError exception attributes."""
    error = ClaudeOAuthError(
        "OAuth token revoked · Please run /login",
        reauth_instruction="Please run /login"
    )

    assert str(error) == "OAuth token revoked · Please run /login"
    assert error.reauth_instruction == "Please run /login"


def test_oauth_error_exception_default_instruction():
    """Test ClaudeOAuthError with default reauth instruction."""
    error = ClaudeOAuthError("OAuth token expired")

    assert str(error) == "OAuth token expired"
    assert error.reauth_instruction == "Please run /login"


def test_oauth_error_inherits_from_runtime_error():
    """Test that ClaudeOAuthError inherits from RuntimeError."""
    error = ClaudeOAuthError("OAuth error")

    assert isinstance(error, RuntimeError)
    assert isinstance(error, ClaudeOAuthError)


def test_detect_oauth_error_takes_priority_over_rate_limit_pattern():
    """Test that OAuth error is detected even if rate limit pattern is present.

    This validates the fix for the false positive issue: if both OAuth error
    and rate limit text are present, we should detect OAuth (more specific)
    rather than treating it as a rate limit (less specific regex).
    """
    # Response contains both OAuth error AND rate limit text
    stdout = (
        '{"type":"result","is_error":true,"result":"OAuth token revoked · Please run /login. '
        'Note: Your usage limit reached and resets 3PM tomorrow."}'
    )
    stderr = ""

    # Should detect as OAuth error (more specific)
    is_oauth_error, oauth_message = detect_oauth_error(stdout, stderr)
    assert is_oauth_error is True
    assert oauth_message is not None
    assert "OAuth token revoked" in oauth_message

    # Rate limit detection should also find the pattern
    from pydantic_ai_claude_code.utils import detect_rate_limit
    is_rate_limited, reset_time = detect_rate_limit(stdout)
    assert is_rate_limited is True

    # But in the actual error handling flow, OAuth should be checked FIRST
    # and raise immediately, preventing the rate limit wait


def test_detect_oauth_error_vs_rate_limit_priority():
    """Test scenarios to verify OAuth vs rate limit detection priority."""

    # Scenario 1: Only OAuth error
    stdout_oauth_only = '{"type":"result","is_error":true,"result":"OAuth token expired. Please run /login"}'
    is_oauth, _ = detect_oauth_error(stdout_oauth_only, "")
    assert is_oauth is True

    # Scenario 2: Only rate limit
    stdout_rate_limit_only = '{"type":"result","is_error":true,"result":"5-hour limit reached ∙ resets 3PM"}'
    is_oauth, _ = detect_oauth_error(stdout_rate_limit_only, "")
    assert is_oauth is False  # Should NOT be detected as OAuth

    # Scenario 3: Both patterns present (edge case)
    stdout_both = (
        '{"type":"result","is_error":true,"result":'
        '"OAuth token revoked. Your 5-hour limit reached and resets 11PM. Please run /login"}'
    )
    is_oauth, msg = detect_oauth_error(stdout_both, "")
    assert is_oauth is True  # OAuth should be detected
    assert msg is not None
    assert "/login" in msg  # Message should contain OAuth instruction
