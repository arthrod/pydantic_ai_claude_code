"""Tests for utility functions."""

import os
import shutil
from unittest import mock

import pytest

from pydantic_ai_claude_code.types import ClaudeCodeSettings
from pydantic_ai_claude_code.utils import (
    build_claude_command,
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
        assert "Follow the instructions in prompt.md" in cmd


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
    ):
        with pytest.raises(RuntimeError, match="Could not find claude CLI binary"):
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

        # Extra args should come before the prompt instruction
        prompt_idx = cmd.index("Follow the instructions in prompt.md")
        debug_idx = cmd.index("--debug")
        assert debug_idx < prompt_idx


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
        assert "Follow the instructions in prompt.md" in cmd
