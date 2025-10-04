"""Tests for utility functions."""

import json

from pydantic_ai_claude_code.types import ClaudeCodeSettings
from pydantic_ai_claude_code.utils import build_claude_command, parse_stream_json_line


def test_build_claude_command_basic():
    """Test building basic Claude command."""
    cmd = build_claude_command()

    assert "claude" in cmd
    assert "--print" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "Follow the instructions in prompt.md" in cmd


def test_build_claude_command_with_settings():
    """Test building command with settings."""
    settings: ClaudeCodeSettings = {
        "model": "sonnet",
        "max_turns": 5,
        "allowed_tools": ["Read", "Edit"],
        "permission_mode": "acceptEdits",
    }

    cmd = build_claude_command(settings=settings)

    assert "--model" in cmd
    assert "sonnet" in cmd
    assert "--max-turns" in cmd
    assert "5" in cmd
    assert "--allowed-tools" in cmd
    assert "Read" in cmd
    assert "Edit" in cmd
    assert "--permission-mode" in cmd
    assert "acceptEdits" in cmd


def test_build_claude_command_stream_json():
    """Test building command with stream-json output."""
    cmd = build_claude_command(output_format="stream-json")

    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--verbose" in cmd  # Required for stream-json


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
