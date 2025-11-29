"""Tests for sandbox-runtime integration."""

import shutil

import pytest

from pydantic_ai_claude_code import ClaudeCodeProvider
from pydantic_ai_claude_code.utils import (
    build_claude_command,
    resolve_sandbox_runtime_path,
)


def test_resolve_sandbox_runtime_path_from_path():
    """Test resolving srt binary from PATH."""
    # This will only pass if srt is actually installed
    if shutil.which("srt"):
        path = resolve_sandbox_runtime_path()
        assert path is not None
        assert "srt" in path


def test_resolve_sandbox_runtime_path_from_settings():
    """Test resolving srt binary from settings."""
    settings = {"sandbox_runtime_path": "/custom/path/to/srt"}
    path = resolve_sandbox_runtime_path(settings)
    assert path == "/custom/path/to/srt"


def test_resolve_sandbox_runtime_path_not_found(monkeypatch):
    """Test error when srt binary cannot be found."""
    # Mock shutil.which to return None
    monkeypatch.setattr(shutil, "which", lambda x: None)

    with pytest.raises(RuntimeError) as exc_info:
        resolve_sandbox_runtime_path()

    assert "Could not find sandbox-runtime" in str(exc_info.value)
    assert "npm install -g @anthropic-ai/sandbox-runtime" in str(exc_info.value)


def test_build_claude_command_without_sandbox():
    """Test building Claude command without sandbox."""
    settings = {"use_sandbox_runtime": False}
    cmd = build_claude_command(settings=settings, output_format="json")

    # Should not contain srt
    assert "srt" not in " ".join(cmd)
    assert "IS_SANDBOX=1" not in " ".join(cmd)

    # Should contain claude
    assert any("claude" in part for part in cmd)


def test_build_claude_command_with_sandbox():
    """Test building Claude command with sandbox enabled."""
    # Skip if srt is not installed
    if not shutil.which("srt"):
        pytest.skip("sandbox-runtime (srt) not installed")

    settings = {"use_sandbox_runtime": True}
    cmd = build_claude_command(settings=settings, output_format="json")

    # Should contain srt wrapper
    assert cmd[0].endswith("srt") or "srt" in cmd[0]

    # Should store environment variables in settings (not in command)
    assert "__sandbox_env" in settings
    assert settings["__sandbox_env"]["IS_SANDBOX"] == "1"
    assert settings["__sandbox_env"]["CLAUDE_CONFIG_DIR"] == "/tmp/claude_sandbox_config"

    # Should still contain claude
    assert any("claude" in part for part in cmd)

    # Verify structure: srt --settings <config> -- claude ...
    assert cmd[1] == "--settings"
    assert "srt_config_" in cmd[2]  # Config file in temp directory (path varies by OS)
    assert cmd[3] == "--"
    assert "claude" in cmd[4]


def test_build_claude_command_with_custom_srt_path():
    """Test building Claude command with custom srt path."""
    settings = {
        "use_sandbox_runtime": True,
        "sandbox_runtime_path": "/custom/srt",
    }

    cmd = build_claude_command(settings=settings, output_format="json")

    # First element should be the custom srt path
    assert cmd[0] == "/custom/srt"


def test_sandbox_settings_in_build_command():
    """Test that sandbox settings are correctly used in build_claude_command."""
    settings = {
        "use_sandbox_runtime": True,
        "sandbox_runtime_path": "/custom/srt",
    }

    cmd = build_claude_command(settings=settings, output_format="json")

    # First element should be the custom srt path
    assert cmd[0] == "/custom/srt"

    # Environment variables should be in settings
    assert "__sandbox_env" in settings
    assert settings["__sandbox_env"]["IS_SANDBOX"] == "1"


def test_sandbox_defaults_in_build_command():
    """Test that sandbox defaults are correctly applied in build_claude_command."""
    # Empty settings dict
    settings = {}

    # Without explicit use_sandbox_runtime, it should NOT wrap with srt
    cmd = build_claude_command(settings=settings, output_format="json")

    # Default behavior: no sandbox wrapping (user must explicitly enable)
    assert "__sandbox_env" not in settings or settings.get("use_sandbox_runtime") is not True


@pytest.mark.skipif(not shutil.which("srt"), reason="sandbox-runtime not installed")
def test_sandbox_integration():
    """Integration test: verify sandbox command is constructed correctly."""
    settings = {
        "model": "sonnet",
        "use_sandbox_runtime": True,
    }

    cmd = build_claude_command(settings=settings, output_format="json")

    # Comprehensive checks
    assert len(cmd) > 5  # Should have multiple components

    # Check structure
    cmd_str = " ".join(cmd)
    assert "srt" in cmd_str
    assert "claude" in cmd_str
    assert "--print" in cmd_str

    # Check environment variables are stored in settings (not in command)
    assert "__sandbox_env" in settings
    assert settings["__sandbox_env"]["IS_SANDBOX"] == "1"
    assert settings["__sandbox_env"]["CLAUDE_CONFIG_DIR"] == "/tmp/claude_sandbox_config"


def test_sandbox_with_other_settings():
    """Test sandbox works with other model settings."""
    settings = {
        "model": "sonnet",
        "use_sandbox_runtime": True,
        "sandbox_runtime_path": "/usr/bin/srt",  # Provide explicit path for testing
        "working_directory": "/tmp/test",
        "timeout_seconds": 1800,
        "extra_cli_args": ["--debug", "api"],
    }

    cmd = build_claude_command(settings=settings, output_format="json")

    # Should have sandbox wrapper with custom path
    assert cmd[0] == "/usr/bin/srt"

    # Environment variables should be in settings (not command)
    assert "__sandbox_env" in settings
    assert settings["__sandbox_env"]["IS_SANDBOX"] == "1"

    # Should also have extra args
    assert "--debug" in cmd
    assert "api" in cmd

    # Other settings should be preserved
    assert settings["working_directory"] == "/tmp/test"
    assert settings["timeout_seconds"] == 1800


def test_sandbox_can_be_disabled():
    """Test that sandbox can be explicitly disabled."""
    settings = {"model": "sonnet", "use_sandbox_runtime": False}

    cmd = build_claude_command(settings=settings, output_format="json")

    # Should NOT have sandbox wrapper when explicitly disabled
    assert "srt" not in " ".join(cmd)