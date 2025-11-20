"""Tests for core modules (core package)."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from pydantic_ai_claude_code.core import (
    detect_oauth_error,
    detect_rate_limit,
    calculate_wait_time,
    detect_cli_infrastructure_failure,
    get_debug_dir,
    save_prompt_debug,
    save_response_debug,
    save_raw_response_to_working_dir,
    resolve_sandbox_runtime_path,
    build_sandbox_config,
    wrap_command_with_sandbox,
)


class TestOAuthHandler:
    """Tests for oauth_handler module."""

    def test_detect_oauth_error_with_oauth_token_revoked(self):
        """Test detecting OAuth token revoked error."""
        stdout = json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "OAuth token revoked - Please run /login"
        })
        
        is_oauth, message = detect_oauth_error(stdout, "")
        
        assert is_oauth is True
        assert "OAuth token revoked" in message

    def test_detect_oauth_error_with_authentication_failed(self):
        """Test detecting authentication failed error."""
        stdout = json.dumps({
            "type": "result",
            "is_error": True,
            "result": "Authentication failed - please login"
        })
        
        is_oauth, message = detect_oauth_error(stdout, "")
        
        assert is_oauth is True
        assert "Authentication failed" in message

    def test_detect_oauth_error_token_expired(self):
        """Test detecting token expired error."""
        stdout = json.dumps({
            "is_error": True,
            "error": "token expired"
        })
        
        is_oauth, message = detect_oauth_error(stdout, "")
        
        assert is_oauth is True
        assert "token expired" in message

    def test_detect_oauth_error_no_error(self):
        """Test that non-error responses return False."""
        stdout = json.dumps({
            "type": "result",
            "is_error": False,
            "result": "Success"
        })
        
        is_oauth, message = detect_oauth_error(stdout, "")
        
        assert is_oauth is False
        assert message is None

    def test_detect_oauth_error_non_oauth_error(self):
        """Test that non-OAuth errors return False."""
        stdout = json.dumps({
            "type": "result",
            "is_error": True,
            "result": "Some other error occurred"
        })
        
        is_oauth, message = detect_oauth_error(stdout, "")
        
        assert is_oauth is False
        assert message is None

    def test_detect_oauth_error_invalid_json(self):
        """Test handling of invalid JSON."""
        stdout = "not valid json"
        
        is_oauth, message = detect_oauth_error(stdout, "")
        
        assert is_oauth is False
        assert message is None

    def test_detect_oauth_error_empty_stdout(self):
        """Test handling of empty stdout."""
        is_oauth, message = detect_oauth_error("", "")
        
        assert is_oauth is False
        assert message is None


class TestRetryLogic:
    """Tests for retry_logic module."""

    def test_detect_rate_limit_with_time(self):
        """Test detecting rate limit with reset time."""
        error_output = "Rate limit reached, resets 3PM"
        
        is_limited, reset_time = detect_rate_limit(error_output)
        
        assert is_limited is True
        assert reset_time == "3PM"

    def test_detect_rate_limit_case_insensitive(self):
        """Test rate limit detection is case insensitive."""
        error_output = "LIMIT REACHED, RESETS 11AM"
        
        is_limited, reset_time = detect_rate_limit(error_output)
        
        assert is_limited is True
        assert reset_time == "11AM"

    def test_detect_rate_limit_no_rate_limit(self):
        """Test that non-rate-limit errors return False."""
        error_output = "Some other error"
        
        is_limited, reset_time = detect_rate_limit(error_output)
        
        assert is_limited is False
        assert reset_time is None

    def test_calculate_wait_time_future_time(self):
        """Test calculating wait time for future reset time."""
        now = datetime.now()
        future_hour = (now + timedelta(hours=2)).hour
        
        # Convert to 12-hour format with AM/PM
        if future_hour == 0:
            reset_time_str = "12AM"
        elif future_hour < 12:
            reset_time_str = f"{future_hour}AM"
        elif future_hour == 12:
            reset_time_str = "12PM"
        else:
            reset_time_str = f"{future_hour - 12}PM"
        
        wait_seconds = calculate_wait_time(reset_time_str)
        
        # Should be between 2 hours and 2 hours + 1 minute (buffer)
        assert 7200 <= wait_seconds <= 7260

    def test_calculate_wait_time_past_time_same_day(self):
        """Test that past time adds a day."""
        now = datetime.now()
        past_hour = (now - timedelta(hours=2)).hour
        
        # Convert to 12-hour format
        if past_hour == 0:
            reset_time_str = "12AM"
        elif past_hour < 12:
            reset_time_str = f"{past_hour}AM"
        elif past_hour == 12:
            reset_time_str = "12PM"
        else:
            reset_time_str = f"{past_hour - 12}PM"
        
        wait_seconds = calculate_wait_time(reset_time_str)
        
        # Should be close to 22 hours (24 - 2) + buffer
        assert wait_seconds > 79200  # More than 22 hours

    def test_calculate_wait_time_invalid_format(self):
        """Test fallback for invalid time format."""
        wait_seconds = calculate_wait_time("invalid")
        
        # Should fallback to 5 minutes
        assert wait_seconds == 300

    def test_detect_cli_infrastructure_failure_module_not_found(self):
        """Test detecting module not found errors."""
        stderr = "Error: Cannot find module 'yoga.wasm'"
        
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_detect_cli_infrastructure_failure_module_not_found_uppercase(self):
        """Test detecting MODULE_NOT_FOUND error."""
        stderr = "MODULE_NOT_FOUND: Cannot resolve module"
        
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_detect_cli_infrastructure_failure_enoent(self):
        """Test detecting ENOENT errors."""
        stderr = "Error: ENOENT: no such file or directory"
        
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_detect_cli_infrastructure_failure_eacces(self):
        """Test detecting EACCES errors."""
        stderr = "Error: EACCES: permission denied"
        
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_detect_cli_infrastructure_failure_no_error(self):
        """Test that non-infrastructure errors return False."""
        stderr = "Some other error"
        
        assert detect_cli_infrastructure_failure(stderr) is False


class TestDebugSaver:
    """Tests for debug_saver module."""

    def test_get_debug_dir_enabled_default_path(self, tmp_path):
        """Test getting debug directory when enabled with True."""
        settings = {"debug_save_prompts": True}
        
        with patch("pydantic_ai_claude_code.core.debug_saver.Path") as mock_path:
            mock_path.return_value.mkdir = Mock()
            result = get_debug_dir(settings)
            
            assert result is not None

    def test_get_debug_dir_custom_path(self, tmp_path):
        """Test getting debug directory with custom path."""
        custom_path = str(tmp_path / "custom_debug")
        settings = {"debug_save_prompts": custom_path}
        
        result = get_debug_dir(settings)
        
        assert result == Path(custom_path)
        assert result.exists()

    def test_get_debug_dir_disabled(self):
        """Test that None is returned when debug is disabled."""
        settings = {"debug_save_prompts": False}
        
        result = get_debug_dir(settings)
        
        assert result is None

    def test_get_debug_dir_no_settings(self):
        """Test that None is returned with no settings."""
        result = get_debug_dir(None)
        
        assert result is None

    def test_save_prompt_debug_creates_file(self, tmp_path):
        """Test saving prompt to debug file."""
        debug_dir = tmp_path / "debug"
        settings = {"debug_save_prompts": str(debug_dir)}
        prompt = "Test prompt content"
        
        save_prompt_debug(prompt, settings)
        
        # Check that a file was created
        files = list(debug_dir.glob("*_prompt.md"))
        assert len(files) == 1
        assert files[0].read_text() == prompt

    def test_save_prompt_debug_increments_counter(self, tmp_path):
        """Test that prompt files are numbered sequentially."""
        debug_dir = tmp_path / "debug"
        settings = {"debug_save_prompts": str(debug_dir)}
        
        save_prompt_debug("First prompt", settings)
        save_prompt_debug("Second prompt", settings)
        
        files = sorted(debug_dir.glob("*_prompt.md"))
        assert len(files) == 2
        # Files should have different numbers
        assert files[0].name.startswith("001_") or files[0].name.startswith("002_")

    def test_save_response_debug_creates_file(self, tmp_path):
        """Test saving response to debug file."""
        debug_dir = tmp_path / "debug"
        settings = {"debug_save_prompts": str(debug_dir)}
        response = {"result": "test", "type": "success"}
        
        # Need to save prompt first to increment counter
        save_prompt_debug("test", settings)
        save_response_debug(response, settings)
        
        # Check that response file was created
        files = list(debug_dir.glob("*_response.json"))
        assert len(files) == 1
        
        saved_response = json.loads(files[0].read_text())
        assert saved_response == response

    def test_save_raw_response_to_working_dir(self, tmp_path):
        """Test saving raw response to working directory."""
        response_file = tmp_path / "response.json"
        settings = {"__response_file_path": str(response_file)}
        response = {"data": "test response"}
        
        save_raw_response_to_working_dir(response, settings)
        
        assert response_file.exists()
        saved = json.loads(response_file.read_text())
        assert saved == response

    def test_save_raw_response_no_path_configured(self):
        """Test that no error occurs when response path not configured."""
        settings = {}
        response = {"data": "test"}
        
        # Should not raise an error
        save_raw_response_to_working_dir(response, settings)


class TestSandboxRuntime:
    """Tests for sandbox_runtime module."""

    def test_resolve_sandbox_runtime_path_from_settings(self):
        """Test resolving srt path from settings."""
        settings = {"sandbox_runtime_path": "/custom/path/to/srt"}
        
        result = resolve_sandbox_runtime_path(settings)
        
        assert result == "/custom/path/to/srt"

    def test_resolve_sandbox_runtime_path_from_env(self):
        """Test resolving srt path from environment variable."""
        with patch.dict(os.environ, {"SANDBOX_RUNTIME_PATH": "/env/path/srt"}):
            result = resolve_sandbox_runtime_path()
            
            assert result == "/env/path/srt"

    def test_resolve_sandbox_runtime_path_from_which(self):
        """Test resolving srt path from PATH."""
        with patch("shutil.which", return_value="/usr/local/bin/srt"):
            result = resolve_sandbox_runtime_path()
            
            assert result == "/usr/local/bin/srt"

    def test_resolve_sandbox_runtime_path_not_found(self):
        """Test error when srt cannot be found."""
        with patch("shutil.which", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(RuntimeError, match="Could not find sandbox-runtime"):
                    resolve_sandbox_runtime_path()

    def test_build_sandbox_config_structure(self):
        """Test sandbox configuration structure."""
        config = build_sandbox_config()
        
        assert "permissions" in config
        assert "allow" in config["permissions"]
        assert isinstance(config["permissions"]["allow"], list)
        
        # Check for required permissions
        allow_list = config["permissions"]["allow"]
        assert any("Bash" in perm for perm in allow_list)
        assert any("/tmp" in perm for perm in allow_list)
        assert any("api.anthropic.com" in perm for perm in allow_list)

    def test_wrap_command_with_sandbox_basic(self):
        """Test wrapping command with sandbox."""
        cmd = ["claude", "--print"]
        settings = {"sandbox_runtime_path": "/usr/bin/srt"}
        
        with patch("tempfile.mkstemp", return_value=(99, "/tmp/config.json")):
            with patch("os.fdopen", mock_open()):
                with patch("os.makedirs"):
                    wrapped_cmd, env = wrap_command_with_sandbox(cmd, settings)
        
        assert wrapped_cmd[0] == "/usr/bin/srt"
        assert "--settings" in wrapped_cmd
        assert "--" in wrapped_cmd
        assert "claude" in wrapped_cmd
        assert env["IS_SANDBOX"] == "1"
        assert "CLAUDE_CONFIG_DIR" in env

    def test_wrap_command_with_sandbox_preserves_args(self):
        """Test that original command arguments are preserved."""
        cmd = ["claude", "--print", "--model", "sonnet", "test.md"]
        settings = {"sandbox_runtime_path": "/usr/bin/srt"}
        
        with patch("tempfile.mkstemp", return_value=(99, "/tmp/config.json")):
            with patch("os.fdopen", mock_open()):
                with patch("os.makedirs"):
                    wrapped_cmd, _ = wrap_command_with_sandbox(cmd, settings)
        
        # Find the position of "--" separator
        separator_idx = wrapped_cmd.index("--")
        
        # Everything after "--" should be the original command
        original_part = wrapped_cmd[separator_idx + 1:]
        assert original_part == cmd

    def test_wrap_command_with_sandbox_copies_credentials(self, tmp_path):
        """Test that credentials are copied to sandbox config dir."""
        cmd = ["claude", "--print"]
        settings = {"sandbox_runtime_path": "/usr/bin/srt"}
        
        # Create mock credentials
        home_claude_dir = tmp_path / "home" / ".claude"
        home_claude_dir.mkdir(parents=True)
        credentials = home_claude_dir / ".credentials.json"
        credentials.write_text('{"token": "test"}')
        
        with patch("tempfile.mkstemp", return_value=(99, "/tmp/config.json")):
            with patch("os.fdopen", mock_open()):
                with patch("pydantic_ai_claude_code.core.sandbox_runtime.Path.home", return_value=tmp_path / "home"):
                    with patch("os.makedirs"):
                        with patch("shutil.copy2") as mock_copy:
                            _, env = wrap_command_with_sandbox(cmd, settings)
                            
                            # Should have attempted to copy credentials
                            assert mock_copy.called