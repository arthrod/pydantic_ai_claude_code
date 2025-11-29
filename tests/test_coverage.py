"""Comprehensive unit tests to increase code coverage.

This test file covers helper functions, error cases, and edge cases
WITHOUT making actual Claude CLI calls.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from pydantic_ai.messages import (
    ModelRequest,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from pydantic_ai_claude_code.exceptions import ClaudeOAuthError
from pydantic_ai_claude_code.model import ClaudeCodeModel
from pydantic_ai_claude_code.streaming import (
    _extract_from_assistant,
    _extract_from_content_block_delta,
    _extract_from_result,
    extract_text_from_stream_event,
)
from pydantic_ai_claude_code.streamed_response import ClaudeCodeStreamedResponse
from pydantic_ai_claude_code.types import ClaudeCodeSettings, ClaudeJSONResponse
from pydantic_ai_claude_code.utils import (
    _check_rate_limit,
    _classify_execution_error,
    _copy_additional_files,
    _format_cli_error_message,
    _get_debug_dir,
    _get_next_call_subdirectory,
    _handle_command_failure,
    _parse_json_response,
    _save_prompt_debug,
    _save_raw_response_to_working_dir,
    _save_response_debug,
    _validate_claude_response,
    build_claude_command,
    calculate_wait_time,
    convert_primitive_value,
    detect_cli_infrastructure_failure,
    detect_oauth_error,
    detect_rate_limit,
    resolve_sandbox_runtime_path,
    strip_markdown_code_fence,
)


# ===== Tests for utils.py =====


class TestConvertPrimitiveValue:
    """Tests for convert_primitive_value function."""

    def test_convert_integer(self):
        """Test integer conversion."""
        assert convert_primitive_value("42", "integer") == 42
        assert convert_primitive_value("-10", "integer") == -10

    def test_convert_number_float(self):
        """Test number conversion to float."""
        assert convert_primitive_value("3.14", "number") == 3.14
        assert convert_primitive_value("1e5", "number") == 100000.0
        assert convert_primitive_value("1E-2", "number") == 0.01

    def test_convert_number_integer(self):
        """Test number conversion to integer when no decimal."""
        assert convert_primitive_value("42", "number") == 42

    def test_convert_boolean(self):
        """Test boolean conversion."""
        assert convert_primitive_value("true", "boolean") is True
        assert convert_primitive_value("TRUE", "boolean") is True
        assert convert_primitive_value("1", "boolean") is True
        assert convert_primitive_value("yes", "boolean") is True
        assert convert_primitive_value("false", "boolean") is False
        assert convert_primitive_value("no", "boolean") is False

    def test_convert_string(self):
        """Test string conversion."""
        assert convert_primitive_value("hello", "string") == "hello"
        assert convert_primitive_value("123", "string") == "123"

    def test_convert_invalid(self):
        """Test invalid conversion returns None."""
        assert convert_primitive_value("not a number", "integer") is None
        # Unknown field type returns None
        assert convert_primitive_value("value", "unknown_type") is None


class TestStripMarkdownCodeFence:
    """Tests for strip_markdown_code_fence function."""

    def test_strip_json_fence(self):
        """Test stripping ```json fence."""
        text = '```json\n{"key": "value"}\n```'
        result = strip_markdown_code_fence(text)
        assert result == '{"key": "value"}'

    def test_strip_plain_fence(self):
        """Test stripping plain ``` fence."""
        text = "```\nsome code\n```"
        result = strip_markdown_code_fence(text)
        assert result == "some code"

    def test_no_fence(self):
        """Test text without fence."""
        text = '{"key": "value"}'
        result = strip_markdown_code_fence(text)
        assert result == '{"key": "value"}'

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        text = "  ```json\n  data  \n```  "
        result = strip_markdown_code_fence(text)
        assert result == "data"


class TestFormatCliErrorMessage:
    """Tests for _format_cli_error_message function."""

    def test_short_runtime_error(self):
        """Test error message for short runtime."""
        msg = _format_cli_error_message(10.5, 1, "Some error")
        assert "10.5s" in msg
        assert "Some error" in msg
        assert "Long runtime" not in msg

    def test_long_runtime_error(self):
        """Test error message for long runtime includes complexity hint."""
        msg = _format_cli_error_message(650.0, 1, "Timeout")
        assert "650.0s" in msg
        assert "Long runtime" in msg
        assert "breaking into smaller tasks" in msg


class TestResolveSandboxRuntimePath:
    """Tests for resolve_sandbox_runtime_path function."""

    def test_from_settings(self):
        """Test resolution from settings."""
        settings: ClaudeCodeSettings = {
            "sandbox_runtime_path": "/custom/srt",
        }
        path = resolve_sandbox_runtime_path(settings)
        assert path == "/custom/srt"

    def test_from_env(self):
        """Test resolution from environment variable."""
        with mock.patch.dict(os.environ, {"SANDBOX_RUNTIME_PATH": "/env/srt"}):
            path = resolve_sandbox_runtime_path({})
            assert path == "/env/srt"

    def test_from_which(self):
        """Test resolution via shutil.which."""
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("shutil.which", return_value="/usr/bin/srt"),
        ):
            path = resolve_sandbox_runtime_path()
            assert path == "/usr/bin/srt"

    def test_not_found(self):
        """Test error when srt not found."""
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="Could not find sandbox-runtime"),
        ):
            resolve_sandbox_runtime_path()


class TestDetectRateLimit:
    """Tests for detect_rate_limit function."""

    def test_detects_rate_limit(self):
        """Test detection of rate limit pattern."""
        error = "5-hour limit reached ∙ resets 3PM"
        is_limited, reset_time = detect_rate_limit(error)
        assert is_limited is True
        assert reset_time == "3PM"

    def test_no_rate_limit(self):
        """Test non-rate-limit error."""
        error = "Some other error"
        is_limited, reset_time = detect_rate_limit(error)
        assert is_limited is False
        assert reset_time is None


class TestCalculateWaitTime:
    """Tests for calculate_wait_time function."""

    def test_calculate_future_time(self):
        """Test wait time calculation for future time."""
        # This will vary based on current time, but should return positive
        wait_seconds = calculate_wait_time("11PM")
        assert wait_seconds >= 0

    def test_invalid_time_format(self):
        """Test fallback for invalid time format."""
        wait_seconds = calculate_wait_time("invalid")
        assert wait_seconds == 300  # 5 minute fallback


class TestDetectCliInfrastructureFailure:
    """Tests for detect_cli_infrastructure_failure function."""

    def test_detects_module_not_found(self):
        """Test detection of missing module error."""
        stderr = "Error: Cannot find module 'yoga.wasm'"
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_detects_module_not_found_code(self):
        """Test detection of MODULE_NOT_FOUND error code."""
        stderr = "MODULE_NOT_FOUND: @anthropic-ai/claude-code"
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_detects_enoent(self):
        """Test detection of ENOENT error."""
        stderr = "ENOENT: no such file or directory"
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_detects_eacces(self):
        """Test detection of EACCES error."""
        stderr = "EACCES: permission denied"
        assert detect_cli_infrastructure_failure(stderr) is True

    def test_no_infrastructure_failure(self):
        """Test normal error is not detected as infrastructure failure."""
        stderr = "Error: Invalid response format"
        assert detect_cli_infrastructure_failure(stderr) is False


class TestCheckRateLimit:
    """Tests for _check_rate_limit function."""

    def test_rate_limit_with_retry_enabled(self):
        """Test rate limit detection when retry is enabled."""
        stdout = "5-hour limit reached ∙ resets 3PM"
        stderr = ""
        should_retry, wait_secs = _check_rate_limit(stdout, stderr, 1, True)
        assert should_retry is True
        assert wait_secs >= 0

    def test_rate_limit_with_retry_disabled(self):
        """Test rate limit not retried when disabled."""
        stdout = "5-hour limit reached ∙ resets 3PM"
        stderr = ""
        should_retry, wait_secs = _check_rate_limit(stdout, stderr, 1, False)
        assert should_retry is False
        assert wait_secs == 0

    def test_no_rate_limit(self):
        """Test no retry for non-rate-limit error."""
        stdout = "Some error"
        stderr = ""
        should_retry, wait_secs = _check_rate_limit(stdout, stderr, 1, True)
        assert should_retry is False
        assert wait_secs == 0


class TestHandleCommandFailure:
    """Tests for _handle_command_failure function."""

    def test_raises_runtime_error(self):
        """Test that command failure raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Some error"):
            _handle_command_failure(
                stdout_text="",
                stderr_text="Some error",
                returncode=1,
                elapsed=5.0,
                prompt_len=100,
                cwd="/tmp/test",
            )


class TestClassifyExecutionError:
    """Tests for _classify_execution_error function."""

    def test_oauth_error_raises_immediately(self):
        """Test OAuth error raises ClaudeOAuthError."""
        stdout = '{"type":"result","is_error":true,"result":"OAuth token revoked · Please run /login"}'
        with pytest.raises(ClaudeOAuthError, match="OAuth"):
            _classify_execution_error(stdout, "", 1, 5.0, True, "/tmp")

    def test_rate_limit_returns_retry(self):
        """Test rate limit returns retry action."""
        stdout = "5-hour limit reached ∙ resets 3PM"
        action, wait_secs = _classify_execution_error(stdout, "", 1, 5.0, True, "/tmp")
        assert action == "retry_rate_limit"
        assert wait_secs >= 0

    def test_infrastructure_failure_returns_retry(self):
        """Test infrastructure failure returns retry action."""
        stderr = "Cannot find module 'yoga.wasm'"
        action, _ = _classify_execution_error("", stderr, 1, 5.0, True, "/tmp")
        assert action == "retry_infra"

    def test_generic_error_raises(self):
        """Test generic error raises RuntimeError."""
        with pytest.raises(RuntimeError):
            _classify_execution_error("", "Generic error", 1, 5.0, True, "/tmp")


class TestParseJsonResponse:
    """Tests for _parse_json_response function."""

    def test_parse_simple_json(self):
        """Test parsing simple JSON response."""
        raw = '{"type":"result","result":"hello"}'
        response = _parse_json_response(raw)
        assert response["result"] == "hello"

    def test_parse_verbose_json(self):
        """Test parsing verbose JSON array response."""
        raw = '[{"type":"other"},{"type":"result","result":"hello"}]'
        response = _parse_json_response(raw)
        assert response["result"] == "hello"

    def test_parse_with_srt_diagnostic(self):
        """Test stripping srt diagnostic output."""
        raw = 'Running: /usr/bin/claude --print\n{"type":"result","result":"hello"}'
        response = _parse_json_response(raw)
        assert response["result"] == "hello"

    def test_no_result_event_in_verbose(self):
        """Test error when no result event in verbose output."""
        raw = '[{"type":"other"},{"type":"another"}]'
        with pytest.raises(RuntimeError, match="No result event"):
            _parse_json_response(raw)


class TestValidateClaudeResponse:
    """Tests for _validate_claude_response function."""

    def test_valid_response(self):
        """Test valid response passes validation."""
        response: ClaudeJSONResponse = {
            "type": "result",
            "result": "success",
            "is_error": False,
        }
        # Should not raise
        _validate_claude_response(response)

    def test_error_response_raises(self):
        """Test error response raises RuntimeError."""
        response: ClaudeJSONResponse = {
            "type": "result",
            "is_error": True,
            "error": "Something went wrong",
        }
        with pytest.raises(RuntimeError, match="Something went wrong"):
            _validate_claude_response(response)


class TestDebugSaving:
    """Tests for debug saving functions."""

    def test_get_debug_dir_disabled(self):
        """Test get_debug_dir returns None when disabled."""
        settings: ClaudeCodeSettings = {}
        assert _get_debug_dir(settings) is None

    def test_get_debug_dir_default_path(self):
        """Test get_debug_dir uses default path."""
        settings: ClaudeCodeSettings = {"debug_save_prompts": True}
        debug_dir = _get_debug_dir(settings)
        assert debug_dir == Path("/tmp/claude_debug")

    def test_get_debug_dir_custom_path(self):
        """Test get_debug_dir uses custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = f"{tmpdir}/custom_debug"
            settings: ClaudeCodeSettings = {"debug_save_prompts": custom_path}
            debug_dir = _get_debug_dir(settings)
            assert debug_dir == Path(custom_path)

    def test_save_prompt_debug(self):
        """Test saving prompt to debug file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {"debug_save_prompts": tmpdir}
            _save_prompt_debug("Test prompt", settings)

            # Check file was created
            files = list(Path(tmpdir).glob("*_prompt.md"))
            assert len(files) >= 1
            assert "Test prompt" in files[0].read_text()

    def test_save_response_debug(self):
        """Test saving response to debug file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {"debug_save_prompts": tmpdir}
            response: ClaudeJSONResponse = {"type": "result", "result": "test"}

            # First save a prompt to increment counter
            _save_prompt_debug("Test", settings)
            _save_response_debug(response, settings)

            # Check file was created
            files = list(Path(tmpdir).glob("*_response.json"))
            assert len(files) >= 1

    def test_save_raw_response_no_settings(self):
        """Test save raw response with no settings."""
        # Should not raise
        _save_raw_response_to_working_dir({"type": "result"}, None)

    def test_save_raw_response_no_path(self):
        """Test save raw response with no path configured."""
        settings: ClaudeCodeSettings = {}
        # Should not raise
        _save_raw_response_to_working_dir({"type": "result"}, settings)

    def test_save_raw_response_success(self):
        """Test successful save of raw response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            response_path = f"{tmpdir}/response.json"
            settings: ClaudeCodeSettings = {"__response_file_path": response_path}
            response: ClaudeJSONResponse = {"type": "result", "result": "test"}

            _save_raw_response_to_working_dir(response, settings)

            assert Path(response_path).exists()
            saved = json.loads(Path(response_path).read_text())
            assert saved["result"] == "test"


class TestCopyAdditionalFiles:
    """Tests for _copy_additional_files function."""

    def test_copy_file_success(self):
        """Test successful file copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            source = Path(tmpdir) / "source.txt"
            source.write_text("test content")

            # Create destination directory
            dest_dir = Path(tmpdir) / "dest"
            dest_dir.mkdir()

            _copy_additional_files(str(dest_dir), {"copied.txt": source})

            dest_file = dest_dir / "copied.txt"
            assert dest_file.exists()
            assert dest_file.read_text() == "test content"

    def test_copy_file_not_found(self):
        """Test error when source file not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                _copy_additional_files(
                    tmpdir,
                    {"dest.txt": Path("/nonexistent/file.txt")},
                )

    def test_copy_directory_error(self):
        """Test error when source is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source_dir"
            source_dir.mkdir()

            dest_dir = Path(tmpdir) / "dest"
            dest_dir.mkdir()

            with pytest.raises(ValueError, match="not a file"):
                _copy_additional_files(str(dest_dir), {"dest": source_dir})


class TestGetNextCallSubdirectory:
    """Tests for _get_next_call_subdirectory function."""

    def test_first_subdirectory(self):
        """Test creating first subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = _get_next_call_subdirectory(tmpdir)
            assert subdir == Path(tmpdir) / "1"
            assert subdir.exists()

    def test_subsequent_subdirectories(self):
        """Test creating subsequent subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first few manually
            (Path(tmpdir) / "1").mkdir()
            (Path(tmpdir) / "2").mkdir()

            subdir = _get_next_call_subdirectory(tmpdir)
            assert subdir == Path(tmpdir) / "3"


class TestBuildClaudeCommandSandbox:
    """Tests for build_claude_command with sandbox enabled."""

    def test_sandbox_wrapping(self):
        """Test command wrapping with sandbox-runtime."""
        settings: ClaudeCodeSettings = {
            "use_sandbox_runtime": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake home directory with no credentials
            fake_home = Path(tmpdir) / "home"
            fake_home.mkdir()

            with (
                mock.patch("shutil.which") as mock_which,
                mock.patch("pydantic_ai_claude_code.utils.Path.home", return_value=fake_home),
            ):
                # Setup mocks
                mock_which.side_effect = lambda x: f"/usr/bin/{x}" if x in ("claude", "srt") else None

                cmd = build_claude_command(settings=settings)

                # Command should start with srt
                assert cmd[0] == "/usr/bin/srt"
                assert "--settings" in cmd
                assert "--" in cmd


# ===== Tests for streaming.py =====


class TestStreamingTextExtraction:
    """Tests for streaming text extraction functions."""

    def test_extract_from_content_block_delta(self):
        """Test extracting text from content_block_delta event."""
        event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        text = _extract_from_content_block_delta(event)
        assert text == "Hello"

    def test_extract_from_content_block_delta_no_text(self):
        """Test no text extracted from non-text delta."""
        event = {
            "type": "content_block_delta",
            "delta": {"type": "other"},
        }
        text = _extract_from_content_block_delta(event)
        assert text is None

    def test_extract_from_content_block_delta_empty_text(self):
        """Test empty text returns None."""
        event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": ""},
        }
        text = _extract_from_content_block_delta(event)
        assert text is None

    def test_extract_from_assistant(self):
        """Test extracting text from assistant event."""
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "World"},
                ]
            },
        }
        text = _extract_from_assistant(event)
        assert text == "Hello World"

    def test_extract_from_assistant_no_message(self):
        """Test no text when message is not a dict."""
        event = {"type": "assistant", "message": "not a dict"}
        text = _extract_from_assistant(event)
        assert text is None

    def test_extract_from_assistant_no_content(self):
        """Test no text when content is not a list."""
        event = {"type": "assistant", "message": {"content": "not a list"}}
        text = _extract_from_assistant(event)
        assert text is None

    def test_extract_from_result(self):
        """Test extracting text from result event."""
        event = {"type": "result", "result": "Final result"}
        text = _extract_from_result(event)
        assert text == "Final result"

    def test_extract_from_result_no_result(self):
        """Test no text when result is not a string."""
        event = {"type": "result", "result": 123}
        text = _extract_from_result(event)
        assert text is None

    def test_extract_text_from_stream_event(self):
        """Test main extraction function routing."""
        # content_block_delta
        event1 = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        assert extract_text_from_stream_event(event1) == "Hello"

        # assistant
        event2 = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "World"}]},
        }
        assert extract_text_from_stream_event(event2) == "World"

        # result
        event3 = {"type": "result", "result": "Done"}
        assert extract_text_from_stream_event(event3) == "Done"

        # unknown
        event4 = {"type": "unknown"}
        assert extract_text_from_stream_event(event4) is None


# ===== Tests for streamed_response.py =====


class TestClaudeCodeStreamedResponse:
    """Tests for ClaudeCodeStreamedResponse class."""

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test basic properties."""
        async def empty_stream():
            return
            yield  # Make it an async generator

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="claude-code:sonnet",
            event_stream=empty_stream(),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        assert response.model_name == "claude-code:sonnet"
        assert response.provider_name == "claude-code"
        assert response.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Wait for background task to complete
        await response._stream_complete.wait()

    @pytest.mark.asyncio
    async def test_handle_assistant_event_first_chunk(self):
        """Test handling first assistant event chunk."""
        async def empty_stream():
            return
            yield

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=empty_stream(),
        )

        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
        }
        result, text_started, full_text = response._handle_assistant_event(
            event, False, ""
        )

        assert result is not None
        assert text_started is True
        assert full_text == "Hello"

        # Wait for background task
        await response._stream_complete.wait()

    @pytest.mark.asyncio
    async def test_handle_assistant_event_delta(self):
        """Test handling subsequent assistant event with delta."""
        async def empty_stream():
            return
            yield

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=empty_stream(),
        )

        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello World"}]},
        }
        result, text_started, full_text = response._handle_assistant_event(
            event, True, "Hello"
        )

        assert result is not None
        assert full_text == "Hello World"

        # Wait for background task
        await response._stream_complete.wait()

    @pytest.mark.asyncio
    async def test_handle_result_event(self):
        """Test handling result event."""
        async def empty_stream():
            return
            yield

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=empty_stream(),
        )

        event = {
            "type": "result",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 5,
            },
        }
        result = response._handle_result_event(event, 5)

        assert result is not None
        assert response._usage.input_tokens == 100
        assert response._usage.output_tokens == 50

        # Wait for background task
        await response._stream_complete.wait()

    @pytest.mark.asyncio
    async def test_process_marker_and_text_find_marker(self):
        """Test processing text with marker detection."""
        async def empty_stream():
            return
            yield

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=empty_stream(),
            streaming_marker="<<<START>>>",
        )

        accumulated, started, text_started = response._process_marker_and_text(
            "prefix<<<START>>>content", "", False, False
        )

        assert started is True
        assert "content" in accumulated

        # Wait for background task
        await response._stream_complete.wait()

    @pytest.mark.asyncio
    async def test_process_marker_and_text_after_started(self):
        """Test processing text after streaming started."""
        async def empty_stream():
            return
            yield

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=empty_stream(),
        )

        accumulated, started, text_started = response._process_marker_and_text(
            "more text", "initial", True, False
        )

        assert started is True
        assert text_started is True
        assert len(response._buffered_events) > 0

        # Wait for background task
        await response._stream_complete.wait()

    @pytest.mark.asyncio
    async def test_handle_assistant_event_no_content(self):
        """Test handling assistant event with no content."""
        async def empty_stream():
            return
            yield

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=empty_stream(),
        )

        # Test with message that's not a dict
        event = {"type": "assistant", "message": "not a dict"}
        result, text_started, full_text = response._handle_assistant_event(
            event, False, ""
        )
        assert result is None
        assert text_started is False

        # Test with content that's not a list
        event = {"type": "assistant", "message": {"content": "not a list"}}
        result, text_started, full_text = response._handle_assistant_event(
            event, False, ""
        )
        assert result is None

        # Wait for background task
        await response._stream_complete.wait()


# ===== Tests for model.py =====


class TestClaudeCodeModelInit:
    """Tests for ClaudeCodeModel initialization."""

    def test_basic_init(self):
        """Test basic model initialization."""
        model = ClaudeCodeModel("sonnet")
        assert model._model_alias == "sonnet"
        assert model.model_name == "claude-code:sonnet"
        assert model.system == "claude-code"

    def test_init_with_provider_preset(self):
        """Test initialization with provider preset."""
        model = ClaudeCodeModel("sonnet", provider_preset="deepseek")
        assert model._provider_preset_id == "deepseek"
        assert "deepseek" in model.model_name

    def test_init_with_cli_path(self):
        """Test initialization with custom CLI path."""
        model = ClaudeCodeModel("sonnet", cli_path="/custom/claude")
        assert model._cli_path == "/custom/claude"


class TestClaudeCodeModelBuildOptions:
    """Tests for _build_options method."""

    def test_build_options_basic(self):
        """Test building basic options."""
        model = ClaudeCodeModel("sonnet")
        settings = model._build_options(None, ModelRequestParameters())

        assert settings["model"] == "sonnet"
        assert settings["dangerously_skip_permissions"] is True
        assert settings["use_temp_workspace"] is True

    def test_build_options_with_model_settings(self):
        """Test building options with model settings."""
        model = ClaudeCodeModel("sonnet")
        model_settings = {
            "working_directory": "/custom/dir",
            "timeout_seconds": 1200,
            "debug_save_prompts": True,
        }
        settings = model._build_options(model_settings, ModelRequestParameters())

        assert settings["working_directory"] == "/custom/dir"
        assert settings["timeout_seconds"] == 1200
        assert settings["debug_save_prompts"] is True

    def test_build_options_with_hooks(self):
        """Test building options with hooks."""
        model = ClaudeCodeModel("sonnet")
        model_settings = {
            "hooks": [{"matcher": {"event": "tool_use"}, "commands": ["echo $TOOL"]}],
        }
        settings = model._build_options(model_settings, ModelRequestParameters())

        assert "__hooks__" in settings
        assert len(settings["__hooks__"]) == 1


class TestClaudeCodeModelCheckToolResults:
    """Tests for _check_has_tool_results method."""

    def test_no_tool_results(self):
        """Test detection of no tool results."""
        model = ClaudeCodeModel("sonnet")
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ]
        assert model._check_has_tool_results(messages) is False

    def test_has_tool_results(self):
        """Test detection of tool results."""
        model = ClaudeCodeModel("sonnet")
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content="Hello"),
                ToolReturnPart(tool_name="test", content="result", tool_call_id="123"),
            ])
        ]
        assert model._check_has_tool_results(messages) is True


class TestClaudeCodeModelXmlToMarkdown:
    """Tests for _xml_to_markdown method."""

    def test_convert_summary(self):
        """Test converting XML summary to markdown."""
        xml = "<summary>This is a test</summary>"
        result = ClaudeCodeModel._xml_to_markdown(xml)
        assert "This is a test" in result

    def test_convert_with_returns(self):
        """Test converting XML with returns section."""
        xml = "<summary>Do something</summary><returns><description>The result</description></returns>"
        result = ClaudeCodeModel._xml_to_markdown(xml)
        assert "Do something" in result
        assert "Returns: The result" in result


class TestClaudeCodeModelJsonExtraction:
    """Tests for JSON extraction methods."""

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown code fence."""
        model = ClaudeCodeModel("sonnet")
        text = '```json\n{"key": "value"}\n```'
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        result = model._extract_json_robust(text, schema)
        assert result["key"] == "value"

    def test_extract_json_object(self):
        """Test extracting JSON object from text."""
        model = ClaudeCodeModel("sonnet")
        text = 'Some text {"key": "value"} more text'
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        result = model._extract_json_robust(text, schema)
        assert result["key"] == "value"

    def test_extract_json_array_wrap(self):
        """Test extracting and wrapping JSON array."""
        model = ClaudeCodeModel("sonnet")
        text = '["a", "b", "c"]'
        schema = {"type": "object", "properties": {"items": {"type": "array"}}}

        result = model._extract_json_robust(text, schema)
        assert "items" in result
        assert result["items"] == ["a", "b", "c"]

    def test_extract_single_field_autowrap(self):
        """Test auto-wrapping single field value."""
        model = ClaudeCodeModel("sonnet")
        text = "42"
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}

        result = model._extract_json_robust(text, schema)
        assert result["count"] == 42

    def test_extract_json_fails(self):
        """Test extraction failure raises JSONDecodeError."""
        model = ClaudeCodeModel("sonnet")
        text = "not valid json or value"
        schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}

        with pytest.raises(json.JSONDecodeError):
            model._extract_json_robust(text, schema)


class TestClaudeCodeModelSchemaValidation:
    """Tests for schema validation methods."""

    def test_validate_missing_required(self):
        """Test validation catches missing required fields."""
        model = ClaudeCodeModel("sonnet")
        data = {"name": "test"}
        schema = {
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        error = model._validate_json_schema(data, schema)
        assert error is not None
        assert "age" in error

    def test_validate_wrong_type_string(self):
        """Test validation catches wrong string type."""
        model = ClaudeCodeModel("sonnet")
        data = {"name": 123}
        schema = {
            "properties": {"name": {"type": "string"}},
        }

        error = model._validate_json_schema(data, schema)
        assert error is not None
        assert "string" in error

    def test_validate_wrong_type_integer(self):
        """Test validation catches wrong integer type."""
        model = ClaudeCodeModel("sonnet")
        data = {"count": "not a number"}
        schema = {
            "properties": {"count": {"type": "integer"}},
        }

        error = model._validate_json_schema(data, schema)
        assert error is not None
        assert "integer" in error

    def test_validate_wrong_type_number(self):
        """Test validation catches wrong number type."""
        model = ClaudeCodeModel("sonnet")
        data = {"value": "not a number"}
        schema = {
            "properties": {"value": {"type": "number"}},
        }

        error = model._validate_json_schema(data, schema)
        assert error is not None
        assert "number" in error

    def test_validate_wrong_type_boolean(self):
        """Test validation catches wrong boolean type."""
        model = ClaudeCodeModel("sonnet")
        data = {"flag": "not a bool"}
        schema = {
            "properties": {"flag": {"type": "boolean"}},
        }

        error = model._validate_json_schema(data, schema)
        assert error is not None
        assert "boolean" in error

    def test_validate_wrong_type_array(self):
        """Test validation catches wrong array type."""
        model = ClaudeCodeModel("sonnet")
        data = {"items": "not an array"}
        schema = {
            "properties": {"items": {"type": "array"}},
        }

        error = model._validate_json_schema(data, schema)
        assert error is not None
        assert "array" in error

    def test_validate_wrong_type_object(self):
        """Test validation catches wrong object type."""
        model = ClaudeCodeModel("sonnet")
        data = {"config": "not an object"}
        schema = {
            "properties": {"config": {"type": "object"}},
        }

        error = model._validate_json_schema(data, schema)
        assert error is not None
        assert "object" in error

    def test_validate_success(self):
        """Test validation passes for valid data."""
        model = ClaudeCodeModel("sonnet")
        data = {
            "name": "test",
            "count": 42,
            "value": 3.14,
            "flag": True,
            "items": [1, 2, 3],
            "config": {"key": "value"},
        }
        schema = {
            "required": ["name", "count"],
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "value": {"type": "number"},
                "flag": {"type": "boolean"},
                "items": {"type": "array"},
                "config": {"type": "object"},
            },
        }

        error = model._validate_json_schema(data, schema)
        assert error is None


class TestClaudeCodeModelUsage:
    """Tests for usage creation methods."""

    def test_create_usage_basic(self):
        """Test creating usage from response."""
        response: ClaudeJSONResponse = {
            "type": "result",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 5,
            },
        }

        usage = ClaudeCodeModel._create_usage(response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_write_tokens == 10
        assert usage.cache_read_tokens == 5

    def test_create_usage_with_details(self):
        """Test creating usage with additional details."""
        response: ClaudeJSONResponse = {
            "type": "result",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "server_tool_use": {"web_search_requests": 3},
            },
            "total_cost_usd": 0.05,
            "duration_ms": 1000,
            "duration_api_ms": 800,
            "num_turns": 2,
        }

        usage = ClaudeCodeModel._create_usage(response)
        assert usage.details["web_search_requests"] == 3
        assert usage.details["total_cost_usd_cents"] == 5
        assert usage.details["duration_ms"] == 1000
        assert usage.details["num_turns"] == 2


class TestClaudeCodeModelGetModelName:
    """Tests for _get_model_name method."""

    def test_get_model_name_from_response(self):
        """Test getting model name from response."""
        model = ClaudeCodeModel("sonnet")
        response: ClaudeJSONResponse = {
            "type": "result",
            "modelUsage": {"claude-3-5-sonnet-20241022": {"input_tokens": 100}},
        }

        name = model._get_model_name(response)
        assert name == "claude-3-5-sonnet-20241022"

    def test_get_model_name_fallback(self):
        """Test fallback to model alias when not in response."""
        model = ClaudeCodeModel("opus")
        response: ClaudeJSONResponse = {"type": "result"}

        name = model._get_model_name(response)
        assert name == "opus"


class TestClaudeCodeModelFunctionTools:
    """Tests for function tool handling."""

    def test_build_function_option_descriptions(self):
        """Test building function option descriptions."""
        model = ClaudeCodeModel("sonnet")

        # Create mock tool
        tool = mock.MagicMock()
        tool.name = "get_weather"
        tool.description = "Get weather for a city"
        tool.parameters_json_schema = {
            "properties": {"city": {"type": "string"}},
        }

        descriptions = model._build_function_option_descriptions([tool])

        assert len(descriptions) == 2  # Tool + none option
        assert "get_weather" in descriptions[0]
        assert "city: string" in descriptions[0]
        assert "none" in descriptions[1]


class TestClaudeCodeModelPrepareWorkingDirectory:
    """Tests for _prepare_working_directory method."""

    def test_prepare_creates_directory(self):
        """Test preparing working directory."""
        model = ClaudeCodeModel("sonnet")
        settings: ClaudeCodeSettings = {}

        model._prepare_working_directory(settings)

        assert "__working_directory" in settings
        assert Path(settings["__working_directory"]).exists()


class TestClaudeCodeModelReadStructuredOutput:
    """Tests for _read_structured_output_file method."""

    def test_read_from_file(self):
        """Test reading structured output from file."""
        model = ClaudeCodeModel("sonnet")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = f"{tmpdir}/output.json"
            Path(file_path).write_text('{"name": "test"}')

            schema = {"required": ["name"], "properties": {"name": {"type": "string"}}}

            data, error = model._read_structured_output_file(file_path, schema)

            assert error is None
            assert data["name"] == "test"

    def test_read_invalid_json(self):
        """Test error on invalid JSON."""
        model = ClaudeCodeModel("sonnet")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = f"{tmpdir}/output.json"
            Path(file_path).write_text("not json")

            schema = {"properties": {"name": {"type": "string"}}}

            data, error = model._read_structured_output_file(file_path, schema)

            assert data is None
            assert error is not None
            assert "formatted correctly" in error

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file returns None."""
        model = ClaudeCodeModel("sonnet")

        data, error = model._read_structured_output_file(
            "/nonexistent/file.json",
            {"properties": {}},
        )

        assert data is None
        assert error is None


# ===== Integration-style tests that still don't make CLI calls =====


class TestHandleFunctionSelectionResponse:
    """Tests for function selection response handling."""

    def test_parse_choice_single(self):
        """Test parsing single CHOICE response."""
        model = ClaudeCodeModel("sonnet")
        response: ClaudeJSONResponse = {"type": "result"}
        settings: ClaudeCodeSettings = {
            "__available_functions__": {"get_weather": mock.MagicMock()},
        }

        result_text = "CHOICE: get_weather"
        result = model._handle_function_selection_response(result_text, response, settings)

        assert settings["__selected_function__"] == "get_weather"

    def test_parse_choice_none(self):
        """Test parsing CHOICE: none response."""
        model = ClaudeCodeModel("sonnet")
        response: ClaudeJSONResponse = {"type": "result"}
        settings: ClaudeCodeSettings = {
            "__available_functions__": {"get_weather": mock.MagicMock()},
        }

        result_text = "CHOICE: none"
        result = model._handle_function_selection_response(result_text, response, settings)

        assert settings["__function_selection_result__"] == "none"

    def test_parse_choice_with_formatting(self):
        """Test parsing CHOICE with markdown formatting."""
        model = ClaudeCodeModel("sonnet")
        response: ClaudeJSONResponse = {"type": "result"}
        settings: ClaudeCodeSettings = {
            "__available_functions__": {"get_weather": mock.MagicMock()},
        }

        result_text = "CHOICE: **get_weather**"
        result = model._handle_function_selection_response(result_text, response, settings)

        assert settings["__selected_function__"] == "get_weather"


class TestHandleUnstructuredOutputResponse:
    """Tests for unstructured output response handling."""

    def test_read_from_file(self):
        """Test reading unstructured output from file."""
        model = ClaudeCodeModel("sonnet")
        response: ClaudeJSONResponse = {"type": "result", "result": "fallback"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = f"{tmpdir}/output.txt"
            Path(output_file).write_text("File content")

            settings: ClaudeCodeSettings = {
                "__unstructured_output_file": output_file,
            }

            result = model._handle_unstructured_output_response(
                "result text", response, settings
            )

            assert len(result.parts) == 1
            assert isinstance(result.parts[0], TextPart)
            assert result.parts[0].content == "File content"

    def test_fallback_to_result(self):
        """Test fallback to result text when file doesn't exist."""
        model = ClaudeCodeModel("sonnet")
        response: ClaudeJSONResponse = {"type": "result", "result": "result text"}

        settings: ClaudeCodeSettings = {
            "__unstructured_output_file": "/nonexistent/file.txt",
        }

        result = model._handle_unstructured_output_response(
            "result text", response, settings
        )

        assert result.parts[0].content == "result text"


# ===== Tests for sync/async execution with mocked subprocess =====


class TestExecuteSyncCommand:
    """Tests for _execute_sync_command function."""

    def test_sync_command_success(self):
        """Test successful sync command execution."""
        from pydantic_ai_claude_code.utils import _execute_sync_command

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout='{"type":"result","result":"test"}',
                stderr="",
                returncode=0,
            )

            result = _execute_sync_command(
                cmd=["claude", "--print"],
                cwd="/tmp/test",
                timeout_seconds=60,
                settings=None,
            )

            assert result.returncode == 0
            mock_run.assert_called_once()

    def test_sync_command_with_provider_env(self):
        """Test sync command with provider environment variables."""
        from pydantic_ai_claude_code.utils import _execute_sync_command

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout='{"type":"result"}',
                stderr="",
                returncode=0,
            )

            settings: ClaudeCodeSettings = {
                "__provider_env": {"OPENAI_API_KEY": "test-key"},
            }

            result = _execute_sync_command(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                settings=settings,
            )

            # Verify env was passed to subprocess
            call_args = mock_run.call_args
            assert call_args.kwargs.get("env") is not None
            assert "OPENAI_API_KEY" in call_args.kwargs["env"]

    def test_sync_command_with_sandbox_env(self):
        """Test sync command with sandbox environment variables."""
        from pydantic_ai_claude_code.utils import _execute_sync_command

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout='{"type":"result"}',
                stderr="",
                returncode=0,
            )

            settings: ClaudeCodeSettings = {
                "__sandbox_env": {"IS_SANDBOX": "1"},
            }

            result = _execute_sync_command(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                settings=settings,
            )

            call_args = mock_run.call_args
            assert call_args.kwargs.get("env") is not None
            assert call_args.kwargs["env"].get("IS_SANDBOX") == "1"

    def test_sync_command_with_prompt_input(self):
        """Test sync command passes prompt via stdin."""
        from pydantic_ai_claude_code.utils import _execute_sync_command

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout='{"type":"result"}',
                stderr="",
                returncode=0,
            )

            settings: ClaudeCodeSettings = {
                "__prompt_text": "Test prompt content",
            }

            result = _execute_sync_command(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                settings=settings,
            )

            call_args = mock_run.call_args
            assert call_args.kwargs.get("input") == "Test prompt content"

    def test_sync_command_timeout(self):
        """Test sync command timeout raises RuntimeError."""
        import subprocess as sp

        from pydantic_ai_claude_code.utils import _execute_sync_command

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = sp.TimeoutExpired(cmd=["claude"], timeout=60)

            with pytest.raises(RuntimeError, match="timeout"):
                _execute_sync_command(
                    cmd=["claude"],
                    cwd="/tmp",
                    timeout_seconds=60,
                    settings=None,
                )


class TestTrySyncExecutionWithRetry:
    """Tests for _try_sync_execution_with_rate_limit_retry function."""

    def test_success_on_first_try(self):
        """Test successful execution on first try."""
        from pydantic_ai_claude_code.utils import _try_sync_execution_with_rate_limit_retry

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout='{"type":"result","result":"success"}',
                stderr="",
                returncode=0,
            )

            response, should_retry = _try_sync_execution_with_rate_limit_retry(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                retry_enabled=True,
                settings=None,
            )

            assert response is not None
            assert should_retry is False
            assert response["result"] == "success"

    def test_infrastructure_failure_returns_retry(self):
        """Test infrastructure failure returns retry flag."""
        from pydantic_ai_claude_code.utils import _try_sync_execution_with_rate_limit_retry

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout="",
                stderr="Cannot find module 'yoga.wasm'",
                returncode=1,
            )

            response, should_retry = _try_sync_execution_with_rate_limit_retry(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                retry_enabled=True,
                settings=None,
            )

            assert response is None
            assert should_retry is True

    def test_rate_limit_retry_with_mocked_sleep(self):
        """Test rate limit triggers retry after sleep."""
        from pydantic_ai_claude_code.utils import _try_sync_execution_with_rate_limit_retry

        call_count = 0

        def mock_run_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock.MagicMock(
                    stdout="5-hour limit reached ∙ resets 3PM",
                    stderr="",
                    returncode=1,
                )
            return mock.MagicMock(
                stdout='{"type":"result","result":"success"}',
                stderr="",
                returncode=0,
            )

        with (
            mock.patch("subprocess.run", side_effect=mock_run_side_effect),
            mock.patch("time.sleep") as mock_sleep,
            mock.patch("pydantic_ai_claude_code.utils.calculate_wait_time", return_value=1),
        ):
            response, should_retry = _try_sync_execution_with_rate_limit_retry(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                retry_enabled=True,
                settings=None,
            )

            assert response is not None
            assert should_retry is False
            assert call_count == 2
            mock_sleep.assert_called_once()


class TestRunClaudeSync:
    """Tests for run_claude_sync function."""

    def test_run_sync_success(self):
        """Test successful sync run."""
        from pydantic_ai_claude_code.utils import run_claude_sync

        with (
            mock.patch("subprocess.run") as mock_run,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            mock_run.return_value = mock.MagicMock(
                stdout='{"type":"result","result":"test"}',
                stderr="",
                returncode=0,
            )

            settings: ClaudeCodeSettings = {
                "working_directory": tmpdir,
            }

            response = run_claude_sync("Test prompt", settings=settings)

            assert response["result"] == "test"

    def test_run_sync_with_infrastructure_retry(self):
        """Test sync run retries on infrastructure failure."""
        from pydantic_ai_claude_code.utils import run_claude_sync

        call_count = 0

        def mock_run_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return mock.MagicMock(
                    stdout="",
                    stderr="MODULE_NOT_FOUND: @anthropic-ai/claude-code",
                    returncode=1,
                )
            return mock.MagicMock(
                stdout='{"type":"result","result":"success"}',
                stderr="",
                returncode=0,
            )

        with (
            mock.patch("subprocess.run", side_effect=mock_run_side_effect),
            mock.patch("time.sleep") as mock_sleep,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            settings: ClaudeCodeSettings = {
                "working_directory": tmpdir,
            }

            response = run_claude_sync("Test", settings=settings)

            assert response["result"] == "success"
            assert call_count == 3
            # Should have slept between retries
            assert mock_sleep.call_count >= 2

    def test_run_sync_max_retries_exceeded(self):
        """Test sync run fails after max retries."""
        from pydantic_ai_claude_code.utils import run_claude_sync

        with (
            mock.patch("subprocess.run") as mock_run,
            mock.patch("time.sleep"),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            mock_run.return_value = mock.MagicMock(
                stdout="",
                stderr="MODULE_NOT_FOUND: persistent failure",
                returncode=1,
            )

            settings: ClaudeCodeSettings = {
                "working_directory": tmpdir,
            }

            with pytest.raises(RuntimeError, match="infrastructure failure persisted"):
                run_claude_sync("Test", settings=settings)


class TestExecuteAsyncCommand:
    """Tests for _execute_async_command function."""

    @pytest.mark.asyncio
    async def test_async_command_success(self):
        """Test successful async command execution."""
        from pydantic_ai_claude_code.utils import _execute_async_command

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b'{"type":"result"}', b"")
        )
        mock_process.returncode = 0

        with mock.patch(
            "pydantic_ai_claude_code.utils.create_subprocess_async",
            return_value=mock_process,
        ):
            stdout, stderr, returncode = await _execute_async_command(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                settings=None,
            )

            assert returncode == 0
            assert b"result" in stdout

    @pytest.mark.asyncio
    async def test_async_command_with_provider_env(self):
        """Test async command with provider environment variables."""
        from pydantic_ai_claude_code.utils import _execute_async_command

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b'{"type":"result"}', b"")
        )
        mock_process.returncode = 0

        with mock.patch(
            "pydantic_ai_claude_code.utils.create_subprocess_async",
            return_value=mock_process,
        ) as mock_create:
            settings: ClaudeCodeSettings = {
                "__provider_env": {"DEEPSEEK_API_KEY": "test-key"},
            }

            await _execute_async_command(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                settings=settings,
            )

            # Verify env was passed
            call_args = mock_create.call_args
            assert call_args.args[2] is not None  # env argument
            assert "DEEPSEEK_API_KEY" in call_args.args[2]

    @pytest.mark.asyncio
    async def test_async_command_with_prompt_input(self):
        """Test async command passes prompt via stdin."""
        from pydantic_ai_claude_code.utils import _execute_async_command

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b'{"type":"result"}', b"")
        )
        mock_process.returncode = 0

        with mock.patch(
            "pydantic_ai_claude_code.utils.create_subprocess_async",
            return_value=mock_process,
        ):
            settings: ClaudeCodeSettings = {
                "__prompt_text": "Test async prompt",
            }

            await _execute_async_command(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                settings=settings,
            )

            # Verify prompt was passed as input
            call_args = mock_process.communicate.call_args
            assert call_args.kwargs.get("input") == b"Test async prompt"

    @pytest.mark.asyncio
    async def test_async_command_timeout(self):
        """Test async command timeout raises RuntimeError."""
        from pydantic_ai_claude_code.utils import _execute_async_command

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = mock.AsyncMock()
        mock_process.wait = mock.AsyncMock()

        with mock.patch(
            "pydantic_ai_claude_code.utils.create_subprocess_async",
            return_value=mock_process,
        ):
            with pytest.raises(RuntimeError, match="timeout"):
                await _execute_async_command(
                    cmd=["claude"],
                    cwd="/tmp",
                    timeout_seconds=1,
                    settings=None,
                )


class TestTryAsyncExecutionWithRetry:
    """Tests for _try_async_execution_with_rate_limit_retry function."""

    @pytest.mark.asyncio
    async def test_async_success_on_first_try(self):
        """Test successful async execution on first try."""
        from pydantic_ai_claude_code.utils import _try_async_execution_with_rate_limit_retry

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b'{"type":"result","result":"success"}', b"")
        )
        mock_process.returncode = 0

        with mock.patch(
            "pydantic_ai_claude_code.utils.create_subprocess_async",
            return_value=mock_process,
        ):
            response, should_retry = await _try_async_execution_with_rate_limit_retry(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                retry_enabled=True,
                settings=None,
            )

            assert response is not None
            assert should_retry is False
            assert response["result"] == "success"

    @pytest.mark.asyncio
    async def test_async_infrastructure_failure_returns_retry(self):
        """Test async infrastructure failure returns retry flag."""
        from pydantic_ai_claude_code.utils import _try_async_execution_with_rate_limit_retry

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b"", b"Cannot find module 'yoga.wasm'")
        )
        mock_process.returncode = 1

        with mock.patch(
            "pydantic_ai_claude_code.utils.create_subprocess_async",
            return_value=mock_process,
        ):
            response, should_retry = await _try_async_execution_with_rate_limit_retry(
                cmd=["claude"],
                cwd="/tmp",
                timeout_seconds=60,
                retry_enabled=True,
                settings=None,
            )

            assert response is None
            assert should_retry is True


class TestRunClaudeAsync:
    """Tests for run_claude_async function."""

    @pytest.mark.asyncio
    async def test_run_async_success(self):
        """Test successful async run."""
        from pydantic_ai_claude_code.utils import run_claude_async

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b'{"type":"result","result":"test"}', b"")
        )
        mock_process.returncode = 0

        with (
            mock.patch(
                "pydantic_ai_claude_code.utils.create_subprocess_async",
                return_value=mock_process,
            ),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            settings: ClaudeCodeSettings = {
                "working_directory": tmpdir,
            }

            response = await run_claude_async("Test prompt", settings=settings)

            assert response["result"] == "test"

    @pytest.mark.asyncio
    async def test_run_async_with_infrastructure_retry(self):
        """Test async run retries on infrastructure failure."""
        from pydantic_ai_claude_code.utils import run_claude_async

        call_count = 0

        async def mock_communicate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return (b"", b"ENOENT: no such file or directory")
            return (b'{"type":"result","result":"success"}', b"")

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock_communicate
        mock_process.returncode = 1 if call_count <= 2 else 0

        def get_returncode():
            return 1 if call_count <= 2 else 0

        type(mock_process).returncode = property(lambda self: get_returncode())

        with (
            mock.patch(
                "pydantic_ai_claude_code.utils.create_subprocess_async",
                return_value=mock_process,
            ),
            mock.patch("asyncio.sleep") as mock_sleep,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            settings: ClaudeCodeSettings = {
                "working_directory": tmpdir,
            }

            response = await run_claude_async("Test", settings=settings)

            assert response["result"] == "success"

    @pytest.mark.asyncio
    async def test_run_async_max_retries_exceeded(self):
        """Test async run fails after max retries."""
        from pydantic_ai_claude_code.utils import run_claude_async

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b"", b"MODULE_NOT_FOUND: persistent failure")
        )
        mock_process.returncode = 1

        with (
            mock.patch(
                "pydantic_ai_claude_code.utils.create_subprocess_async",
                return_value=mock_process,
            ),
            mock.patch("asyncio.sleep"),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            settings: ClaudeCodeSettings = {
                "working_directory": tmpdir,
            }

            with pytest.raises(RuntimeError, match="infrastructure failure persisted"):
                await run_claude_async("Test", settings=settings)


# ===== Tests for streaming.py run_claude_streaming =====


class TestRunClaudeStreaming:
    """Tests for run_claude_streaming function."""

    @pytest.mark.asyncio
    async def test_streaming_basic(self):
        """Test basic streaming with mocked process."""
        from pydantic_ai_claude_code.streaming import run_claude_streaming

        mock_stdout = mock.AsyncMock()
        lines = [
            b'{"type":"message_start"}\n',
            b'{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n',
            b'{"type":"result","result":"Done"}\n',
            b"",  # EOF
        ]
        line_iter = iter(lines)
        mock_stdout.readline = mock.AsyncMock(side_effect=lambda: next(line_iter))

        mock_process = mock.AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock.AsyncMock()
        mock_process.wait = mock.AsyncMock()
        mock_process.returncode = 0

        with mock.patch(
            "pydantic_ai_claude_code.streaming.create_subprocess_async",
            return_value=mock_process,
        ):
            events = []
            async for event in run_claude_streaming(["claude", "--streaming"], cwd="/tmp"):
                events.append(event)

            assert len(events) == 3
            assert events[0]["type"] == "message_start"
            assert events[2]["type"] == "result"

    @pytest.mark.asyncio
    async def test_streaming_unwraps_verbose(self):
        """Test streaming unwraps verbose stream_event wrapper."""
        from pydantic_ai_claude_code.streaming import run_claude_streaming

        mock_stdout = mock.AsyncMock()
        lines = [
            b'{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}}\n',
            b"",  # EOF
        ]
        line_iter = iter(lines)
        mock_stdout.readline = mock.AsyncMock(side_effect=lambda: next(line_iter))

        mock_process = mock.AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock.AsyncMock()
        mock_process.wait = mock.AsyncMock()
        mock_process.returncode = 0

        with mock.patch(
            "pydantic_ai_claude_code.streaming.create_subprocess_async",
            return_value=mock_process,
        ):
            events = []
            async for event in run_claude_streaming(["claude"], cwd="/tmp"):
                events.append(event)

            assert len(events) == 1
            # Should have unwrapped the nested event
            assert events[0]["type"] == "content_block_delta"

    @pytest.mark.asyncio
    async def test_streaming_skips_invalid_json(self):
        """Test streaming skips invalid JSON lines."""
        from pydantic_ai_claude_code.streaming import run_claude_streaming

        mock_stdout = mock.AsyncMock()
        lines = [
            b'{"type":"message_start"}\n',
            b"not valid json\n",
            b'{"type":"result"}\n',
            b"",  # EOF
        ]
        line_iter = iter(lines)
        mock_stdout.readline = mock.AsyncMock(side_effect=lambda: next(line_iter))

        mock_process = mock.AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock.AsyncMock()
        mock_process.wait = mock.AsyncMock()
        mock_process.returncode = 0

        with mock.patch(
            "pydantic_ai_claude_code.streaming.create_subprocess_async",
            return_value=mock_process,
        ):
            events = []
            async for event in run_claude_streaming(["claude"], cwd="/tmp"):
                events.append(event)

            # Should have skipped invalid JSON
            assert len(events) == 2

    @pytest.mark.asyncio
    async def test_streaming_error_on_failure(self):
        """Test streaming raises error on process failure."""
        from pydantic_ai_claude_code.streaming import run_claude_streaming

        mock_stdout = mock.AsyncMock()
        mock_stdout.readline = mock.AsyncMock(return_value=b"")

        mock_stderr = mock.AsyncMock()
        mock_stderr.read = mock.AsyncMock(return_value=b"Process failed")

        mock_process = mock.AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait = mock.AsyncMock()
        mock_process.returncode = 1

        with mock.patch(
            "pydantic_ai_claude_code.streaming.create_subprocess_async",
            return_value=mock_process,
        ):
            with pytest.raises(RuntimeError, match="Process failed"):
                async for event in run_claude_streaming(["claude"], cwd="/tmp"):
                    pass


# ===== Tests for streamed_response.py background consumption =====


class TestStreamedResponseBackgroundConsumption:
    """Tests for ClaudeCodeStreamedResponse background stream consumption."""

    @pytest.mark.asyncio
    async def test_consume_stream_with_content_block_delta(self):
        """Test consuming stream with content_block_delta events."""
        async def mock_stream():
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " World"}}
            yield {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 5}}

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=mock_stream(),
        )

        # Wait for stream to complete
        await response._stream_complete.wait()

        # Should have buffered events
        assert len(response._buffered_events) > 0
        # Usage should be set
        assert response._usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_consume_stream_with_marker(self):
        """Test consuming stream with streaming marker."""
        async def mock_stream():
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "prefix<<<MARKER>>>actual content"}}
            yield {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 5}}

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=mock_stream(),
            streaming_marker="<<<MARKER>>>",
        )

        # Wait for stream to complete
        await response._stream_complete.wait()

        # Should have buffered events after marker
        assert len(response._buffered_events) > 0

    @pytest.mark.asyncio
    async def test_consume_stream_skips_message_start(self):
        """Test consuming stream skips message_start events."""
        async def mock_stream():
            yield {"type": "message_start"}
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Text"}}
            yield {"type": "result", "usage": {"input_tokens": 1, "output_tokens": 1}}

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=mock_stream(),
        )

        # Wait for stream to complete
        await response._stream_complete.wait()

        # message_start should be skipped
        assert response._usage.input_tokens == 1

    @pytest.mark.asyncio
    async def test_consume_stream_skips_other_indices(self):
        """Test consuming stream skips content blocks with index != 0."""
        async def mock_stream():
            yield {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Ignored"}}
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Used"}}
            yield {"type": "result", "usage": {"input_tokens": 1, "output_tokens": 1}}

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=mock_stream(),
        )

        # Wait for stream to complete
        await response._stream_complete.wait()

        # Should process only index 0
        assert len(response._buffered_events) > 0

    @pytest.mark.asyncio
    async def test_get_event_iterator(self):
        """Test _get_event_iterator yields events from buffer."""
        async def mock_stream():
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Test"}}
            yield {"type": "result", "usage": {"input_tokens": 1, "output_tokens": 1}}

        response = ClaudeCodeStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            model_name="test",
            event_stream=mock_stream(),
        )

        # Wait for stream to complete
        await response._stream_complete.wait()

        # Get events from iterator
        events = []
        async for event in response._get_event_iterator():
            events.append(event)

        assert len(events) > 0


# ===== Tests for working directory setup =====


class TestSetupWorkingDirectory:
    """Tests for _setup_working_directory_and_prompt function."""

    def test_setup_with_no_settings(self):
        """Test setup with no settings creates temp directory."""
        from pydantic_ai_claude_code.utils import _setup_working_directory_and_prompt

        cwd = _setup_working_directory_and_prompt("Test prompt", None)

        assert Path(cwd).exists()
        assert (Path(cwd) / "prompt.md").exists()

    def test_setup_with_existing_working_dir(self):
        """Test setup uses pre-determined working directory."""
        from pydantic_ai_claude_code.utils import _setup_working_directory_and_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {
                "__working_directory": tmpdir,
            }

            cwd = _setup_working_directory_and_prompt("Test", settings)

            assert cwd == tmpdir
            assert (Path(cwd) / "prompt.md").exists()

    def test_setup_creates_temp_base_for_empty_settings(self):
        """Test setup creates temp base directory for empty settings."""
        from pydantic_ai_claude_code.utils import _setup_working_directory_and_prompt

        settings: ClaudeCodeSettings = {}

        cwd = _setup_working_directory_and_prompt("Test", settings)

        assert Path(cwd).exists()
        assert "__temp_base_directory" in settings
        assert "__working_directory" in settings

    def test_setup_reuses_temp_base(self):
        """Test setup reuses existing temp base directory."""
        from pydantic_ai_claude_code.utils import _setup_working_directory_and_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {
                "__temp_base_directory": tmpdir,
            }

            cwd1 = _setup_working_directory_and_prompt("Test 1", settings)

            # Reset working directory to allow next call
            del settings["__working_directory"]

            cwd2 = _setup_working_directory_and_prompt("Test 2", settings)

            # Both should be under the same temp base
            assert tmpdir in cwd1
            assert tmpdir in cwd2
            # But different subdirectories
            assert cwd1 != cwd2


# ===== Tests for save raw response with error handling =====


class TestSaveRawResponseErrorHandling:
    """Tests for _save_raw_response_to_working_dir error handling."""

    def test_save_raw_response_handles_write_error(self):
        """Test save raw response handles write errors gracefully."""
        from pydantic_ai_claude_code.utils import _save_raw_response_to_working_dir

        settings: ClaudeCodeSettings = {
            "__response_file_path": "/nonexistent/directory/response.json",
        }
        response: ClaudeJSONResponse = {"type": "result", "result": "test"}

        # Should not raise, just log warning
        _save_raw_response_to_working_dir(response, settings)


# ===== Tests for OAuth error detection =====


class TestDetectOAuthError:
    """Tests for detect_oauth_error function."""

    def test_detect_oauth_token_revoked(self):
        """Test detection of OAuth token revoked error."""
        stdout = '{"type":"result","is_error":true,"result":"OAuth token revoked · Please run /login"}'
        is_oauth, message = detect_oauth_error(stdout, "")

        assert is_oauth is True
        assert "OAuth token revoked" in message

    def test_detect_auth_expired(self):
        """Test detection of auth expired error."""
        stdout = '{"type":"result","is_error":true,"result":"Auth expired, please login again"}'
        is_oauth, message = detect_oauth_error(stdout, "")

        assert is_oauth is True
        assert "Auth expired" in message

    def test_no_oauth_error_on_normal_response(self):
        """Test no OAuth error on normal response."""
        stdout = '{"type":"result","result":"Success"}'
        is_oauth, message = detect_oauth_error(stdout, "")

        assert is_oauth is False
        assert message is None

    def test_no_oauth_error_on_invalid_json(self):
        """Test no OAuth error on invalid JSON."""
        stdout = "not json"
        is_oauth, message = detect_oauth_error(stdout, "")

        assert is_oauth is False
        assert message is None


# ===== Additional edge case tests =====


class TestAddSettingsFlags:
    """Tests for _add_settings_flags helper function."""

    def test_add_allowed_tools(self):
        """Test adding allowed tools to command."""
        from pydantic_ai_claude_code.utils import _add_tool_permission_flags

        cmd: list[str] = []
        settings: ClaudeCodeSettings = {
            "allowed_tools": ["Bash", "Read", "Write"],
        }

        _add_tool_permission_flags(cmd, settings)

        assert "--allowed-tools" in cmd
        assert "Bash" in cmd
        assert "Read" in cmd

    def test_add_disallowed_tools(self):
        """Test adding disallowed tools to command."""
        from pydantic_ai_claude_code.utils import _add_tool_permission_flags

        cmd: list[str] = []
        settings: ClaudeCodeSettings = {
            "disallowed_tools": ["WebFetch"],
        }

        _add_tool_permission_flags(cmd, settings)

        assert "--disallowed-tools" in cmd
        assert "WebFetch" in cmd

    def test_add_model_flags(self):
        """Test adding model flags to command."""
        from pydantic_ai_claude_code.utils import _add_model_flags

        cmd: list[str] = []
        settings: ClaudeCodeSettings = {
            "model": "claude-3-5-sonnet",
            "fallback_model": "claude-3-haiku",
            "session_id": "test-session",
        }

        _add_model_flags(cmd, settings)

        assert "--model" in cmd
        assert "claude-3-5-sonnet" in cmd
        assert "--fallback-model" in cmd
        assert "--session-id" in cmd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
