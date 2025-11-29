"""Tests for SDK types and errors."""

import pytest

from pydantic_ai_claude_code._sdk import (
    # Errors
    AuthenticationError,
    CLIConnectionError,
    CLINotFoundError,
    ClaudeSDKError,
    ProcessError,
    RateLimitError,
    TimeoutError,
    # Types
    ClaudeAgentOptions,
    HookConfig,
    HookEvent,
    HookMatcher,
    PermissionMode,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    SDKResponse,
    SDKUsage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
)


class TestSDKErrors:
    """Test SDK error classes."""

    def test_claude_sdk_error_base(self):
        """Test base ClaudeSDKError."""
        error = ClaudeSDKError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_cli_connection_error(self):
        """Test CLIConnectionError."""
        error = CLIConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, ClaudeSDKError)

    def test_cli_not_found_error_default_message(self):
        """Test CLINotFoundError with default message."""
        error = CLINotFoundError()
        assert "Claude CLI not found in PATH" in str(error)
        assert isinstance(error, CLIConnectionError)

    def test_cli_not_found_error_custom_message(self):
        """Test CLINotFoundError with custom message."""
        error = CLINotFoundError("Custom message")
        assert str(error) == "Custom message"

    def test_process_error(self):
        """Test ProcessError with attributes."""
        error = ProcessError("Process failed", return_code=1, stderr="Error output")
        assert str(error) == "Process failed"
        assert error.return_code == 1
        assert error.stderr == "Error output"
        assert isinstance(error, ClaudeSDKError)

    def test_process_error_without_attributes(self):
        """Test ProcessError without optional attributes."""
        error = ProcessError("Process failed")
        assert error.return_code is None
        assert error.stderr is None

    def test_timeout_error_default(self):
        """Test TimeoutError with default message."""
        error = TimeoutError()
        assert "timed out" in str(error)
        assert error.timeout_seconds is None

    def test_timeout_error_with_timeout(self):
        """Test TimeoutError with timeout seconds."""
        error = TimeoutError("Custom timeout", timeout_seconds=30)
        assert str(error) == "Custom timeout"
        assert error.timeout_seconds == 30

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, ClaudeSDKError)

    def test_rate_limit_error_default(self):
        """Test RateLimitError with default message."""
        error = RateLimitError()
        assert "Rate limit exceeded" in str(error)
        assert error.retry_after_seconds is None

    def test_rate_limit_error_with_retry(self):
        """Test RateLimitError with retry_after_seconds."""
        error = RateLimitError("Rate limited", retry_after_seconds=60)
        assert str(error) == "Rate limited"
        assert error.retry_after_seconds == 60


class TestSDKTypes:
    """Test SDK type definitions."""

    def test_text_block(self):
        """Test TextBlock TypedDict."""
        block: TextBlock = {"type": "text", "text": "Hello"}
        assert block["type"] == "text"
        assert block["text"] == "Hello"

    def test_tool_use_block(self):
        """Test ToolUseBlock TypedDict."""
        block: ToolUseBlock = {
            "type": "tool_use",
            "id": "tool-123",
            "name": "read_file",
            "input": {"path": "/test"},
        }
        assert block["type"] == "tool_use"
        assert block["id"] == "tool-123"
        assert block["name"] == "read_file"
        assert block["input"]["path"] == "/test"

    def test_tool_result_block(self):
        """Test ToolResultBlock TypedDict."""
        block: ToolResultBlock = {
            "type": "tool_result",
            "tool_use_id": "tool-123",
            "content": "File contents",
            "is_error": False,
        }
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tool-123"
        assert block["is_error"] is False

    def test_claude_agent_options(self):
        """Test ClaudeAgentOptions TypedDict."""
        options: ClaudeAgentOptions = {
            "model": "sonnet",
            "cwd": "/path/to/project",
            "allowed_tools": ["Read", "Edit"],
            "permission_mode": "bypassPermissions",
            "max_turns": 10,
        }
        assert options["model"] == "sonnet"
        assert options["cwd"] == "/path/to/project"
        assert "Read" in options["allowed_tools"]
        assert options["permission_mode"] == "bypassPermissions"

    def test_permission_mode_literal(self):
        """Test PermissionMode literal type."""
        # All valid values
        modes: list[PermissionMode] = [
            "bypassPermissions",
            "acceptEdits",
            "default",
            "plan",
        ]
        assert len(modes) == 4

    def test_hook_config(self):
        """Test HookConfig TypedDict."""
        hook: HookConfig = {
            "matcher": {"event": "tool_use", "tool_name": "Bash"},
            "commands": ["echo $TOOL_NAME"],
            "timeout_ms": 5000,
        }
        assert hook["matcher"]["event"] == "tool_use"
        assert "echo" in hook["commands"][0]

    def test_hook_matcher(self):
        """Test HookMatcher TypedDict."""
        matcher: HookMatcher = {
            "event": "tool_use",
            "tool_name": "Edit",
            "pattern": "*.py",
        }
        assert matcher["event"] == "tool_use"
        assert matcher["tool_name"] == "Edit"

    def test_hook_event(self):
        """Test HookEvent TypedDict."""
        event: HookEvent = {
            "type": "tool_use",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "result": "file.txt",
        }
        assert event["type"] == "tool_use"
        assert event["tool_name"] == "Bash"

    def test_permission_result_allow(self):
        """Test PermissionResultAllow TypedDict."""
        result: PermissionResultAllow = {
            "behavior": "allow",
            "updated_input": {"command": "ls -la"},
        }
        assert result["behavior"] == "allow"
        assert result["updated_input"]["command"] == "ls -la"

    def test_permission_result_deny(self):
        """Test PermissionResultDeny TypedDict."""
        result: PermissionResultDeny = {
            "behavior": "deny",
            "message": "Dangerous operation",
        }
        assert result["behavior"] == "deny"
        assert "Dangerous" in result["message"]

    def test_tool_permission_context(self):
        """Test ToolPermissionContext TypedDict."""
        context: ToolPermissionContext = {
            "session_id": "sess-123",
            "turn_number": 5,
            "tool_history": [{"name": "Read", "success": True}],
            "working_directory": "/tmp",
        }
        assert context["session_id"] == "sess-123"
        assert context["turn_number"] == 5

    def test_sdk_usage(self):
        """Test SDKUsage TypedDict."""
        usage: SDKUsage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
            "total_cost_usd": 0.001,
        }
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50

    def test_sdk_response(self):
        """Test SDKResponse TypedDict."""
        response: SDKResponse = {
            "messages": [],
            "final_result": "Done",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "session_id": "sess-456",
            "duration_ms": 1500,
        }
        assert response["final_result"] == "Done"
        assert response["duration_ms"] == 1500


class TestImportExports:
    """Test that all exports are accessible."""

    def test_import_all_from_sdk(self):
        """Test that all expected types can be imported."""
        from pydantic_ai_claude_code._sdk import (
            AssistantMessage,
            CanUseTool,
            ContentBlock,
            Message,
            PermissionResult,
            ResultMessage,
            UserMessage,
        )

        # Just verify they can be imported
        assert PermissionResult is not None
        assert Message is not None
        assert ContentBlock is not None
        assert UserMessage is not None
        assert AssistantMessage is not None
        assert ResultMessage is not None
        assert CanUseTool is not None
