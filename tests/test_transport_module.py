"""Tests for transport module."""

import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from pydantic_ai_claude_code.transport import (
    EnhancedCLITransport,
    convert_settings_to_sdk_options,
)
from pydantic_ai_claude_code.exceptions import ClaudeOAuthError


class TestConvertSettings:
    """Tests for convert_settings_to_sdk_options function."""

    def test_convert_settings_basic(self):
        """Test basic settings conversion."""
        settings = {
            "working_directory": "/path/to/dir",
            "append_system_prompt": "Custom prompt",
            "allowed_tools": ["Read", "Edit"],
        }
        
        result = convert_settings_to_sdk_options(settings)
        
        assert result["cwd"] == "/path/to/dir"
        assert result["system_prompt"] == "Custom prompt"
        assert result["allowed_tools"] == ["Read", "Edit"]

    def test_convert_settings_dangerously_skip_permissions(self):
        """Test conversion of skip permissions flag."""
        settings = {"dangerously_skip_permissions": True}
        
        result = convert_settings_to_sdk_options(settings)
        
        assert result["permission_mode"] == "acceptEdits"

    def test_convert_settings_claude_cli_path(self):
        """Test conversion of CLI path."""
        settings = {"claude_cli_path": "/custom/claude"}
        
        result = convert_settings_to_sdk_options(settings)
        
        assert result["cli_path"] == "/custom/claude"

    def test_convert_settings_empty(self):
        """Test conversion with empty settings."""
        settings = {}
        
        result = convert_settings_to_sdk_options(settings)
        
        assert result == {}


class TestEnhancedCLITransport:
    """Tests for EnhancedCLITransport class."""

    def test_init_basic(self):
        """Test basic initialization."""
        transport = EnhancedCLITransport("test prompt")
        
        assert transport.prompt == "test prompt"
        assert transport.settings == {}

    def test_init_with_settings(self):
        """Test initialization with settings."""
        settings = {"working_directory": "/tmp", "model": "sonnet"}
        transport = EnhancedCLITransport("test prompt", settings)
        
        assert transport.prompt == "test prompt"
        assert transport.settings == settings

    @pytest.mark.asyncio
    async def test_execute_setup_working_directory(self):
        """Test that execute sets up working directory."""
        transport = EnhancedCLITransport("test prompt", {})
        
        with patch.object(transport, "_setup_working_directory", return_value="/tmp/test"):
            with patch.object(transport, "_build_command", return_value=["claude"]):
                with patch.object(transport, "_try_execution_with_retry", return_value=({"result": "ok"}, False)):
                    result = await transport.execute()
                    
                    assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_execute_handles_oauth_error(self):
        """Test that OAuth errors are properly raised."""
        transport = EnhancedCLITransport("test prompt", {})
        
        with patch.object(transport, "_setup_working_directory", return_value="/tmp/test"):
            with patch.object(transport, "_build_command", return_value=["claude"]):
                with patch.object(transport, "_try_execution_with_retry", side_effect=ClaudeOAuthError("Token expired")):
                    with pytest.raises(ClaudeOAuthError):
                        await transport.execute()

    @pytest.mark.asyncio
    async def test_execute_retries_infrastructure_failures(self):
        """Test retry logic for infrastructure failures."""
        transport = EnhancedCLITransport("test prompt", {})
        
        # Mock to fail twice then succeed
        call_count = 0
        async def mock_try_execution(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return (None, True)  # Infrastructure failure
            return ({"result": "success"}, False)
        
        with patch.object(transport, "_setup_working_directory", return_value="/tmp/test"):
            with patch.object(transport, "_build_command", return_value=["claude"]):
                with patch.object(transport, "_try_execution_with_retry", side_effect=mock_try_execution):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        result = await transport.execute()
                        
                        assert result == {"result": "success"}
                        assert call_count == 3

    def test_setup_working_directory_creates_temp(self):
        """Test working directory creation."""
        transport = EnhancedCLITransport("test prompt", {})
        
        with patch("tempfile.mkdtemp", return_value="/tmp/mock_dir"):
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.write_text"):
                    cwd = transport._setup_working_directory()
                    
                    assert "/tmp/mock_dir" in cwd
                    assert transport.settings.get("__working_directory") is not None

    def test_setup_working_directory_uses_existing(self):
        """Test using existing working directory."""
        transport = EnhancedCLITransport("test prompt", {"working_directory": "/custom/dir"})
        
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.iterdir", return_value=[]):
                with patch("pathlib.Path.write_text"):
                    cwd = transport._setup_working_directory()
                    
                    assert "/custom/dir" in cwd

    def test_build_command_basic(self):
        """Test building basic command."""
        transport = EnhancedCLITransport("test prompt", {})
        transport.settings = {"__working_directory": "/tmp/test"}
        
        with patch("pydantic_ai_claude_code.transport.sdk_transport.resolve_claude_cli_path", return_value="/usr/bin/claude"):
            cmd = transport._build_command()
            
            assert cmd[0] == "/usr/bin/claude"
            assert "--print" in cmd
            assert "--output-format" in cmd
            assert "json" in cmd

    def test_build_command_with_model(self):
        """Test building command with model specification."""
        transport = EnhancedCLITransport("test prompt", {"model": "sonnet"})
        transport.settings["__working_directory"] = "/tmp/test"
        
        with patch("pydantic_ai_claude_code.transport.sdk_transport.resolve_claude_cli_path", return_value="/usr/bin/claude"):
            cmd = transport._build_command()
            
            assert "--model" in cmd
            assert "sonnet" in cmd

    def test_build_command_with_sandbox(self):
        """Test building command with sandbox enabled."""
        transport = EnhancedCLITransport("test prompt", {"use_sandbox_runtime": True})
        transport.settings["__working_directory"] = "/tmp/test"
        
        with patch("pydantic_ai_claude_code.transport.sdk_transport.resolve_claude_cli_path", return_value="/usr/bin/claude"):
            with patch("pydantic_ai_claude_code.transport.sdk_transport.wrap_command_with_sandbox") as mock_wrap:
                mock_wrap.return_value = (["srt", "--", "claude"], {"IS_SANDBOX": "1"})
                
                cmd = transport._build_command()
                
                assert cmd[0] == "srt"
                assert transport._sandbox_env["IS_SANDBOX"] == "1"

    @pytest.mark.asyncio
    async def test_execute_command_basic(self):
        """Test basic command execution."""
        transport = EnhancedCLITransport("test prompt", {})
        
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"stdout", b"stderr"))
        mock_process.returncode = 0
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            stdout, stderr, returncode = await transport._execute_command(
                ["claude"],
                "/tmp/test",
                300
            )
            
            assert stdout == b"stdout"
            assert stderr == b"stderr"
            assert returncode == 0

    @pytest.mark.asyncio
    async def test_execute_command_timeout(self):
        """Test command execution with timeout."""
        transport = EnhancedCLITransport("test prompt", {})
        
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = Mock()
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(asyncio.TimeoutError):
                await transport._execute_command(
                    ["claude"],
                    "/tmp/test",
                    1  # Very short timeout
                )

    def test_process_response_valid_json(self):
        """Test processing valid JSON response."""
        transport = EnhancedCLITransport("test prompt", {})
        
        json_response = json.dumps({"result": "success", "type": "completion"})
        
        result = transport._process_response(json_response)
        
        assert result == {"result": "success", "type": "completion"}

    @pytest.mark.asyncio
    async def test_try_execution_with_retry_rate_limit(self):
        """Test rate limit retry logic."""
        transport = EnhancedCLITransport("test prompt", {"retry_on_rate_limit": True})
        
        # First call returns rate limit, second succeeds
        call_count = 0
        async def mock_execute_command(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (b"", b"Rate limit reached, resets 11AM", 1)
            return (b'{"result": "success"}', b"", 0)
        
        with patch.object(transport, "_execute_command", side_effect=mock_execute_command):
            with patch.object(transport, "_classify_error") as mock_classify:
                mock_classify.side_effect = [
                    ("retry_rate_limit", 0),  # First call triggers retry
                    None  # Second call succeeds
                ]
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    response, should_retry = await transport._try_execution_with_retry(
                        ["claude"],
                        "/tmp/test",
                        300,
                        True
                    )
                    
                    assert response is not None
                    assert call_count == 2


class TestIntegration:
    """Integration tests for transport module."""

    @pytest.mark.asyncio
    async def test_full_execution_flow_success(self):
        """Test full successful execution flow."""
        settings = {
            "model": "sonnet",
            "working_directory": "/tmp/test"
        }
        transport = EnhancedCLITransport("What is 2+2?", settings)
        
        # Mock all external dependencies
        with patch("pydantic_ai_claude_code.transport.sdk_transport.resolve_claude_cli_path", return_value="/usr/bin/claude"):
            with patch("tempfile.mkdtemp", return_value="/tmp/mock"):
                with patch("pathlib.Path.mkdir"):
                    with patch("pathlib.Path.iterdir", return_value=[]):
                        with patch("pathlib.Path.write_text"):
                            mock_process = AsyncMock()
                            mock_process.communicate = AsyncMock(
                                return_value=(b'{"result": "4", "type": "completion"}', b"")
                            )
                            mock_process.returncode = 0
                            
                            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                                result = await transport.execute()
                                
                                assert result["result"] == "4"
                                assert result["type"] == "completion"