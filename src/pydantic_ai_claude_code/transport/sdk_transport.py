"""Enhanced CLI transport bridging Claude Agent SDK with our features.

This module provides a custom transport layer that extends the basic
subprocess CLI handling with our unique features:
- OAuth error detection
- Rate limit retry logic
- Sandbox runtime support
- Debug saving
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, cast

from ..core.oauth_handler import detect_oauth_error
from ..core.retry_logic import (
    detect_rate_limit,
    calculate_wait_time,
    detect_cli_infrastructure_failure,
)
from ..core.sandbox_runtime import wrap_command_with_sandbox
from ..core.debug_saver import (
    save_prompt_debug,
    save_response_debug,
    save_raw_response_to_working_dir,
)
from ..exceptions import ClaudeOAuthError
from ..types import ClaudeCodeSettings, ClaudeJSONResponse
from .._utils.file_utils import get_next_call_subdirectory, copy_additional_files

logger = logging.getLogger(__name__)

# Constants
LONG_RUNTIME_THRESHOLD_SECONDS = 600  # 10 minutes threshold for long runtime warnings
MAX_CLI_RETRIES = 3  # Maximum retries for transient CLI infrastructure failures
RETRY_BACKOFF_BASE = 2  # Exponential backoff base (seconds)


def convert_settings_to_sdk_options(settings: ClaudeCodeSettings) -> dict[str, Any]:
    """Convert our ClaudeCodeSettings to Claude Agent SDK options format.

    Args:
        settings: Our settings dict

    Returns:
        Dict compatible with ClaudeAgentOptions
    """
    sdk_options: dict[str, Any] = {}

    # Map our settings to SDK options
    if settings.get("working_directory"):
        sdk_options["cwd"] = settings["working_directory"]

    if settings.get("append_system_prompt"):
        sdk_options["system_prompt"] = settings["append_system_prompt"]

    if settings.get("allowed_tools"):
        sdk_options["allowed_tools"] = settings["allowed_tools"]

    if settings.get("dangerously_skip_permissions"):
        sdk_options["permission_mode"] = "acceptEdits"

    if settings.get("claude_cli_path"):
        sdk_options["cli_path"] = settings["claude_cli_path"]

    return sdk_options


class EnhancedCLITransport:
    """Enhanced CLI transport with our unique features.

    This transport extends basic subprocess handling with:
    - OAuth error detection and ClaudeOAuthError raising
    - Rate limit detection and automatic retry
    - Infrastructure failure detection and retry with backoff
    - Sandbox runtime wrapping
    - Debug saving functionality
    """

    def __init__(
        self,
        prompt: str,
        settings: ClaudeCodeSettings | None = None,
    ):
        """Initialize the enhanced transport.

        Args:
            prompt: The prompt to send to Claude
            settings: Optional settings for configuration
        """
        self.prompt = prompt
        self.settings = settings or {}
        self._working_directory: str | None = None
        self._sandbox_env: dict[str, str] = {}

    async def execute(self) -> ClaudeJSONResponse:
        """Execute the CLI command with all our enhancements.

        Returns:
            Claude JSON response

        Raises:
            ClaudeOAuthError: If OAuth token is expired
            RuntimeError: For other CLI failures
        """
        retry_enabled = self.settings.get("retry_on_rate_limit", True)
        timeout_seconds = self.settings.get("timeout_seconds", 900)

        # Setup working directory and prompt file
        cwd = self._setup_working_directory()

        # Build command
        cmd = self._build_command()

        # Execute with retries
        for attempt in range(MAX_CLI_RETRIES):
            try:
                response, should_retry_infra = await self._try_execution_with_retry(
                    cmd, cwd, timeout_seconds, retry_enabled
                )

                if response:
                    save_response_debug(response, self.settings)
                    return response

                # Infrastructure failure
                if should_retry_infra and attempt < MAX_CLI_RETRIES - 1:
                    backoff_seconds = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "CLI infrastructure failure (attempt %d/%d). "
                        "Retrying in %d seconds...",
                        attempt + 1,
                        MAX_CLI_RETRIES,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue
                elif should_retry_infra:
                    raise RuntimeError("Claude CLI infrastructure failure persisted")

            except RuntimeError as e:
                if attempt >= MAX_CLI_RETRIES - 1:
                    raise
                if detect_cli_infrastructure_failure(str(e)):
                    backoff_seconds = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "CLI infrastructure failure in execution (attempt %d/%d). "
                        "Retrying in %d seconds...",
                        attempt + 1,
                        MAX_CLI_RETRIES,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue
                raise

        raise RuntimeError("Claude CLI failed after maximum retry attempts")

    def _setup_working_directory(self) -> str:
        """Setup working directory and write prompt file.

        Returns:
            Working directory path
        """
        # Check if already determined
        existing = self.settings.get("__working_directory")
        if existing:
            cwd = existing
            Path(cwd).mkdir(parents=True, exist_ok=True)
        else:
            # Determine base directory
            base_dir = self.settings.get("working_directory")

            if not base_dir:
                # Create temp base directory
                temp_base = self.settings.get("__temp_base_directory")
                if not temp_base:
                    temp_base = tempfile.mkdtemp(prefix="claude_prompt_")
                    self.settings["__temp_base_directory"] = temp_base
                base_dir = temp_base

            # Create numbered subdirectory
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            cwd_path = get_next_call_subdirectory(base_dir)
            cwd = str(cwd_path)

        # Copy additional files
        additional_files = self.settings.get("additional_files")
        if additional_files:
            copy_additional_files(cwd, additional_files)

        # Write prompt file
        prompt_file = Path(cwd) / "prompt.md"
        prompt_file.write_text(self.prompt)
        logger.info("Wrote prompt to: %s", prompt_file)

        # Save debug
        save_prompt_debug(self.prompt, self.settings)

        # Store in settings
        self.settings["__working_directory"] = cwd
        self.settings["__response_file_path"] = str(Path(cwd) / "response.json")
        self.settings["__prompt_text"] = self.prompt

        self._working_directory = cwd
        return cwd

    def _build_command(self) -> list[str]:
        """Build the Claude CLI command.

        Returns:
            Command list
        """
        from ..utils_legacy import resolve_claude_cli_path

        claude_path = resolve_claude_cli_path(self.settings)
        cmd = [claude_path, "--print", "--output-format", "json"]

        # Add permission mode
        permission_mode = self.settings.get("permission_mode") or "bypassPermissions"
        cmd.extend(["--permission-mode", permission_mode])

        # Add skip permissions flag
        if self.settings.get("dangerously_skip_permissions"):
            cmd.append("--dangerously-skip-permissions")

        # Add model flags
        model = self.settings.get("model")
        if model:
            cmd.extend(["--model", model])

        fallback_model = self.settings.get("fallback_model")
        if fallback_model:
            cmd.extend(["--fallback-model", fallback_model])

        # Add tool permissions
        allowed_tools = self.settings.get("allowed_tools")
        if allowed_tools:
            cmd.append("--allowed-tools")
            cmd.extend(allowed_tools)

        disallowed_tools = self.settings.get("disallowed_tools")
        if disallowed_tools:
            cmd.append("--disallowed-tools")
            cmd.extend(disallowed_tools)

        # Add system prompt
        system_prompt = self.settings.get("append_system_prompt")
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Add extra CLI args
        extra_args = self.settings.get("extra_cli_args")
        if extra_args:
            cmd.extend(extra_args)

        # Wrap with sandbox if enabled
        if self.settings.get("use_sandbox_runtime"):
            cmd, self._sandbox_env = wrap_command_with_sandbox(cmd, self.settings)
            self.settings["__sandbox_env"] = self._sandbox_env

        return cmd

    async def _try_execution_with_retry(
        self,
        cmd: list[str],
        cwd: str,
        timeout_seconds: int,
        retry_enabled: bool,
    ) -> tuple[ClaudeJSONResponse | None, bool]:
        """Try execution with rate limit retry.

        Returns:
            Tuple of (response or None, should_retry_infra)
        """
        while True:
            start_time = time.time()
            stdout, stderr, returncode = await self._execute_command(
                cmd, cwd, timeout_seconds
            )
            elapsed = time.time() - start_time

            if returncode != 0:
                stdout_text = stdout.decode() if stdout else ""
                stderr_text = stderr.decode() if stderr else ""

                # Classify error
                action, wait_seconds = self._classify_error(
                    stdout_text, stderr_text, returncode, elapsed, retry_enabled, cwd
                )

                if action == "retry_rate_limit":
                    await asyncio.sleep(int(wait_seconds))
                    logger.info("Wait complete, retrying...")
                    continue
                elif action == "retry_infra":
                    return None, True

            # Success
            response = self._process_response(stdout.decode())
            return response, False

    async def _execute_command(
        self,
        cmd: list[str],
        cwd: str,
        timeout_seconds: int,
    ) -> tuple[bytes, bytes, int]:
        """Execute CLI command asynchronously.

        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        # Build environment
        env = None
        if self.settings.get("__sandbox_env"):
            env = os.environ.copy()
            env.update(self.settings["__sandbox_env"])

        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            # Get prompt to pass via stdin
            prompt_input = None
            if self.settings.get("__prompt_text"):
                prompt_input = self.settings["__prompt_text"].encode("utf-8")

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt_input),
                timeout=timeout_seconds,
            )
            return stdout, stderr, process.returncode or 0

        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            raise RuntimeError(
                f"Claude CLI timeout after {timeout_seconds}s. "
                "Consider breaking into smaller tasks or increasing timeout_seconds."
            )

    def _classify_error(
        self,
        stdout_text: str,
        stderr_text: str,
        returncode: int,
        elapsed: float,
        retry_enabled: bool,
        cwd: str,
    ) -> tuple[str, float]:
        """Classify error and determine action.

        Returns:
            Tuple of (action, wait_seconds)

        Raises:
            ClaudeOAuthError: For OAuth errors
            RuntimeError: For generic errors
        """
        # Check OAuth first
        is_oauth, oauth_msg = detect_oauth_error(stdout_text, stderr_text)
        if is_oauth and oauth_msg:
            raise ClaudeOAuthError(
                f"Authentication expired after {elapsed:.1f}s: {oauth_msg}",
                reauth_instruction=oauth_msg if "/login" in oauth_msg else "Please run /login"
            )

        # Check rate limit
        if retry_enabled:
            error_output = stdout_text + "\n" + stderr_text
            is_limited, reset_time = detect_rate_limit(error_output)
            if is_limited and reset_time:
                wait_seconds = calculate_wait_time(reset_time)
                logger.info("Rate limit hit. Waiting %d minutes...", wait_seconds // 60)
                return ("retry_rate_limit", wait_seconds)

        # Check infrastructure
        if detect_cli_infrastructure_failure(stderr_text):
            return ("retry_infra", 0.0)

        # Generic error
        stderr = stderr_text if stderr_text else "(no error output)"
        if elapsed > LONG_RUNTIME_THRESHOLD_SECONDS:
            msg = (
                f"Claude CLI failed after {elapsed:.1f}s: {stderr}\n"
                f"Long runtime suggests task complexity - consider smaller tasks."
            )
        else:
            msg = f"Claude CLI error after {elapsed:.1f}s: {stderr}"
        raise RuntimeError(msg)

    def _process_response(self, raw_stdout: str) -> ClaudeJSONResponse:
        """Process CLI response.

        Args:
            raw_stdout: Raw stdout from CLI

        Returns:
            Parsed response
        """
        # Strip srt diagnostic output
        if raw_stdout.startswith("Running: "):
            first_newline = raw_stdout.find('\n')
            if first_newline > 0:
                raw_stdout = raw_stdout[first_newline + 1:]

        raw_response = json.loads(raw_stdout)

        if isinstance(raw_response, list):
            # Verbose format - find result event
            for event in raw_response:
                if isinstance(event, dict) and event.get("type") == "result":
                    response = cast(ClaudeJSONResponse, event)
                    break
            else:
                raise RuntimeError("No result event in Claude CLI output")
        else:
            response = cast(ClaudeJSONResponse, raw_response)

        # Validate
        if response.get("is_error"):
            error_msg = response.get("error", "Unknown error")
            raise RuntimeError(f"Claude CLI error: {error_msg}")

        # Save to working directory
        save_raw_response_to_working_dir(response, self.settings)

        return response
