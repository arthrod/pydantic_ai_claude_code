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
    """
    Translate internal ClaudeCodeSettings into a dict of Claude Agent SDK options.
    
    Parameters:
        settings (ClaudeCodeSettings): Source settings mapping; may include keys like "working_directory", "append_system_prompt", "allowed_tools", "dangerously_skip_permissions", and "claude_cli_path".
    
    Returns:
        dict[str, Any]: SDK-compatible options with keys such as "cwd", "system_prompt", "allowed_tools", "permission_mode", and "cli_path" populated when present.
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
        """
        Create an EnhancedCLITransport configured with the given prompt and optional settings.
        
        Parameters:
            prompt (str): The prompt text to send to Claude.
            settings (ClaudeCodeSettings | None): Optional configuration for the transport; defaults to an empty mapping when not provided.
        
        Notes:
            Initializes internal state:
              - self._working_directory is set to None.
              - self._sandbox_env is initialized as an empty dict.
        """
        self.prompt = prompt
        self.settings = settings or {}
        self._working_directory: str | None = None
        self._sandbox_env: dict[str, str] = {}

    async def execute(self) -> ClaudeJSONResponse:
        """
        Run the enhanced Claude CLI end-to-end, handling working directory setup, sandboxing, rate-limit retries, infrastructure backoff, and debug saving.
        
        Returns:
            ClaudeJSONResponse: Parsed Claude JSON response.
        
        Raises:
            ClaudeOAuthError: If OAuth reauthentication is required.
            RuntimeError: If the CLI fails after the maximum retry attempts or encounters an unrecoverable error.
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
        """
        Prepare a filesystem working directory for the current prompt and write the prompt file.
        
        Creates or reuses a base directory (a temporary base is created if none is configured), makes a numbered subdirectory for this call, copies any configured additional files into the directory, writes the prompt to `prompt.md`, saves prompt debug information, and records working paths in settings.
        
        Returns:
            str: Path to the created or reused working directory.
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
        """
        Constructs the Claude CLI command from the transport's settings and applies sandbox wrapping if enabled.
        
        Returns:
            cmd (list[str]): Command argument list for invoking the Claude CLI. If sandbox runtime is enabled, this method updates self._sandbox_env and settings["__sandbox_env"] as a side effect.
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
        """
        Attempt the CLI command and handle retryable conditions such as rate limits and infrastructure failures.
        
        Parameters:
            cmd: The CLI command and arguments to run.
            cwd: Working directory for the command.
            timeout_seconds: Maximum time to wait for the command to complete.
            retry_enabled: If True, rate-limit errors will be retried after the recommended wait; if False, rate-limit errors are treated as failures.
        
        Returns:
            (response, should_retry_infra) — `response` is the parsed Claude JSON response when execution succeeds, or `None` when an infrastructure retry is recommended; `should_retry_infra` is `True` when an infrastructure retry is suggested, `False` otherwise.
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
        """
        Run the given CLI command in a subprocess, supplying prompt text and sandboxed environment when configured.
        
        If settings contain "__prompt_text", that text is sent to the process's stdin. If settings contain "__sandbox_env", those values are merged into the process environment. Waits up to timeout_seconds for completion.
        
        Returns:
            tuple[bytes, bytes, int]: (stdout bytes, stderr bytes, process return code)
        
        Raises:
            RuntimeError: If the subprocess does not complete within timeout_seconds.
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
        """
        Determine the appropriate recovery action for a CLI failure and, when applicable, how long to wait before retrying.
        
        Checks for OAuth expiration and raises ClaudeOAuthError when reauthentication is required. If retrying on rate limits is enabled and a rate-limit is detected, returns ("retry_rate_limit", wait_seconds) where wait_seconds is the computed delay. If an infrastructure-level CLI failure is detected, returns ("retry_infra", 0.0). For other errors, raises RuntimeError with a contextual message including elapsed time and stderr output.
        
        Parameters:
            stdout_text (str): Captured standard output from the CLI process.
            stderr_text (str): Captured standard error from the CLI process.
            returncode (int): Process exit code.
            elapsed (float): Seconds elapsed while the process ran.
            retry_enabled (bool): Whether rate-limit retry logic is permitted.
            cwd (str): Working directory where the CLI was executed (used for context in messages).
        
        Returns:
            tuple[str, float]: A pair (action, wait_seconds). `action` is one of:
                - "retry_rate_limit": wait and retry after `wait_seconds`.
                - "retry_infra": indicate an infrastructure retry (wait_seconds is 0.0).
        Raises:
            ClaudeOAuthError: If output indicates an expired or invalid OAuth session and reauthentication is required.
            RuntimeError: For non-retriable failures; message includes elapsed time and stderr output.
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
        """
        Parse Claude CLI stdout into a ClaudeJSONResponse.
        
        Strips an initial "Running: " diagnostic line if present, parses the JSON output, and when the output is a list selects the event with type "result". Validates that the parsed response does not indicate an error and saves the raw response to the configured working directory.
        
        Parameters:
            raw_stdout (str): Raw stdout text produced by the Claude CLI.
        
        Returns:
            ClaudeJSONResponse: The parsed response object extracted from the CLI output.
        
        Raises:
            RuntimeError: If a "result" event cannot be found in verbose (list) output, or if the parsed response indicates an error.
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