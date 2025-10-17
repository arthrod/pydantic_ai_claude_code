"""Claude Code model implementation for Pydantic AI."""

from __future__ import annotations as _annotations

import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from .messages import format_messages_for_claude
from .provider import ClaudeCodeProvider
from .streamed_response import ClaudeCodeStreamedResponse
from .streaming import run_claude_streaming
from .structure_converter import (
    build_structure_instructions,
    read_structure_from_filesystem,
)
from .types import ClaudeCodeSettings, ClaudeJSONResponse
from .utils import build_claude_command, run_claude_async

logger = logging.getLogger(__name__)


class ClaudeCodeModel(Model):
    """Pydantic AI model implementation using Claude Code CLI.

    This model wraps the local Claude CLI to provide a Pydantic AI compatible
    interface, supporting all features available through the CLI including:
    - Structured responses via output validation
    - Tool/function calling (via Claude's built-in tools)
    - Web search (via Claude's WebSearch tool)
    - Multi-turn conversations
    """

    def __init__(
        self,
        model_name: str = "sonnet",
        *,
        provider: ClaudeCodeProvider | None = None,
    ):
        """Initialize Claude Code model.

        Args:
            model_name: Name of the Claude model to use (e.g., "sonnet", "opus",
                or full model name like "claude-sonnet-4-5-20250929")
            provider: Optional provider for managing Claude CLI execution
        """
        self._model_name = model_name
        self.provider = provider or ClaudeCodeProvider()

        logger.debug("Initialized ClaudeCodeModel with model_name=%s", model_name)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return f"claude-code:{self._model_name}"

    @property
    def system(self) -> str:
        """Get the system identifier."""
        return "claude-code"

    def _build_unstructured_output_instruction(
        self, settings: ClaudeCodeSettings
    ) -> str:
        """Build instruction for unstructured (text) output via file.

        Args:
            settings: Settings dict to store output file path

        Returns:
            Instruction string to append to system prompt
        """
        # Generate unique filename for unstructured output
        output_filename = f"/tmp/claude_unstructured_output_{uuid.uuid4().hex}.txt"
        settings["__unstructured_output_file"] = output_filename

        logger.debug("Unstructured output file path: %s", output_filename)

        instruction = f"""# Output Instructions

## File Path

```
{output_filename}
```

---

## Requirements

1. **Use the Write tool** to create the file with your complete response
2. **Content rules:**
   - Include ONLY your direct answer
   - No preambles or introductions
   - No meta-commentary about what you're doing
   - Just the answer itself

---

> **IMPORTANT:** All output must go to the file path specified above."""

        return instruction

    def _build_structured_output_instruction(
        self, output_tool: Any, settings: ClaudeCodeSettings
    ) -> str:
        """Build instruction for structured output using improved converter.

        Args:
            output_tool: The output tool definition
            settings: Settings dict to store output file path

        Returns:
            Instruction string to append to system prompt
        """
        schema = output_tool.parameters_json_schema

        # Generate unique filename for structured output
        output_filename = f"/tmp/claude_structured_output_{uuid.uuid4().hex}.json"
        settings["__structured_output_file"] = output_filename

        logger.debug("Structured output file path: %s", output_filename)

        # Generate unique temp dir for field data
        temp_data_dir = f"/tmp/claude_data_structure_{uuid.uuid4().hex[:8]}"
        settings["__temp_json_dir"] = temp_data_dir

        # Use new structure converter to build instructions
        return build_structure_instructions(schema, temp_data_dir)

    def _check_has_tool_results(self, messages: list[ModelMessage]) -> bool:
        """Check if messages contain tool results.

        Args:
            messages: List of messages to check

        Returns:
            True if any message contains ToolReturnPart
        """
        return any(
            isinstance(part, ToolReturnPart)
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

    def _build_function_tools_prompt(
        self, function_tools: list[Any]
    ) -> tuple[str, dict[str, Any]]:
        """Build function selection prompt and available functions dict.

        Args:
            function_tools: List of function tool definitions

        Returns:
            Tuple of (prompt string, available functions dict)
        """
        logger.info("=" * 80)
        logger.info("BUILDING FUNCTION TOOLS PROMPT - Total tools: %d", len(function_tools))
        logger.info("=" * 80)

        option_descriptions = []
        for i, tool in enumerate(function_tools, 1):
            desc = tool.description or "No description"
            schema = tool.parameters_json_schema
            logger.info("Tool %d: %s", i, tool.name)
            logger.info("  Description: %s", desc)
            logger.info("  Schema: %s", json.dumps(schema, indent=2))

            params = schema.get("properties", {})
            param_hints = []
            for param_name, param_schema in params.items():
                param_type = param_schema.get("type", "unknown")
                param_hints.append(f"{param_name}: {param_type}")
            params_str = (
                f" (parameters: {', '.join(param_hints)})" if param_hints else ""
            )
            option_descriptions.append(f"{i}. {tool.name}{params_str} - {desc}")
        option_descriptions.append(
            f"{len(function_tools) + 1}. none - Answer directly without calling any function"
        )

        prompt = f"""# Function Selection Task

## Your Role

You are **SELECTING** which function(s) to use - NOT executing them.

---

## Available Functions

{chr(10).join(option_descriptions)}

---

## Instructions

### Step 1: Analyze the Request
Read the user's request carefully to understand what information or action is needed.

### Step 2: Make Your Decision
Determine if you need to call function(s) (options 1-{len(function_tools)}) or can answer directly (option {len(function_tools) + 1}).

### Step 3: Respond with Exact Format

#### Single Function
```
CHOICE: function_name
```

#### Multiple Functions
```
CHOICE: function_name1
CHOICE: function_name2
```

#### No Function Needed
```
CHOICE: none
```

**Example:** `CHOICE: {function_tools[0].name if function_tools else "none"}`

---

> **CRITICAL:**
> - Do NOT include explanations or reasoning
> - Do NOT try to execute these functions - they are not built-in tools
> - You are ONLY making a selection"""

        available_functions = {tool.name: tool for tool in function_tools}
        return prompt, available_functions

    def _build_system_prompt_parts(
        self,
        model_request_parameters: ModelRequestParameters,
        has_tool_results: bool,
        settings: ClaudeCodeSettings,
        is_streaming: bool = False,
    ) -> list[str]:
        """Build system prompt parts for request.

        Args:
            model_request_parameters: Request parameters (includes output_tools and function_tools)
            has_tool_results: Whether messages contain tool results
            settings: Settings dict (will be modified)
            is_streaming: Whether this is a streaming request (skips file output instructions)

        Returns:
            List of system prompt parts
        """
        system_prompt_parts = []
        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )

        # Only include user's custom system prompt if we don't have tool results yet
        if (
            not has_tool_results
            and model_request_parameters
            and hasattr(model_request_parameters, "system_prompt")
        ):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        # Add function tools prompt ONLY if there are no tool results yet
        if function_tools and not has_tool_results:
            function_selection_prompt, available_functions = (
                self._build_function_tools_prompt(function_tools)
            )
            settings["__function_selection_mode__"] = True
            settings["__available_functions__"] = available_functions
            system_prompt_parts.append(function_selection_prompt)

        # Add output instructions (only if not in function call mode)
        # Skip file writing instructions for streaming - we need direct text output
        if not is_streaming:
            if output_tools and not function_tools:
                json_instruction = self._build_structured_output_instruction(
                    output_tools[0], settings
                )
                system_prompt_parts.append(json_instruction)
            elif not function_tools:
                unstructured_instruction = self._build_unstructured_output_instruction(
                    settings
                )
                system_prompt_parts.append(unstructured_instruction)

        return system_prompt_parts

    def _assemble_final_prompt(
        self,
        messages: list[ModelMessage],
        system_prompt_parts: list[str],
        settings: ClaudeCodeSettings,
        has_tool_results: bool,
    ) -> str:
        """Assemble final prompt with system instructions.

        Args:
            messages: Message list
            system_prompt_parts: System prompt parts to prepend
            settings: Settings dict
            has_tool_results: Whether to skip system prompt in messages

        Returns:
            Final assembled prompt string
        """
        prompt = format_messages_for_claude(
            messages, skip_system_prompt=has_tool_results
        )
        logger.debug("Formatted prompt length: %d chars", len(prompt))

        # Prepend system instructions
        if system_prompt_parts:
            combined_system_prompt = "\n\n".join(system_prompt_parts)
            prompt = f"{combined_system_prompt}\n\n{prompt}"
            logger.debug(
                "Added %d chars of system instructions to prompt",
                len(combined_system_prompt),
            )

        # Include user-specified append_system_prompt
        existing_prompt = settings.get("append_system_prompt")
        if existing_prompt:
            prompt = f"{existing_prompt}\n\n{prompt}"
            settings.pop("append_system_prompt", None)
            logger.debug(
                "Added %d chars of user system prompt to prompt file",
                len(existing_prompt),
            )

        return prompt

    def _create_model_response_with_usage(
        self,
        response: ClaudeJSONResponse,
        parts: list[TextPart | ToolCallPart],
    ) -> ModelResponse:
        """Create ModelResponse with usage from Claude response.

        Args:
            response: Claude response to extract usage from
            parts: Response parts (TextPart, ToolCallPart, etc.)

        Returns:
            ModelResponse with usage and timestamp
        """
        usage = ClaudeCodeModel._create_usage(response)
        model_name = self._get_model_name(response)
        return ModelResponse(
            parts=parts,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            usage=usage,
        )

    async def _handle_unstructured_follow_up(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Handle unstructured follow-up request when function selection is 'none'.

        Args:
            messages: Original message list
            model_request_parameters: Request parameters

        Returns:
            Model response with unstructured output
        """
        logger.info(
            "Function selection was 'none', making follow-up unstructured request"
        )

        unstructured_settings = self.provider.get_settings(model=self._model_name)
        system_prompt_parts = []

        if model_request_parameters and hasattr(
            model_request_parameters, "system_prompt"
        ):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        unstructured_instruction = self._build_unstructured_output_instruction(
            unstructured_settings
        )
        system_prompt_parts.append(unstructured_instruction)

        unstructured_prompt = self._assemble_final_prompt(
            messages, system_prompt_parts, unstructured_settings, has_tool_results=False
        )

        logger.debug(
            "Making unstructured request with prompt length: %d",
            len(unstructured_prompt),
        )
        unstructured_response = await run_claude_async(
            unstructured_prompt, settings=unstructured_settings
        )

        return self._convert_response(
            unstructured_response,
            output_tools=[],
            function_tools=[],
            settings=unstructured_settings,
        )

    def _build_retry_prompt(
        self,
        messages: list[ModelMessage],
        schema: dict[str, Any],
        arg_settings: ClaudeCodeSettings,
        error_msg: str,
    ) -> str:
        """Build prompt for retry attempt after validation error.

        Args:
            messages: Original message list
            schema: JSON schema for function parameters
            arg_settings: Settings dict (will be modified with new temp directory)
            error_msg: Error message from previous attempt

        Returns:
            Retry prompt string
        """
        # Generate new temp directory for retry
        temp_data_dir = f"/tmp/claude_data_structure_{uuid.uuid4().hex[:8]}"
        arg_settings["__temp_json_dir"] = temp_data_dir

        # Rebuild instruction with new temp directory
        instruction = self._build_argument_collection_instruction(
            schema, arg_settings
        )

        retry_instruction = f"""
PREVIOUS ATTEMPT HAD ERRORS:
{error_msg}

Please fix the issues above and try again. Follow the directory structure instructions carefully."""

        return f"{instruction}\n\n{format_messages_for_claude(messages, skip_system_prompt=True)}\n\n{retry_instruction}"

    async def _try_collect_arguments(
        self,
        current_prompt: str,
        arg_settings: ClaudeCodeSettings,
        selected_function: str,
        schema: dict[str, Any],
    ) -> tuple[ModelResponse | None, str | None, ClaudeJSONResponse]:
        """Try to collect arguments for selected function.

        Args:
            current_prompt: Prompt to send to Claude
            arg_settings: Settings for argument collection
            selected_function: Name of selected function
            schema: JSON schema for function parameters

        Returns:
            Tuple of (model_response, error_msg, claude_response).
            model_response is not None on success, error_msg is not None on retriable error.
        """
        arg_response = await run_claude_async(current_prompt, settings=arg_settings)

        # Read structured output from directory structure
        structured_file = arg_settings.get("__structured_output_file")
        if structured_file:
            parsed_args, error_msg = self._read_structured_output_file(
                structured_file, schema, arg_settings
            )

            if parsed_args:
                logger.info(
                    "Successfully collected arguments for %s: %s",
                    selected_function,
                    parsed_args,
                )
                tool_call = ToolCallPart(
                    tool_name=str(selected_function),
                    args=parsed_args,
                    tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                )
                return (
                    self._create_model_response_with_usage(arg_response, [tool_call]),
                    None,
                    arg_response,
                )

            return None, error_msg, arg_response

        return None, None, arg_response

    def _setup_argument_collection(
        self,
        messages: list[ModelMessage],
        selected_function: str,
        available_functions: dict[str, Any],
        arg_response_for_usage: ClaudeJSONResponse,
    ) -> tuple[ModelResponse | None, ClaudeCodeSettings, dict[str, Any], str]:
        """Set up argument collection by validating function and building initial prompt.

        Args:
            messages: Original message list
            selected_function: Name of selected function
            available_functions: Dict of available function definitions
            arg_response_for_usage: Response to extract usage from

        Returns:
            Tuple of (error_response, arg_settings, schema, arg_prompt).
            If error_response is not None, return it immediately.
        """
        logger.info(
            "Function '%s' was selected, collecting arguments", selected_function
        )

        tool_def = available_functions.get(selected_function)
        if not tool_def:
            return (
                self._create_model_response_with_usage(
                    arg_response_for_usage,
                    [TextPart(content=f"Error: Function '{selected_function}' not found")],
                ),
                {},
                {},
                "",
            )

        arg_settings = self.provider.get_settings(model=self._model_name)
        schema = tool_def.parameters_json_schema

        # Build initial prompt
        instruction = self._build_argument_collection_instruction(schema, arg_settings)
        arg_prompt = format_messages_for_claude(messages, skip_system_prompt=True)
        arg_prompt = f"{instruction}\n\n{arg_prompt}"

        existing_prompt = arg_settings.get("append_system_prompt")
        if existing_prompt:
            arg_prompt = f"{existing_prompt}\n\n{arg_prompt}"
            arg_settings.pop("append_system_prompt", None)

        return None, arg_settings, schema, arg_prompt

    async def _handle_argument_collection(
        self,
        messages: list[ModelMessage],
        selected_function: str,
        available_functions: dict[str, Any],
        arg_response_for_usage: ClaudeJSONResponse,
    ) -> ModelResponse:
        """Handle argument collection for selected function using file/folder structure.

        Args:
            messages: Original message list
            selected_function: Name of selected function
            available_functions: Dict of available function definitions
            arg_response_for_usage: Response to extract usage from

        Returns:
            Model response with tool call or error
        """
        error_response, arg_settings, schema, arg_prompt = (
            self._setup_argument_collection(
                messages, selected_function, available_functions, arg_response_for_usage
            )
        )

        if error_response:
            return error_response

        max_retries = 1
        error_msg = None

        for attempt in range(max_retries + 1):
            if attempt == 0:
                logger.info("=" * 80)
                logger.info("PHASE 2: ARGUMENT COLLECTION - Prompt length: %d", len(arg_prompt))
                logger.info("=" * 80)
                logger.info("%s", arg_prompt)
                logger.info("=" * 80)
                current_prompt = arg_prompt
            else:
                logger.info(
                    "Retrying argument collection (attempt %d/%d) after validation error",
                    attempt + 1,
                    max_retries + 1,
                )
                current_prompt = self._build_retry_prompt(
                    messages, schema, arg_settings, error_msg
                )
                logger.info("=" * 80)
                logger.info("RETRY PROMPT - length: %d", len(current_prompt))
                logger.info("=" * 80)
                logger.info("%s", current_prompt)
                logger.info("=" * 80)

            model_response, error_msg, arg_response = await self._try_collect_arguments(
                current_prompt,
                arg_settings,
                selected_function,
                schema,
            )

            if model_response or error_msg:
                logger.info("=" * 80)
                logger.info("PHASE 2: ARGUMENT COLLECTION RESULT")
                logger.info("=" * 80)
                if model_response:
                    logger.info("Success! Extracted args: %s", model_response.parts[0].args if model_response.parts else "N/A")
                if error_msg:
                    logger.info("Error: %s", error_msg)
                logger.info("=" * 80)

            if model_response:
                return model_response

            # If error and not last attempt, retry
            if error_msg and attempt < max_retries:
                logger.warning(
                    "Argument extraction failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    error_msg,
                )
                continue

            # Last attempt failed, return error to user
            if error_msg:
                logger.error(
                    "Argument extraction failed after %d attempts: %s",
                    attempt + 1,
                    error_msg,
                )
                return self._create_model_response_with_usage(
                    arg_response,
                    [TextPart(content=error_msg)],
                )

        # Fallback to error if file reading failed
        result_text = arg_response.get("result", "")
        error_msg = (
            f"Could not interpret the parameters from response: {result_text[:500]}"
        )
        return self._create_model_response_with_usage(
            arg_response,
            [TextPart(content=error_msg)],
        )

    def _build_argument_collection_instruction(
        self, schema: dict[str, Any], settings: ClaudeCodeSettings
    ) -> str:
        """Build instruction for argument collection using file/folder structure.

        This reuses the structured output approach that has proven reliable.

        Args:
            schema: JSON schema for function parameters
            settings: Settings dict to store file paths

        Returns:
            Instruction string with file/folder structure
        """
        # Generate unique filename and temp dir (same as structured output)
        output_filename = f"/tmp/claude_structured_output_{uuid.uuid4().hex}.json"
        settings["__structured_output_file"] = output_filename

        temp_data_dir = f"/tmp/claude_data_structure_{uuid.uuid4().hex[:8]}"
        settings["__temp_json_dir"] = temp_data_dir

        logger.debug("Argument collection will use temp directory: %s", temp_data_dir)

        # Use new structure converter to build instructions
        return build_structure_instructions(schema, temp_data_dir)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make an async request to Claude Code CLI.

        Args:
            messages: List of messages in the conversation
            model_settings: Optional model settings from Agent (can include timeout_seconds, etc.)
            model_request_parameters: Model request parameters

        Returns:
            Model response with embedded usage information
        """
        logger.info(
            "Starting non-streaming request with %d messages, output_tools=%s, function_tools=%s",
            len(messages),
            len(model_request_parameters.output_tools)
            if model_request_parameters and model_request_parameters.output_tools
            else 0,
            len(model_request_parameters.function_tools)
            if model_request_parameters and model_request_parameters.function_tools
            else 0,
        )

        # Get settings from provider and merge with model_settings from pydantic_ai
        settings = self.provider.get_settings(model=self._model_name)
        if model_settings:
            # Merge model_settings into provider settings (model_settings takes precedence)
            settings.update(model_settings)
        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )

        # Check if we have tool results in the conversation
        has_tool_results = self._check_has_tool_results(messages)

        # Build system prompt with appropriate instructions
        system_prompt_parts = self._build_system_prompt_parts(
            model_request_parameters,
            has_tool_results,
            settings,
        )

        # Assemble final prompt with system instructions
        prompt = self._assemble_final_prompt(
            messages, system_prompt_parts, settings, has_tool_results
        )

        # Run Claude CLI and convert response
        logger.info("=" * 80)
        logger.info("FULL PROMPT BEING SENT TO CLAUDE:")
        logger.info("=" * 80)
        logger.info("%s", prompt)
        logger.info("=" * 80)
        response = await run_claude_async(prompt, settings=settings)
        logger.info("=" * 80)
        logger.info("CLAUDE RESPONSE:")
        logger.info("=" * 80)
        logger.info("%s", json.dumps(response, indent=2))
        logger.info("=" * 80)
        result = self._convert_response(
            response,
            output_tools=output_tools,
            function_tools=function_tools,
            settings=settings,
        )

        # Handle function selection follow-ups if needed
        logger.debug(
            "After _convert_response: function_tools=%s, __function_selection_mode__=%s, result.parts=%s",
            bool(function_tools),
            settings.get("__function_selection_mode__"),
            [type(p).__name__ for p in result.parts],
        )

        if (
            function_tools
            and settings.get("__function_selection_mode__")
            and len(result.parts) == 1
            and isinstance(result.parts[0], TextPart)
        ):
            content = result.parts[0].content

            if "[Function selection: none" in content:
                return await self._handle_unstructured_follow_up(
                    messages, model_request_parameters
                )

            if "[Function selected:" in content:
                selected_function = settings.get("__selected_function__")
                if selected_function:
                    available_functions = settings.get("__available_functions__", {})
                    if isinstance(available_functions, dict):
                        return await self._handle_argument_collection(
                            messages, selected_function, available_functions, response
                        )

        return result

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        _run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to Claude Code CLI.

        Args:
            messages: List of messages in the conversation
            model_settings: Optional model settings from Agent (can include timeout_seconds, etc.)
            model_request_parameters: Model request parameters
            run_context: Optional run context

        Yields:
            Streamed response object
        """
        logger.info(
            "Starting streaming request with %d messages, output_tools=%s, function_tools=%s",
            len(messages),
            len(model_request_parameters.output_tools)
            if model_request_parameters and model_request_parameters.output_tools
            else 0,
            len(model_request_parameters.function_tools)
            if model_request_parameters and model_request_parameters.function_tools
            else 0,
        )

        # Get settings from provider and merge with model_settings from pydantic_ai
        settings = self.provider.get_settings(model=self._model_name)
        if model_settings:
            # Merge model_settings into provider settings (model_settings takes precedence)
            settings.update(model_settings)
        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )

        # Streaming is only supported for plain text responses
        if output_tools:
            raise ValueError(
                "Streaming is not supported with structured output (output_tools). "
                "Structured output requires file-based JSON construction which is incompatible with streaming."
            )
        if function_tools:
            raise ValueError(
                "Streaming is not supported with function tools. "
                "Function calling requires file-based argument collection which is incompatible with streaming."
            )

        # Check if we have tool results in the conversation
        has_tool_results = self._check_has_tool_results(messages)

        # Build system prompt with appropriate instructions
        # Pass is_streaming=True to skip file writing instructions
        system_prompt_parts = self._build_system_prompt_parts(
            model_request_parameters,
            has_tool_results,
            settings,
            is_streaming=True,
        )

        # Assemble final prompt with system instructions
        prompt = self._assemble_final_prompt(
            messages, system_prompt_parts, settings, has_tool_results
        )

        # Add streaming marker instruction
        streaming_marker = "<<<STREAM_START>>>"
        prompt = f"""IMPORTANT: After completing any tool use, begin your final response with the exact marker: {streaming_marker}

Then provide your complete response after the marker.

{prompt}"""
        settings["__streaming_marker__"] = streaming_marker

        # Setup working directory for streaming
        import tempfile

        cwd = settings.get("working_directory")
        if not cwd:
            cwd = tempfile.mkdtemp(prefix="claude_prompt_")

        Path(cwd).mkdir(parents=True, exist_ok=True)
        prompt_file = Path(cwd) / "prompt.md"
        prompt_file.write_text(prompt)

        # Build command and create event stream
        cmd = build_claude_command(settings=settings, output_format="stream-json")
        event_stream = run_claude_streaming(cmd, cwd=cwd)

        # Create and yield streaming response
        streamed_response = ClaudeCodeStreamedResponse(
            model_request_parameters=model_request_parameters,
            model_name=f"claude-code:{self._model_name}",
            event_stream=event_stream,
            timestamp=datetime.now(timezone.utc),
            streaming_marker=streaming_marker,
        )

        yield streamed_response

    def _handle_function_selection_response(
        self,
        result_text: str,
        response: ClaudeJSONResponse,
        settings: ClaudeCodeSettings,
    ) -> ModelResponse:
        """Handle function selection mode response parsing.

        Args:
            result_text: Response text from Claude
            response: Full Claude response for usage extraction
            settings: Settings dict (will be modified with selected function)

        Returns:
            ModelResponse with function selection result
        """
        logger.debug("Function selection response text: %s", result_text[:500])

        # Get available function names for validation
        available_functions = settings.get("__available_functions__", {})
        if isinstance(available_functions, dict):
            valid_options = [name.lower() for name in available_functions] + ["none"]
        else:
            valid_options = ["none"]

        # Look for "CHOICE:" format - handle markdown bold/italic formatting and multiple choices
        matched_option = None
        import re

        # Extract ALL CHOICE lines using findall (handles multiple tool selection)
        choice_matches = re.findall(
            r"CHOICE:\s*[\*_]*(\w+)[\*_]*", result_text, re.IGNORECASE
        )

        if choice_matches:
            # Validate all extracted choices
            valid_choices = [
                c.strip().lower()
                for c in choice_matches
                if c.strip().lower() in valid_options
            ]

            if valid_choices:
                # Take the FIRST valid choice for THIS iteration
                matched_option = valid_choices[0]

                # Log if multiple functions were identified
                if len(valid_choices) > 1:
                    logger.info(
                        "Multiple functions identified: %s. Processing '%s' first. "
                        "Agent will handle remaining tools in subsequent iterations.",
                        ", ".join(valid_choices),
                        matched_option,
                    )

        if matched_option:
            logger.debug(
                "Function selection result: selected_function=%s", matched_option
            )

            parts: list[TextPart | ToolCallPart] = []
            if matched_option == "none":
                # Claude chose to answer directly - signal needs unstructured response
                logger.info(
                    "Function selection returned 'none' - will trigger unstructured response"
                )
                parts.append(
                    TextPart(content="[Function selection: none - answering directly]")
                )
            else:
                # Claude selected a function - signal needs argument collection
                logger.info(
                    "Function selected: %s - will trigger argument collection",
                    matched_option,
                )
                parts.append(
                    TextPart(
                        content=f"[Function selected: {matched_option} - collecting arguments]"
                    )
                )
                # Store selected function for next request
                settings["__selected_function__"] = matched_option

            return self._create_model_response_with_usage(response, parts)

        # Could not parse option selection
        logger.error(
            "Could not parse function selection from response: %s", result_text[:200]
        )
        parts = [
            TextPart(
                content=f"Error: Could not parse option selection. Response: {result_text}"
            )
        ]
        return self._create_model_response_with_usage(response, parts)

    def _handle_structured_output_response(
        self,
        result_text: str,
        response: ClaudeJSONResponse,
        output_tools: list[Any],
        settings: ClaudeCodeSettings | None,
    ) -> ModelResponse:
        """Handle structured output response via tool call.

        Args:
            result_text: Response text from Claude
            response: Full Claude response for usage extraction
            output_tools: List of output tool definitions
            settings: Settings dict

        Returns:
            ModelResponse with tool call or error text
        """
        output_tool = output_tools[0]
        tool_name = output_tool.name
        schema = output_tool.parameters_json_schema

        try:
            # Check if Claude created a structured output file
            structured_file = (
                settings.get("__structured_output_file") if settings else None
            )

            if structured_file:
                parsed_data, error_msg = self._read_structured_output_file(
                    structured_file, schema, settings
                )

                if error_msg:
                    # Return error as text so Pydantic AI can retry
                    return self._create_model_response_with_usage(
                        response, [TextPart(content=error_msg)]
                    )

                if parsed_data:
                    # Validation passed, create tool call
                    logger.debug("Successfully created structured output from file")
                    tool_call = ToolCallPart(
                        tool_name=tool_name,
                        args=parsed_data,
                        tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                    )
                    return self._create_model_response_with_usage(response, [tool_call])

                # No file found, use fallback extraction
                logger.warning(
                    "Structured output file not found, using fallback JSON extraction"
                )
                parsed_data = self._extract_json_robust(result_text, schema)
                tool_call = ToolCallPart(
                    tool_name=tool_name,
                    args=parsed_data,
                    tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                )
                return self._create_model_response_with_usage(response, [tool_call])

            # Fallback: Use robust extraction with multiple strategies
            logger.debug(
                "No structured output file configured, using robust JSON extraction"
            )
            parsed_data = self._extract_json_robust(result_text, schema)
            tool_call = ToolCallPart(
                tool_name=tool_name,
                args=parsed_data,
                tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
            )
            return self._create_model_response_with_usage(response, [tool_call])

        except json.JSONDecodeError as e:
            # If JSON parsing fails, return as text
            # Pydantic AI will retry with validation error
            logger.error("Failed to parse structured output JSON: %s", e)
            return self._create_model_response_with_usage(
                response, [TextPart(content=result_text)]
            )

    def _handle_unstructured_output_response(
        self,
        result_text: str,
        response: ClaudeJSONResponse,
        settings: ClaudeCodeSettings | None,
    ) -> ModelResponse:
        """Handle unstructured text output response.

        Args:
            result_text: Response text from Claude
            response: Full Claude response for usage extraction
            settings: Settings dict

        Returns:
            ModelResponse with text content
        """
        logger.debug("Processing unstructured output")

        # Check if we instructed Claude to write to a file
        unstructured_file_obj = (
            settings.get("__unstructured_output_file") if settings else None
        )
        unstructured_file = (
            str(unstructured_file_obj) if unstructured_file_obj else None
        )

        if unstructured_file and Path(unstructured_file).exists():
            # Read content from file that Claude created
            try:
                logger.debug(
                    "Reading unstructured output from file: %s", unstructured_file
                )
                with open(unstructured_file) as f:
                    file_content = f.read()
                logger.debug(
                    "Successfully read %d bytes from unstructured output file",
                    len(file_content),
                )
                return self._create_model_response_with_usage(
                    response, [TextPart(content=file_content)]
                )
            except Exception as e:
                # Fallback to CLI response if file read fails
                logger.warning(
                    "Failed to read unstructured output file, using CLI response: %s", e
                )
                return self._create_model_response_with_usage(
                    response, [TextPart(content=result_text)]
                )

        # Fallback to CLI response if no file
        if unstructured_file:
            logger.warning("Unstructured output file not found: %s", unstructured_file)
        logger.debug("Using CLI response text for unstructured output")
        return self._create_model_response_with_usage(
            response, [TextPart(content=result_text)]
        )

    def _convert_response(
        self,
        response: ClaudeJSONResponse,
        output_tools: list[Any] | None = None,
        function_tools: list[Any] | None = None,
        settings: ClaudeCodeSettings | None = None,
    ) -> ModelResponse:
        """Convert Claude JSON response to ModelResponse.

        Args:
            response: Claude CLI JSON response
            output_tools: Optional output tool definitions for structured output
            function_tools: Optional function tool definitions for tool calling
            settings: Settings that may contain structured output file path

        Returns:
            Pydantic AI ModelResponse with embedded usage
        """
        # Extract result text
        result_text = response.get("result", "")

        # Check for function selection mode
        if function_tools and settings and settings.get("__function_selection_mode__"):
            return self._handle_function_selection_response(
                result_text, response, settings
            )

        # Check for structured output
        if output_tools and len(output_tools) > 0:
            return self._handle_structured_output_response(
                result_text, response, output_tools, settings
            )

        # Default to unstructured output
        return self._handle_unstructured_output_response(
            result_text, response, settings
        )

    def _cleanup_temp_file(self, file_path: str | Path) -> None:
        """Safely remove temporary file.

        Args:
            file_path: Path to file to remove
        """
        with contextlib.suppress(Exception):
            Path(file_path).unlink()

    def _validate_json_schema(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> str | None:
        """Validate JSON data against schema.

        Args:
            data: JSON data to validate
            schema: JSON schema to validate against

        Returns:
            Error message if validation fails, None if valid
        """
        # Check required fields
        required_fields = schema.get("required", [])
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return f"Please provide: {', '.join(missing_fields)}\nCurrent content: {json.dumps(data)}"

        # Validate field types
        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name in data:
                expected_type = field_schema.get("type")
                actual_value = data[field_name]

                # Type checking
                type_valid = True
                if (
                    expected_type == "string"
                    and not isinstance(actual_value, str)
                    or expected_type == "integer"
                    and not isinstance(actual_value, int)
                    or expected_type == "number"
                    and not isinstance(actual_value, (int, float))
                    or expected_type == "boolean"
                    and not isinstance(actual_value, bool)
                    or expected_type == "array"
                    and not isinstance(actual_value, list)
                    or expected_type == "object"
                    and not isinstance(actual_value, dict)
                ):
                    type_valid = False

                if not type_valid:
                    return f"The value for '{field_name}' should be a {expected_type}, but it's a {type(actual_value).__name__}\nCurrent content: {json.dumps(data)}"

        return None

    def _cleanup_temp_directory(self, temp_path: Path) -> None:
        """Clean up temporary directory.

        Args:
            temp_path: Path to temporary directory to remove
        """
        import shutil

        shutil.rmtree(temp_path, ignore_errors=True)

    def _try_read_directory_structure(
        self,
        settings: ClaudeCodeSettings | None,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Try to read JSON from directory structure.

        Args:
            settings: Settings that may contain temp directory path
            schema: JSON schema to validate against

        Returns:
            Tuple of (parsed_data, error_message). One will be None.
        """
        temp_json_dir = settings.get("__temp_json_dir") if settings else None
        if not temp_json_dir or not isinstance(temp_json_dir, str):
            return None, None

        temp_path = Path(temp_json_dir)

        # Check if directory exists (no need for completion marker since CLI execution is synchronous)
        if not temp_path.exists():
            return None, None

        logger.debug("Found temp directory structure at: %s", temp_path)

        try:
            # Use new structure converter to read and validate
            parsed_data = read_structure_from_filesystem(schema, temp_path)
            logger.debug("Successfully read JSON from directory structure")
            return parsed_data, None

        except RuntimeError as e:
            # RuntimeError contains user-friendly error messages from converter
            logger.error("Failed to read directory structure: %s", e)
            return None, str(e)
        except Exception as e:
            logger.error("Unexpected error reading directory: %s", e)
            return None, f"Could not read the data structure: {e}"

    def _try_read_json_file(
        self,
        file_path: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Try to read and parse JSON from file.

        Args:
            file_path: Path to JSON file
            schema: JSON schema to validate against

        Returns:
            Tuple of (parsed_data, error_message). One will be None.
        """
        if not Path(file_path).exists():
            logger.debug("Structured output file not found: %s", file_path)
            return None, None

        logger.debug("Reading structured output file: %s", file_path)

        # Read file
        try:
            with open(file_path) as f:
                file_content = f.read()
            logger.debug("Read %d bytes from structured output file", len(file_content))
        except Exception as e:
            logger.error("Failed to read structured output file: %s", e)
            return None, f"Failed to read file: {e}"

        # Parse JSON
        try:
            parsed_data = json.loads(file_content)
            logger.debug("Successfully parsed JSON from structured output file")
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in structured output file: %s", e)
            return (
                None,
                f"The file content isn't formatted correctly: {e}\nFile content:\n{file_content}",
            )

        # Validate schema
        validation_error = self._validate_json_schema(parsed_data, schema)
        if validation_error:
            logger.error("Schema validation failed: %s", validation_error)
            return None, validation_error

        # Validation passed
        logger.debug("Structured output validated successfully")
        return parsed_data, None

    def _read_structured_output_file(
        self,
        file_path: str,
        schema: dict[str, Any],
        settings: ClaudeCodeSettings | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Read and validate structured output file.

        Args:
            file_path: Path to structured output file
            schema: JSON schema to validate against
            settings: Settings that may contain temp directory path

        Returns:
            Tuple of (parsed_data, error_message). One will be None.
        """
        # Try reading from directory structure first
        parsed_data, error_msg = self._try_read_directory_structure(settings, schema)
        if parsed_data is not None or error_msg is not None:
            return parsed_data, error_msg

        # Fall back to reading JSON file
        return self._try_read_json_file(file_path, schema)

    def _try_extract_from_markdown(self, text: str) -> dict[str, Any] | None:
        """Try to extract JSON by stripping markdown code blocks.

        Args:
            text: Text that may contain markdown-wrapped JSON

        Returns:
            Parsed JSON dict if successful, None otherwise
        """
        cleaned = text.strip()

        # Remove markdown code blocks (```json or ```)
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    def _try_extract_json_object(self, text: str) -> dict[str, Any] | None:
        """Try to extract JSON object using regex pattern matching.

        Args:
            text: Text that may contain JSON object

        Returns:
            Parsed JSON dict if successful, None otherwise
        """
        import re

        json_pattern = r"\{(?:[^{}]|\{[^{}]*\})*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    def _try_extract_json_array(
        self, text: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Try to extract JSON array and wrap in single-field schema.

        Args:
            text: Text that may contain JSON array
            schema: JSON schema to validate against

        Returns:
            Wrapped JSON dict if successful, None otherwise
        """
        import re

        array_pattern = r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]"
        array_matches = re.findall(array_pattern, text, re.DOTALL)

        for match in array_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    properties = schema.get("properties", {})
                    if len(properties) == 1:
                        field_name = list(properties.keys())[0]
                        return {field_name: parsed}
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _convert_primitive_value(
        value: str, field_type: str
    ) -> int | float | bool | str | None:
        """Convert string value to typed primitive.

        Args:
            value: String value to convert
            field_type: Target type (integer, number, boolean, string)

        Returns:
            Converted value or None if conversion fails
        """
        try:
            if field_type == "integer":
                return int(value)
            if field_type == "number":
                return float(value)
            if field_type == "boolean":
                return value.lower() in ("true", "1", "yes")
            if field_type == "string":
                return value
        except (ValueError, AttributeError):
            pass

        return None

    def _try_single_field_autowrap(
        self, text: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Try to auto-wrap value for single-field schemas.

        Args:
            text: Text containing a value to wrap
            schema: JSON schema (must have single field)

        Returns:
            Wrapped JSON dict if successful, None otherwise
        """
        properties = schema.get("properties", {})
        if len(properties) != 1:
            return None

        field_name = list(properties.keys())[0]
        field_type = properties[field_name].get("type")

        cleaned = text.strip()

        # Try parsing as JSON first (could be array/object)
        try:
            parsed_value = json.loads(cleaned)
            if (
                field_type == "array"
                and isinstance(parsed_value, list)
                or field_type == "object"
                and isinstance(parsed_value, dict)
            ):
                return {field_name: parsed_value}
        except json.JSONDecodeError:
            pass

        # Handle comma-separated lists for arrays
        if field_type == "array":
            value = cleaned.strip()
            if "," in value or " and " in value or " or " in value:
                value = value.replace(" and ", ",").replace(" or ", ",")
                items = [item.strip().strip("\"'") for item in value.split(",")]
                items = [item for item in items if item]
                if items:
                    return {field_name: items}

        # Try primitive value wrapping
        value = cleaned.strip()

        # Remove quotes
        if (
            value.startswith('"')
            and value.endswith('"')
            or value.startswith("'")
            and value.endswith("'")
        ):
            value = value[1:-1]

        # Type conversion
        converted = ClaudeCodeModel._convert_primitive_value(value, field_type)
        if converted is not None:
            return {field_name: converted}

        return None

    def _extract_json_robust(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Extract JSON from text using multiple robust strategies.

        Args:
            text: Raw text that may contain JSON
            schema: JSON schema for the expected structure

        Returns:
            Extracted JSON as dict

        Raises:
            json.JSONDecodeError: If JSON cannot be extracted
        """
        # Try each extraction strategy in order
        result = self._try_extract_from_markdown(text)
        if result:
            return result

        result = self._try_extract_json_object(text)
        if result:
            return result

        result = self._try_extract_json_array(text, schema)
        if result:
            return result

        result = self._try_single_field_autowrap(text, schema)
        if result:
            return result

        # If all strategies fail, raise error
        raise json.JSONDecodeError(
            "Could not extract valid JSON from response", text, 0
        )

    def _get_model_name(self, response: ClaudeJSONResponse) -> str:
        """Extract model name from Claude response.

        Args:
            response: Claude CLI JSON response

        Returns:
            Model name string
        """
        model_name = self._model_name
        if "modelUsage" in response and response.get("modelUsage"):
            # Use the actual model name from the response
            model_names = list(response["modelUsage"].keys())
            if model_names:
                model_name = model_names[0]
        return model_name

    @staticmethod
    def _create_usage(response: ClaudeJSONResponse) -> RequestUsage:
        """Create usage info from Claude response.

        Args:
            response: Claude CLI JSON response

        Returns:
            Request usage information
        """
        usage_data = response.get("usage", {})
        server_tool_use = (
            usage_data.get("server_tool_use", {})
            if isinstance(usage_data, dict)
            else {}
        )
        web_search_requests = (
            server_tool_use.get("web_search_requests", 0)
            if isinstance(server_tool_use, dict)
            else 0
        )

        return RequestUsage(
            input_tokens=usage_data.get("input_tokens", 0)
            if isinstance(usage_data, dict)
            else 0,
            cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0)
            if isinstance(usage_data, dict)
            else 0,
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0)
            if isinstance(usage_data, dict)
            else 0,
            output_tokens=usage_data.get("output_tokens", 0)
            if isinstance(usage_data, dict)
            else 0,
            details={
                "web_search_requests": web_search_requests,
                "total_cost_usd_cents": int(
                    response.get("total_cost_usd", 0.0) * 100
                ),  # Store as cents
                "duration_ms": response.get("duration_ms", 0),
                "duration_api_ms": response.get("duration_api_ms", 0),
                "num_turns": response.get("num_turns", 0),
            },
        )
