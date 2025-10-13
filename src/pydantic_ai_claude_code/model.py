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
from .tools import format_tools_for_prompt, parse_tool_calls
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

        instruction = f"""Write your answer to: {output_filename}

Use the Write tool to create the file with your response. Write ONLY your direct answer - no preambles, no explanations, just the answer."""

        return instruction

    def _build_structured_output_instruction(
        self, output_tool: Any, settings: ClaudeCodeSettings
    ) -> str:
        """Build JSON instruction for structured output.

        Args:
            output_tool: The output tool definition
            settings: Settings dict to store output file path

        Returns:
            JSON instruction string to append to system prompt
        """
        schema = output_tool.parameters_json_schema
        properties = schema.get("properties", {})

        # Build concrete example
        example_obj: dict[str, Any] = {}
        for field, props in properties.items():
            field_type = props.get("type", "string")
            if field_type == "integer":
                example_obj[field] = 42
            elif field_type == "number":
                example_obj[field] = 3.14
            elif field_type == "boolean":
                example_obj[field] = True
            elif field_type == "array":
                example_obj[field] = ["item1", "item2"]
            elif field_type == "object":
                example_obj[field] = {"key": "value"}
            else:
                example_obj[field] = "example value"

        # Generate unique filename for structured output
        output_filename = f"/tmp/claude_structured_output_{uuid.uuid4().hex}.json"
        settings["__structured_output_file"] = output_filename

        logger.debug("Structured output file path: %s", output_filename)

        json_instruction = f"""CRITICAL: STRUCTURED OUTPUT MODE

You must create a JSON file at: {output_filename}

The JSON file MUST match this schema EXACTLY:
{json.dumps(schema, indent=2)}

Example valid JSON (use this structure):
{json.dumps(example_obj, indent=2)}

INSTRUCTIONS:
1. Use the Write tool to create the file {output_filename}
2. The file must contain ONLY valid JSON (no explanations, no markdown)
3. The JSON must include ALL required fields: {list(properties.keys())}
4. After creating the file, do NOT output anything else

Create the file now."""

        return json_instruction

    async def request(
        self,
        messages: list[ModelMessage],
        _model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make an async request to Claude Code CLI.

        Args:
            messages: List of messages in the conversation
            model_settings: Optional model settings
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

        # Format messages into a prompt
        prompt = format_messages_for_claude(messages)

        # Get settings from provider
        settings = self.provider.get_settings(model=self._model_name)

        # Check if we need structured output or function tools
        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )

        logger.debug("Formatted prompt length: %d chars", len(prompt))

        # Build system prompt
        system_prompt_parts = []

        if model_request_parameters and hasattr(
            model_request_parameters, "system_prompt"
        ):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        # Check if we have tool results yet
        has_tool_results = any(
            isinstance(part, ToolReturnPart)
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

        # Add function tools prompt ONLY if there are no tool results yet
        # Once we have results, we don't want to confuse Claude with tool definitions
        if function_tools and not has_tool_results:
            # First call: Use structured output to get function call as JSON
            # Build a schema for the function call
            tool_call_schema = {
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "enum": [tool.name for tool in function_tools],
                        "description": "The name of the function to call",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "The arguments to pass to the function",
                    },
                },
                "required": ["function_name", "arguments"],
            }

            # Build description of available functions
            func_descriptions = []
            for tool in function_tools:
                params_desc = json.dumps(tool.parameters_json_schema, indent=2)
                func_descriptions.append(
                    f"- {tool.name}: {tool.description or 'No description'}\n  Parameters: {params_desc}"
                )

            # Generate unique filename for function call output
            function_call_file = f"/tmp/claude_function_call_{uuid.uuid4().hex}.json"
            settings["__function_call_file"] = function_call_file

            function_call_instruction = f"""FUNCTION CALL MODE

Analyze the user's request and determine if you need to call one of these functions:

{chr(10).join(func_descriptions)}

If you need to call a function, use the Write tool to create: {function_call_file}

The JSON file must match this schema:
{json.dumps(tool_call_schema, indent=2)}

Example:
{{"function_name": "process_data", "arguments": {{"name": "Widget", "count": 10}}}}

CRITICAL: Write ONLY valid JSON to the file. After creating the file, do NOT output anything else."""

            system_prompt_parts.append(function_call_instruction)
        # No else needed - after function execution, Claude just composes natural response

        # Add output instructions (only if not in function call mode)
        if output_tools and not function_tools:
            # Structured output: instruct Claude to write JSON to file
            json_instruction = self._build_structured_output_instruction(
                output_tools[0], settings
            )
            system_prompt_parts.append(json_instruction)
        elif not function_tools:
            # Unstructured output: instruct to write to file
            unstructured_instruction = self._build_unstructured_output_instruction(
                settings
            )
            system_prompt_parts.append(unstructured_instruction)

        # Build complete prompt with system instructions
        # Write to prompt.md instead of CLI args to avoid argument list size limit
        if system_prompt_parts:
            combined_system_prompt = "\n\n".join(system_prompt_parts)
            # Prepend system prompt to the conversation prompt
            prompt = f"{combined_system_prompt}\n\n{prompt}"
            logger.debug(
                "Added %d chars of system instructions to prompt",
                len(combined_system_prompt),
            )

        # Also include any user-specified append_system_prompt in the prompt file
        existing_prompt = settings.get("append_system_prompt")
        if existing_prompt:
            prompt = f"{existing_prompt}\n\n{prompt}"
            # Remove from settings so it's not duplicated as CLI arg
            settings.pop("append_system_prompt", None)
            logger.debug(
                "Added %d chars of user system prompt to prompt file",
                len(existing_prompt),
            )

        # Run Claude CLI
        response = await run_claude_async(prompt, settings=settings)

        # Convert to ModelResponse with usage
        return self._convert_response(
            response,
            output_tools=output_tools,
            function_tools=function_tools,
            settings=settings,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        _model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        _run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to Claude Code CLI.

        Args:
            messages: List of messages in the conversation
            model_settings: Optional model settings
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

        # Format messages into a prompt
        prompt = format_messages_for_claude(messages)

        # Get settings from provider
        settings = self.provider.get_settings(model=self._model_name)

        # Check if we need structured output or function tools
        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )

        # Build system prompt (same as request method)
        system_prompt_parts = []

        if model_request_parameters and hasattr(
            model_request_parameters, "system_prompt"
        ):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        # Add function tools prompt ONLY if there are no tool results yet
        if function_tools:
            # Check if any messages contain tool results
            has_tool_results = any(
                isinstance(part, ToolReturnPart)
                for msg in messages
                if isinstance(msg, ModelRequest)
                for part in msg.parts
            )

            if not has_tool_results:
                # First call: show tool definitions so Claude can decide to use them
                tools_prompt = format_tools_for_prompt(function_tools)
                system_prompt_parts.append(tools_prompt)
            # Else: tool already called, don't show definitions, let Claude compose response

        # Add output instructions
        if output_tools:
            # Structured output: instruct Claude to write JSON to file
            json_instruction = self._build_structured_output_instruction(
                output_tools[0], settings
            )
            system_prompt_parts.append(json_instruction)
        elif not function_tools:
            # Unstructured output: instruct to write to file
            unstructured_instruction = self._build_unstructured_output_instruction(
                settings
            )
            system_prompt_parts.append(unstructured_instruction)

        # Build complete prompt with system instructions
        # Write to prompt.md instead of CLI args to avoid argument list size limit
        if system_prompt_parts:
            combined_system_prompt = "\n\n".join(system_prompt_parts)
            # Prepend system prompt to the conversation prompt
            prompt = f"{combined_system_prompt}\n\n{prompt}"
            logger.debug(
                "Added %d chars of system instructions to prompt",
                len(combined_system_prompt),
            )

        # Also include any user-specified append_system_prompt in the prompt file
        existing_prompt = settings.get("append_system_prompt")
        if existing_prompt:
            prompt = f"{existing_prompt}\n\n{prompt}"
            # Remove from settings so it's not duplicated as CLI arg
            settings.pop("append_system_prompt", None)
            logger.debug(
                "Added %d chars of user system prompt to prompt file",
                len(existing_prompt),
            )

        # Get working directory
        import tempfile

        cwd = settings.get("working_directory")

        # If no working directory, create a temp one
        if not cwd:
            cwd = tempfile.mkdtemp(prefix="claude_prompt_")

        # Ensure working directory exists
        Path(cwd).mkdir(parents=True, exist_ok=True)

        # Write prompt to prompt.md in the working directory
        prompt_file = Path(cwd) / "prompt.md"
        prompt_file.write_text(prompt)

        # Build command for streaming
        cmd = build_claude_command(settings=settings, output_format="stream-json")

        # Create event stream
        event_stream = run_claude_streaming(cmd, cwd=cwd)

        # Determine model name
        model_name = self._model_name

        # Create and yield streaming response
        streamed_response = ClaudeCodeStreamedResponse(
            model_request_parameters=model_request_parameters,
            model_name=f"claude-code:{model_name}",
            event_stream=event_stream,
            timestamp=datetime.now(timezone.utc),
        )

        yield streamed_response

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

        # Create response parts
        parts: list[Any] = []

        # First, check for function tool calls (takes precedence)
        if function_tools:
            # Check if we have a function call file
            function_call_file_obj = (
                settings.get("__function_call_file") if settings else None
            )
            function_call_file = (
                str(function_call_file_obj) if function_call_file_obj else None
            )

            if function_call_file and Path(function_call_file).exists():
                # Read function call from file
                try:
                    logger.debug(
                        "Reading function call from file: %s", function_call_file
                    )
                    with open(function_call_file) as f:
                        function_call_data = json.load(f)

                    # Cleanup temp file
                    self._cleanup_temp_file(function_call_file)

                    # Extract function name and arguments
                    function_name = function_call_data.get("function_name")
                    arguments = function_call_data.get("arguments", {})

                    if function_name:
                        logger.debug(
                            "Parsed function call: %s with %d args",
                            function_name,
                            len(arguments),
                        )
                        tool_call = ToolCallPart(
                            tool_name=function_name,
                            args=arguments,
                            tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                        )
                        parts.append(tool_call)

                        # Create usage info and return early
                        usage = self._create_usage(response)
                        model_name = self._get_model_name(response)
                        return ModelResponse(
                            parts=[tool_call],
                            model_name=model_name,
                            timestamp=datetime.now(timezone.utc),
                            usage=usage,
                        )
                except Exception as e:
                    logger.error("Failed to read function call file: %s", e)
                    # Fall through to text parsing fallback
            else:
                logger.debug("No function call file found, checking response text")

            # Fallback: try parsing from response text (legacy EXECUTE format)
            tool_calls = parse_tool_calls(result_text)
            if tool_calls:
                logger.debug("Parsed %d tool calls from response text", len(tool_calls))
                parts.extend(tool_calls)
                usage = self._create_usage(response)
                model_name = self._get_model_name(response)
                return ModelResponse(
                    parts=parts,
                    model_name=model_name,
                    timestamp=datetime.now(timezone.utc),
                    usage=usage,
                )
            else:
                logger.debug(
                    "No tool calls found despite function_tools being provided"
                )

        # Check if we need to return structured output via tool call
        if output_tools and len(output_tools) > 0:
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
                        structured_file, schema
                    )

                    if error_msg:
                        # Return error as text so Pydantic AI can retry
                        parts.append(TextPart(content=error_msg))
                        model_name = self._get_model_name(response)
                        usage = self._create_usage(response)
                        return ModelResponse(
                            parts=parts,
                            model_name=model_name,
                            timestamp=datetime.now(timezone.utc),
                            usage=usage,
                        )

                    if parsed_data:
                        # Validation passed, create tool call
                        logger.debug("Successfully created structured output from file")
                        parts.append(
                            ToolCallPart(
                                tool_name=tool_name,
                                args=parsed_data,
                                tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                            )
                        )
                    else:
                        # No file found, use fallback extraction
                        logger.warning(
                            "Structured output file not found, using fallback JSON extraction"
                        )
                        parsed_data = self._extract_json_robust(result_text, schema)
                        parts.append(
                            ToolCallPart(
                                tool_name=tool_name,
                                args=parsed_data,
                                tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                            )
                        )
                else:
                    # Fallback: Use robust extraction with multiple strategies
                    logger.debug(
                        "No structured output file configured, using robust JSON extraction"
                    )
                    parsed_data = self._extract_json_robust(result_text, schema)
                    parts.append(
                        ToolCallPart(
                            tool_name=tool_name,
                            args=parsed_data,
                            tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                        )
                    )
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return as text
                # Pydantic AI will retry with validation error
                logger.error("Failed to parse structured output JSON: %s", e)
                parts.append(TextPart(content=result_text))
        else:
            # Unstructured text response - read from file if Claude created it
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
                    # Cleanup temp file
                    self._cleanup_temp_file(unstructured_file)
                    logger.debug(
                        "Successfully read %d bytes from unstructured output file",
                        len(file_content),
                    )
                    parts.append(TextPart(content=file_content))
                except Exception as e:
                    # Fallback to CLI response if file read fails
                    logger.warning(
                        "Failed to read unstructured output file, using CLI response: %s",
                        e,
                    )
                    parts.append(TextPart(content=result_text))
            else:
                # Fallback to CLI response if no file
                if unstructured_file:
                    logger.warning(
                        "Unstructured output file not found: %s", unstructured_file
                    )
                logger.debug("Using CLI response text for unstructured output")
                parts.append(TextPart(content=result_text))

        # Determine model name and create usage
        model_name = self._get_model_name(response)
        usage = self._create_usage(response)

        return ModelResponse(
            parts=parts,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            usage=usage,
        )

    def _cleanup_temp_file(self, file_path: str | Path) -> None:
        """Safely remove temporary file.

        Args:
            file_path: Path to file to remove
        """
        with contextlib.suppress(Exception):
            Path(file_path).unlink()

    def _validate_json_schema(self, data: dict, schema: dict) -> str | None:
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
            return f"Missing required fields: {missing_fields}\nReceived data: {json.dumps(data)}"

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
                    return f"Field '{field_name}' has wrong type. Expected {expected_type}, got {type(actual_value).__name__}\nReceived data: {json.dumps(data)}"

        return None

    def _read_structured_output_file(
        self, file_path: str, schema: dict
    ) -> tuple[dict | None, str | None]:
        """Read and validate structured output file.

        Args:
            file_path: Path to structured output file
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
            self._cleanup_temp_file(file_path)
            return None, f"Failed to read file: {e}"

        # Parse JSON
        try:
            parsed_data = json.loads(file_content)
            logger.debug("Successfully parsed JSON from structured output file")
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in structured output file: %s", e)
            self._cleanup_temp_file(file_path)
            return None, f"Invalid JSON in file: {e}\nFile content:\n{file_content}"

        # Validate schema
        validation_error = self._validate_json_schema(parsed_data, schema)
        if validation_error:
            logger.error("Schema validation failed: %s", validation_error)
            self._cleanup_temp_file(file_path)
            return None, validation_error

        # Validation passed - clean up file
        logger.debug("Structured output validated successfully, cleaning up file")
        self._cleanup_temp_file(file_path)
        return parsed_data, None

    def _extract_json_robust(self, text: str, schema: dict) -> dict:
        """Extract JSON from text using multiple robust strategies.

        Args:
            text: Raw text that may contain JSON
            schema: JSON schema for the expected structure

        Returns:
            Extracted JSON as dict

        Raises:
            json.JSONDecodeError: If JSON cannot be extracted
        """
        import json
        import re

        # Strategy 1: Strip markdown code blocks and parse
        cleaned = text.strip()

        # Remove markdown code blocks (```json or ```)
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # Try parsing cleaned text
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON using regex
        # Match { ... } objects (handles nested braces)
        json_pattern = r"\{(?:[^{}]|\{[^{}]*\})*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # Strategy 3: Extract JSON array using regex
        array_pattern = r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]"
        array_matches = re.findall(array_pattern, text, re.DOTALL)

        for match in array_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    # If single-field schema expects array, wrap it
                    properties = schema.get("properties", {})
                    if len(properties) == 1:
                        field_name = list(properties.keys())[0]
                        return {field_name: parsed}
                    # For multi-field schemas, can't use bare array
                    continue
            except json.JSONDecodeError:
                continue

        # Strategy 4: For single-field schemas, try to auto-wrap values
        properties = schema.get("properties", {})
        if len(properties) == 1:
            field_name = list(properties.keys())[0]
            field_type = properties[field_name].get("type")

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
                # Try parsing as comma-separated list
                if "," in value or " and " in value or " or " in value:
                    # Replace common separators
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
            try:
                if field_type == "integer":
                    return {field_name: int(value)}
                elif field_type == "number":
                    return {field_name: float(value)}
                elif field_type == "boolean":
                    bool_val = value.lower() in ("true", "1", "yes")
                    return {field_name: bool_val}
                elif field_type == "string":
                    return {field_name: value}
            except (ValueError, AttributeError):
                pass

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

    def _create_usage(self, response: ClaudeJSONResponse) -> RequestUsage:
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
