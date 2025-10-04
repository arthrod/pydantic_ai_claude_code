"""Claude Code model implementation for Pydantic AI."""

from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
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

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return f"claude-code:{self._model_name}"

    @property
    def system(self) -> str:
        """Get the system identifier."""
        return "claude-code"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
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
        # Format messages into a prompt
        prompt = format_messages_for_claude(messages)

        # Get settings from provider
        settings = self.provider.get_settings(model=self._model_name)

        # Check if we need structured output or function tools
        output_tools = model_request_parameters.output_tools if model_request_parameters else []
        function_tools = model_request_parameters.function_tools if model_request_parameters else []

        # Build system prompt
        system_prompt_parts = []

        if model_request_parameters and hasattr(model_request_parameters, "system_prompt"):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        # Add function tools prompt if present
        if function_tools:
            tools_prompt = format_tools_for_prompt(function_tools)
            system_prompt_parts.append(tools_prompt)

        # If there are output tools (structured output), instruct Claude to return JSON
        if output_tools:
            output_tool = output_tools[0]  # Get the first (and usually only) output tool
            schema = output_tool.parameters_json_schema
            import json

            # Build concrete example
            properties = schema.get('properties', {})
            example_obj = {}
            for field, props in properties.items():
                field_type = props.get('type', 'string')
                if field_type == 'integer':
                    example_obj[field] = 42
                elif field_type == 'number':
                    example_obj[field] = 3.14
                elif field_type == 'boolean':
                    example_obj[field] = True
                elif field_type == 'array':
                    example_obj[field] = ["item1", "item2"]
                elif field_type == 'object':
                    example_obj[field] = {"key": "value"}
                else:
                    example_obj[field] = "example value"

            json_instruction = f"""CRITICAL: You MUST respond with ONLY a JSON object matching this schema.

Schema:
{json.dumps(schema, indent=2)}

Your response MUST be a JSON OBJECT with these fields: {list(properties.keys())}

Example response format (use this EXACT structure):
{json.dumps(example_obj, indent=2)}

RULES:
- Your ENTIRE response must be ONLY the JSON object
- Start with {{ and end with }}
- Include ALL fields from the schema
- Do NOT just return a single value like "7" or "hello"
- Do NOT add explanations, markdown, or extra text
- The response must be a complete JSON object, not a primitive value"""

            system_prompt_parts.append(json_instruction)

        if system_prompt_parts:
            combined_prompt = "\n\n".join(system_prompt_parts)
            existing_prompt = settings.get("append_system_prompt")
            if existing_prompt:
                settings["append_system_prompt"] = f"{existing_prompt}\n\n{combined_prompt}"
            else:
                settings["append_system_prompt"] = combined_prompt

        # Run Claude CLI
        response = await run_claude_async(prompt, settings=settings)

        # Convert to ModelResponse with usage
        return self._convert_response(response, output_tools=output_tools, function_tools=function_tools)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
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
        # Format messages into a prompt
        prompt = format_messages_for_claude(messages)

        # Get settings from provider
        settings = self.provider.get_settings(model=self._model_name)

        # Check if we need structured output or function tools
        output_tools = model_request_parameters.output_tools if model_request_parameters else []
        function_tools = model_request_parameters.function_tools if model_request_parameters else []

        # Build system prompt (same as request method)
        system_prompt_parts = []

        if model_request_parameters and hasattr(model_request_parameters, "system_prompt"):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        if function_tools:
            tools_prompt = format_tools_for_prompt(function_tools)
            system_prompt_parts.append(tools_prompt)

        if output_tools:
            output_tool = output_tools[0]
            schema = output_tool.parameters_json_schema
            import json

            # Build concrete example
            properties = schema.get('properties', {})
            example_obj = {}
            for field, props in properties.items():
                field_type = props.get('type', 'string')
                if field_type == 'integer':
                    example_obj[field] = 42
                elif field_type == 'number':
                    example_obj[field] = 3.14
                elif field_type == 'boolean':
                    example_obj[field] = True
                elif field_type == 'array':
                    example_obj[field] = ["item1", "item2"]
                elif field_type == 'object':
                    example_obj[field] = {"key": "value"}
                else:
                    example_obj[field] = "example value"

            json_instruction = f"""CRITICAL: You MUST respond with ONLY a JSON object matching this schema.

Schema:
{json.dumps(schema, indent=2)}

Your response MUST be a JSON OBJECT with these fields: {list(properties.keys())}

Example response format (use this EXACT structure):
{json.dumps(example_obj, indent=2)}

RULES:
- Your ENTIRE response must be ONLY the JSON object
- Start with {{ and end with }}
- Include ALL fields from the schema
- Do NOT just return a single value like "7" or "hello"
- Do NOT add explanations, markdown, or extra text
- The response must be a complete JSON object, not a primitive value"""
            system_prompt_parts.append(json_instruction)

        if system_prompt_parts:
            combined_prompt = "\n\n".join(system_prompt_parts)
            existing_prompt = settings.get("append_system_prompt")
            if existing_prompt:
                settings["append_system_prompt"] = f"{existing_prompt}\n\n{combined_prompt}"
            else:
                settings["append_system_prompt"] = combined_prompt

        # Build command for streaming
        cmd = build_claude_command(prompt, settings=settings, output_format="stream-json")
        cwd = settings.get("working_directory")

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
    ) -> ModelResponse:
        """Convert Claude JSON response to ModelResponse.

        Args:
            response: Claude CLI JSON response
            output_tools: Optional output tool definitions for structured output
            function_tools: Optional function tool definitions for tool calling

        Returns:
            Pydantic AI ModelResponse with embedded usage
        """
        import json
        import uuid

        # Extract result text
        result_text = response.get("result", "")

        # Create response parts
        parts: list[Any] = []

        # First, check for function tool calls (takes precedence)
        if function_tools:
            tool_calls = parse_tool_calls(result_text)
            if tool_calls:
                parts.extend(tool_calls)
                # Create usage info and return early
                usage = self._create_usage(response)
                model_name = self._get_model_name(response)
                return ModelResponse(
                    parts=parts,
                    model_name=model_name,
                    timestamp=datetime.now(timezone.utc),
                    usage=usage,
                )

        # Check if we need to return structured output via tool call
        if output_tools and len(output_tools) > 0:
            output_tool = output_tools[0]
            tool_name = output_tool.name

            try:
                # Try to parse the response as JSON
                # Remove markdown code blocks if present
                cleaned_text = result_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                # Parse JSON
                parsed_data = json.loads(cleaned_text)

                # Create a tool call with the parsed data
                parts.append(
                    ToolCallPart(
                        tool_name=tool_name,
                        args=parsed_data,
                        tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
                    )
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, return as text
                # Pydantic AI will retry with validation error
                parts.append(TextPart(content=result_text))
        else:
            # Regular text response
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

        return RequestUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            details={
                "web_search_requests": usage_data.get("server_tool_use", {}).get(
                    "web_search_requests", 0
                ),
                "total_cost_usd_cents": int(response.get("total_cost_usd", 0.0) * 100),  # Store as cents
                "duration_ms": response.get("duration_ms", 0),
                "duration_api_ms": response.get("duration_api_ms", 0),
                "num_turns": response.get("num_turns", 0),
            },
        )
