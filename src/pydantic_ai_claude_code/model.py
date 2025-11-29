"""Claude Code model implementation for Pydantic AI.

This model wraps the local Claude CLI to provide a Pydantic AI compatible
interface with support for:
- Structured responses via output validation
- Tool/function calling
- Multi-turn conversations
- Streaming responses
- Provider presets for third-party providers
"""

from __future__ import annotations

import json
import logging
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

from ._sdk.types import ClaudeAgentOptions, HookConfig
from .messages import format_messages_for_claude
from .provider_presets import (
    ProviderPreset,
    compute_provider_environment,
    get_preset,
)
from .response_utils import (
    create_tool_call_part,
    get_working_directory,
)
from .streamed_response import ClaudeCodeStreamedResponse
from .streaming import run_claude_streaming
from .structure_converter import (
    build_structure_instructions,
    read_structure_from_filesystem,
)
from .temp_path_utils import generate_output_file_path, generate_temp_directory_path
from .types import ClaudeCodeSettings, ClaudeJSONResponse
from .utils import (
    _determine_working_directory,
    _setup_working_directory_and_prompt,
    build_claude_command,
    convert_primitive_value,
    run_claude_async,
    strip_markdown_code_fence,
)

logger = logging.getLogger(__name__)


class ClaudeCodeModel(Model):
    """Pydantic AI model implementation using Claude Code CLI.

    This model wraps the local Claude CLI to provide a Pydantic AI compatible
    interface, supporting all features available through the CLI.

    Settings are merged in order:
    1. Provider preset defaults
    2. Agent config (from model_request_parameters)
    3. Run-time overrides (from model_settings)

    Examples:
        >>> from pydantic_ai import Agent
        >>> agent = Agent(model='claude-code:sonnet')
        >>> result = await agent.run('Hello!')

        # With provider preset
        >>> agent = Agent(model='claude-code:deepseek:sonnet')

        # With hooks at run-time
        >>> result = await agent.run(
        ...     'Hello!',
        ...     model_settings={'hooks': [{'matcher': {'event': 'tool_use'}, 'commands': ['echo $TOOL_NAME']}]}
        ... )
    """

    def __init__(
        self,
        model_name: str = "sonnet",
        *,
        provider_preset: str | None = None,
        cli_path: str | None = None,
    ):
        """Initialize Claude Code model.

        Args:
            model_name: Name of the Claude model to use (e.g., "sonnet", "opus",
                or full model name like "claude-sonnet-4-5-20250929")
            provider_preset: Optional preset ID (deepseek, kimi, etc.)
            cli_path: Optional path to Claude CLI binary
        """
        self._model_alias = model_name
        self._provider_preset_id = provider_preset
        self._cli_path = cli_path
        self._actual_model_name: str = model_name
        self._preset_env_vars: dict[str, str] = {}
        self._provider_preset: ProviderPreset | None = None

        # Load and apply provider preset if specified
        if provider_preset:
            self._provider_preset = get_preset(provider_preset)
            if self._provider_preset:
                # Compute environment variables (NOT applied globally)
                self._preset_env_vars = compute_provider_environment(
                    self._provider_preset,
                    override_existing=False,
                )
                # Get actual model name from preset
                self._actual_model_name = self._provider_preset.get_model_name(model_name)
                logger.info(
                    "Loaded provider preset '%s' with %d environment variables, model: %s",
                    provider_preset,
                    len(self._preset_env_vars),
                    self._actual_model_name,
                )
            else:
                logger.warning(
                    "Provider preset '%s' not found. "
                    "Available presets can be listed with list_presets()",
                    provider_preset,
                )

        logger.debug(
            "Initialized ClaudeCodeModel: model_alias=%s, actual_model=%s, preset=%s",
            model_name,
            self._actual_model_name,
            provider_preset,
        )

    @property
    def model_name(self) -> str:
        """Get the full model identifier."""
        if self._provider_preset_id:
            return f"claude-code:{self._provider_preset_id}:{self._model_alias}"
        return f"claude-code:{self._model_alias}"

    @property
    def system(self) -> str:
        """Get the system identifier."""
        return "claude-code"

    def _build_options(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ClaudeCodeSettings:
        """Build execution options by merging all settings sources.

        Merge order (later overrides earlier):
        1. Provider preset defaults
        2. Agent config (from model_request_parameters)
        3. Run-time overrides (from model_settings)

        Args:
            model_settings: Run-time settings from agent.run()
            model_request_parameters: Agent-level config (tools, output schema)

        Returns:
            Merged settings dictionary
        """
        settings: dict[str, Any] = {}

        # 1. Base settings from constructor
        settings["model"] = self._actual_model_name
        if self._cli_path:
            settings["claude_cli_path"] = self._cli_path

        # Provider preset env vars (passed to subprocess)
        if self._preset_env_vars:
            settings["__provider_env"] = self._preset_env_vars.copy()

        # Always bypass permissions (full access)
        settings["dangerously_skip_permissions"] = True

        # Default settings
        settings["use_temp_workspace"] = True
        settings["retry_on_rate_limit"] = True
        settings["timeout_seconds"] = 900
        settings["use_sandbox_runtime"] = True

        # 2. Agent-level config (from model_request_parameters)
        if model_request_parameters.function_tools:
            # Register function tools as allowed
            settings["allowed_tools"] = [
                tool.name for tool in model_request_parameters.function_tools
            ]

        # 3. Run-time overrides (from model_settings)
        if model_settings:
            # Map pydantic_ai model_settings to our settings format
            if "working_directory" in model_settings:
                settings["working_directory"] = model_settings["working_directory"]
            if "timeout_seconds" in model_settings:
                settings["timeout_seconds"] = model_settings["timeout_seconds"]
            if "additional_files" in model_settings:
                settings["additional_files"] = model_settings["additional_files"]
            if "debug_save_prompts" in model_settings:
                settings["debug_save_prompts"] = model_settings["debug_save_prompts"]
            if "append_system_prompt" in model_settings:
                settings["append_system_prompt"] = model_settings["append_system_prompt"]
            if "verbose" in model_settings:
                settings["verbose"] = model_settings["verbose"]

            # Handle hooks from model_settings
            if "hooks" in model_settings:
                hooks_config = model_settings["hooks"]
                if isinstance(hooks_config, list):
                    settings["__hooks__"] = hooks_config

            # Handle extra CLI args
            if "extra_cli_args" in model_settings:
                settings["extra_cli_args"] = model_settings["extra_cli_args"]

        return settings  # type: ignore[return-value]

    def _check_has_tool_results(self, messages: list[ModelMessage]) -> bool:
        """Check if messages contain tool results."""
        return any(
            isinstance(part, ToolReturnPart)
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

    @staticmethod
    def _xml_to_markdown(xml_text: str) -> str:
        """Convert XML-tagged description to markdown format."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(xml_text, "html.parser")

        summary = soup.find("summary")
        returns = soup.find("returns")

        parts = []
        if summary:
            summary_text = summary.get_text(strip=True)
            if summary_text and not summary_text.endswith("."):
                summary_text += "."
            parts.append(summary_text)
        if returns:
            desc = returns.find("description")
            if desc:
                parts.append(f"Returns: {desc.get_text(strip=True)}")

        return " ".join(parts) if parts else xml_text

    def _build_unstructured_output_instruction(
        self, settings: ClaudeCodeSettings
    ) -> str:
        """Build instruction for unstructured (text) output via file."""
        working_dir = get_working_directory(settings)
        output_filename = generate_output_file_path(
            working_dir, "claude_unstructured_output", ".txt"
        )
        settings["__unstructured_output_file"] = output_filename

        return f"""# Output Instructions

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

> **IMPORTANT:** All output must go to the file path specified above.

---

## The User's Request

**Use the Read tool to read the file `user_request.md`. This contains the user's request. Write your response to that request to the output file specified above.**

"""

    def _build_structured_output_instruction(
        self, output_tool: Any, settings: ClaudeCodeSettings
    ) -> str:
        """Build instruction for structured output using improved converter."""
        schema = output_tool.parameters_json_schema

        working_dir = get_working_directory(settings)
        output_filename = generate_output_file_path(
            working_dir, "claude_structured_output", ".json"
        )
        settings["__structured_output_file"] = output_filename

        temp_data_dir = generate_temp_directory_path(
            working_dir, "claude_data_structure", short_id=True
        )
        settings["__temp_json_dir"] = temp_data_dir

        return build_structure_instructions(schema, temp_data_dir)

    def _build_function_option_descriptions(
        self, function_tools: list[Any]
    ) -> list[str]:
        """Build option descriptions for function selection prompt."""
        option_descriptions = []
        for i, tool in enumerate(function_tools, 1):
            desc = tool.description or "No description"
            desc_clean = self._xml_to_markdown(desc)

            schema = tool.parameters_json_schema
            params = schema.get("properties", {})
            param_hints = [
                f"{param_name}: {param_schema.get('type', 'unknown')}"
                for param_name, param_schema in params.items()
            ]
            params_str = (
                f" (parameters: {', '.join(param_hints)})" if param_hints else ""
            )
            option_descriptions.append(f"{i}. {tool.name}{params_str} - {desc_clean}")

        option_descriptions.append(
            f"{len(function_tools) + 1}. none - Answer directly without calling any function"
        )
        return option_descriptions

    def _build_function_tools_prompt(
        self, function_tools: list[Any]
    ) -> tuple[str, dict[str, Any]]:
        """Build function selection prompt and available functions dict."""
        option_descriptions = self._build_function_option_descriptions(function_tools)

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
> - You are ONLY making a selection

---

## The User's Request

**Use the Read tool to read the file `user_request.md`. This contains the user's request. Analyze it to determine which function(s) to call.**

"""

        available_functions = {tool.name: tool for tool in function_tools}
        return prompt, available_functions

    def _build_system_prompt_parts(
        self,
        model_request_parameters: ModelRequestParameters,
        has_tool_results: bool,
        settings: ClaudeCodeSettings,
        is_streaming: bool = False,
    ) -> list[str]:
        """Build system prompt parts for request."""
        system_prompt_parts = []
        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )

        # Add tool synthesis instruction when tool results are present
        if has_tool_results:
            tool_synthesis_instruction = """# Task: Synthesize Tool Results

**Tool result files are provided below in the conversation. Read these files and use the information to answer the user's request naturally.**

---
"""
            system_prompt_parts.append(tool_synthesis_instruction)

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

        # Add output instructions
        if not is_streaming:
            should_add_output_instructions = False

            if has_tool_results:
                should_add_output_instructions = True
            elif not function_tools:
                should_add_output_instructions = True

            if should_add_output_instructions:
                if output_tools:
                    json_instruction = self._build_structured_output_instruction(
                        output_tools[0], settings
                    )
                    system_prompt_parts.append(json_instruction)
                else:
                    unstructured_instruction = (
                        self._build_unstructured_output_instruction(settings)
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
        """Assemble final prompt with system instructions."""
        working_dir = settings.get("__working_directory", "/tmp")

        # Format user messages and write to file
        formatted_messages = format_messages_for_claude(
            messages, skip_system_prompt=has_tool_results, working_dir=working_dir
        )

        user_request_path = Path(working_dir) / "user_request.md"
        user_request_path.parent.mkdir(parents=True, exist_ok=True)
        with open(user_request_path, "w", encoding="utf-8") as f:
            f.write(formatted_messages)

        # Build prompt with system instructions only
        prompt = ""

        existing_prompt = settings.get("append_system_prompt")
        if existing_prompt:
            prompt = f"{existing_prompt}\n\n"
            settings.pop("append_system_prompt", None)

        if system_prompt_parts:
            combined_system_prompt = "\n\n".join(system_prompt_parts)
            prompt = f"{prompt}{combined_system_prompt}"

        return prompt

    def _create_model_response_with_usage(
        self,
        response: ClaudeJSONResponse,
        parts: list[TextPart | ToolCallPart],
    ) -> ModelResponse:
        """Create ModelResponse with usage from Claude response."""
        usage = self._create_usage(response)
        model_name = self._get_model_name(response)
        return ModelResponse(
            parts=parts,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            usage=usage,
        )

    def _prepare_working_directory(self, settings: ClaudeCodeSettings) -> None:
        """Determine and set working directory early."""
        if "__working_directory" not in settings:
            working_dir = _determine_working_directory(settings)
            settings["__working_directory"] = working_dir

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make an async request to Claude Code CLI.

        Args:
            messages: List of messages in the conversation
            model_settings: Optional model settings from Agent (can include timeout_seconds, hooks, etc.)
            model_request_parameters: Model request parameters

        Returns:
            Model response with embedded usage information
        """
        logger.info(
            "Starting non-streaming request with %d messages",
            len(messages),
        )

        # Build options by merging all settings sources
        settings = self._build_options(model_settings, model_request_parameters)

        # Determine working directory early
        self._prepare_working_directory(settings)

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

        # Assemble final prompt
        prompt = self._assemble_final_prompt(
            messages, system_prompt_parts, settings, has_tool_results
        )

        # Run Claude CLI and convert response
        response = await run_claude_async(prompt, settings=settings)
        result = self._convert_response(
            response,
            output_tools=output_tools,
            function_tools=function_tools,
            settings=settings,
        )

        # Handle function selection follow-ups if needed
        return await self._handle_function_selection_followup(
            messages,
            model_request_parameters,
            settings,
            response,
            result,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to Claude Code CLI."""
        logger.info(
            "Starting streaming request with %d messages",
            len(messages),
        )

        # Build options by merging all settings sources
        settings = self._build_options(model_settings, model_request_parameters)

        # Determine working directory early
        self._prepare_working_directory(settings)

        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )

        # Streaming is only supported for plain text responses
        if output_tools:
            raise ValueError(
                "Streaming is not supported with structured output (output_tools)."
            )
        if function_tools:
            raise ValueError(
                "Streaming is not supported with function tools."
            )

        # Check if we have tool results
        has_tool_results = self._check_has_tool_results(messages)

        # Build system prompt
        system_prompt_parts = self._build_system_prompt_parts(
            model_request_parameters,
            has_tool_results,
            settings,
            is_streaming=True,
        )

        # Assemble final prompt
        prompt = self._assemble_final_prompt(
            messages, system_prompt_parts, settings, has_tool_results
        )

        # Add streaming marker
        streaming_marker = "<<<STREAM_START>>>"
        prompt = f"""Begin with the marker "{streaming_marker}" on its own line, then provide your response.

Use the Read tool to read `user_request.md` for the user's request.

{prompt}"""
        settings["__streaming_marker__"] = streaming_marker  # type: ignore[typeddict-unknown-key]

        # Setup working directory
        cwd = _setup_working_directory_and_prompt(prompt, settings)

        # Build command and create event stream
        cmd = build_claude_command(settings=settings, output_format="stream-json")
        event_stream = run_claude_streaming(cmd, cwd=cwd)

        # Create and yield streaming response
        streamed_response = ClaudeCodeStreamedResponse(
            model_request_parameters=model_request_parameters,
            model_name=f"claude-code:{self._model_alias}",
            event_stream=event_stream,
            timestamp=datetime.now(timezone.utc),
            streaming_marker=streaming_marker,
        )

        yield streamed_response

    # ===== Response Handling Methods =====

    def _handle_function_selection_response(
        self,
        result_text: str,
        response: ClaudeJSONResponse,
        settings: ClaudeCodeSettings,
    ) -> ModelResponse:
        """Handle function selection mode response parsing."""
        import re

        available_functions = settings.get("__available_functions__", {})
        if isinstance(available_functions, dict):
            valid_options = [name.lower() for name in available_functions] + ["none"]
        else:
            valid_options = ["none"]

        # Extract CHOICE lines
        choice_matches = re.findall(
            r"CHOICE:\s*[\*_]*(\w+)[\*_]*", result_text, re.IGNORECASE
        )

        matched_option = None
        if choice_matches:
            valid_choices = [
                c.strip().lower()
                for c in choice_matches
                if c.strip().lower() in valid_options
            ]
            if valid_choices:
                matched_option = valid_choices[0]

        if matched_option:
            parts: list[TextPart | ToolCallPart] = []
            if matched_option == "none":
                settings["__function_selection_result__"] = "none"
                parts.append(
                    TextPart(content="[Function selection: none - answering directly]")
                )
            else:
                settings["__selected_function__"] = matched_option
                settings["__function_selection_result__"] = "selected"
                parts.append(
                    TextPart(
                        content=f"[Function selected: {matched_option} - collecting arguments]"
                    )
                )

            return self._create_model_response_with_usage(response, parts)

        # Could not parse
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
        """Handle structured output response via tool call."""
        output_tool = output_tools[0]
        tool_name = output_tool.name
        schema = output_tool.parameters_json_schema

        try:
            structured_file = (
                settings.get("__structured_output_file") if settings else None
            )

            if structured_file:
                parsed_data, error_msg = self._read_structured_output_file(
                    structured_file, schema, settings
                )

                if error_msg:
                    return self._create_model_response_with_usage(
                        response, [TextPart(content=error_msg)]
                    )

                if parsed_data:
                    tool_call = create_tool_call_part(
                        tool_name=tool_name,
                        args=parsed_data,
                    )
                    return self._create_model_response_with_usage(response, [tool_call])

                # No file found, use fallback
                parsed_data = self._extract_json_robust(result_text, schema)
                tool_call = create_tool_call_part(
                    tool_name=tool_name,
                    args=parsed_data,
                )
                return self._create_model_response_with_usage(response, [tool_call])

            # Fallback
            parsed_data = self._extract_json_robust(result_text, schema)
            tool_call = create_tool_call_part(
                tool_name=tool_name,
                args=parsed_data,
            )
            return self._create_model_response_with_usage(response, [tool_call])

        except json.JSONDecodeError:
            return self._create_model_response_with_usage(
                response, [TextPart(content=result_text)]
            )

    def _handle_unstructured_output_response(
        self,
        result_text: str,
        response: ClaudeJSONResponse,
        settings: ClaudeCodeSettings | None,
    ) -> ModelResponse:
        """Handle unstructured text output response."""
        unstructured_file_obj = (
            settings.get("__unstructured_output_file") if settings else None
        )
        unstructured_file = (
            str(unstructured_file_obj) if unstructured_file_obj else None
        )

        if unstructured_file and Path(unstructured_file).exists():
            try:
                with open(unstructured_file, encoding="utf-8") as f:
                    file_content = f.read()
                return self._create_model_response_with_usage(
                    response, [TextPart(content=file_content)]
                )
            except Exception:
                return self._create_model_response_with_usage(
                    response, [TextPart(content=result_text)]
                )

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
        """Convert Claude JSON response to ModelResponse."""
        result_text = response.get("result", "")

        if function_tools and settings and settings.get("__function_selection_mode__"):
            return self._handle_function_selection_response(
                result_text, response, settings
            )

        if output_tools and len(output_tools) > 0:
            return self._handle_structured_output_response(
                result_text, response, output_tools, settings
            )

        return self._handle_unstructured_output_response(
            result_text, response, settings
        )

    # ===== Function Selection Follow-up Methods =====

    async def _handle_function_selection_followup(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        settings: ClaudeCodeSettings,
        response: ClaudeJSONResponse,
        result: ModelResponse,
    ) -> ModelResponse:
        """Handle function selection follow-up routing."""
        function_tools = (
            model_request_parameters.function_tools if model_request_parameters else []
        )
        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )

        if not (function_tools and settings.get("__function_selection_mode__")):
            return result

        selection_result = settings.get("__function_selection_result__")

        if selection_result == "none":
            if output_tools:
                return await self._handle_structured_follow_up(
                    messages, model_request_parameters, settings
                )
            else:
                return await self._handle_unstructured_follow_up(
                    messages, model_request_parameters, settings
                )

        elif selection_result == "selected":
            selected_function = settings.get("__selected_function__")
            if selected_function:
                available_functions = settings.get("__available_functions__", {})
                if isinstance(available_functions, dict):
                    return await self._handle_argument_collection(
                        messages,
                        selected_function,
                        available_functions,
                        response,
                        settings,
                    )

        return result

    async def _handle_structured_follow_up(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        original_settings: ClaudeCodeSettings | None = None,
    ) -> ModelResponse:
        """Handle structured follow-up request when function selection is 'none'."""
        settings = self._build_options(None, model_request_parameters)

        # Preserve user-provided settings
        if original_settings:
            for key in ["additional_files", "timeout_seconds", "debug_save_prompts"]:
                if key in original_settings:
                    settings[key] = original_settings[key]

        settings["__function_selection_mode__"] = False
        self._prepare_working_directory(settings)

        output_tools = (
            model_request_parameters.output_tools if model_request_parameters else []
        )

        system_prompt_parts = []
        if model_request_parameters and hasattr(model_request_parameters, "system_prompt"):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        if output_tools:
            structured_instruction = self._build_structured_output_instruction(
                output_tools[0], settings
            )
            system_prompt_parts.append(structured_instruction)

        prompt = self._assemble_final_prompt(
            messages, system_prompt_parts, settings, has_tool_results=False
        )

        response = await run_claude_async(prompt, settings=settings)
        return self._convert_response(
            response,
            output_tools=output_tools,
            function_tools=[],
            settings=settings,
        )

    async def _handle_unstructured_follow_up(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        original_settings: ClaudeCodeSettings | None = None,
    ) -> ModelResponse:
        """Handle unstructured follow-up request when function selection is 'none'."""
        settings = self._build_options(None, model_request_parameters)

        if original_settings:
            for key in ["additional_files", "timeout_seconds", "debug_save_prompts"]:
                if key in original_settings:
                    settings[key] = original_settings[key]

        settings["__function_selection_mode__"] = False
        self._prepare_working_directory(settings)

        system_prompt_parts = []
        if model_request_parameters and hasattr(model_request_parameters, "system_prompt"):
            sp = getattr(model_request_parameters, "system_prompt", None)
            if sp:
                system_prompt_parts.append(sp)

        unstructured_instruction = self._build_unstructured_output_instruction(settings)
        system_prompt_parts.append(unstructured_instruction)

        prompt = self._assemble_final_prompt(
            messages, system_prompt_parts, settings, has_tool_results=False
        )

        response = await run_claude_async(prompt, settings=settings)
        return self._convert_response(
            response,
            output_tools=[],
            function_tools=[],
            settings=settings,
        )

    # ===== Argument Collection Methods =====

    async def _handle_argument_collection(
        self,
        messages: list[ModelMessage],
        selected_function: str,
        available_functions: dict[str, Any],
        arg_response_for_usage: ClaudeJSONResponse,
        original_settings: ClaudeCodeSettings | None = None,
    ) -> ModelResponse:
        """Handle argument collection for selected function."""
        tool_def = available_functions.get(selected_function)
        if not tool_def:
            return self._create_model_response_with_usage(
                arg_response_for_usage,
                [TextPart(content=f"Error: Function '{selected_function}' not found")],
            )

        settings = self._build_options(None, ModelRequestParameters())

        if original_settings:
            for key in ["additional_files", "timeout_seconds", "debug_save_prompts"]:
                if key in original_settings:
                    settings[key] = original_settings[key]

        schema = tool_def.parameters_json_schema
        settings["__tool_name"] = tool_def.name
        settings["__tool_description"] = tool_def.description

        working_dir = _determine_working_directory(settings)
        settings["__working_directory"] = working_dir

        instruction = self._build_argument_collection_instruction(
            schema, settings, tool_def.name, tool_def.description
        )

        # Format messages
        formatted_messages = format_messages_for_claude(
            messages, skip_system_prompt=True, working_dir=working_dir
        )
        user_request_path = Path(working_dir) / "user_request.md"
        user_request_path.parent.mkdir(parents=True, exist_ok=True)
        with open(user_request_path, "w", encoding="utf-8") as f:
            f.write(formatted_messages)

        arg_prompt = instruction
        existing_prompt = settings.get("append_system_prompt")
        if existing_prompt:
            arg_prompt = f"{existing_prompt}\n\n{arg_prompt}"
            settings.pop("append_system_prompt", None)

        # Try collection with retries
        max_retries = 1
        error_msg = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                arg_prompt = self._build_retry_prompt(
                    messages, schema, settings, error_msg or ""
                )

            arg_response = await run_claude_async(arg_prompt, settings=settings)

            structured_file = settings.get("__structured_output_file")
            if structured_file:
                parsed_args, error_msg = self._read_structured_output_file(
                    structured_file, schema, settings
                )

                if parsed_args:
                    tool_call = create_tool_call_part(
                        tool_name=str(selected_function),
                        args=parsed_args,
                    )
                    return self._create_model_response_with_usage(
                        arg_response, [tool_call]
                    )

                if error_msg and attempt < max_retries:
                    continue

                if error_msg:
                    return self._create_model_response_with_usage(
                        arg_response, [TextPart(content=error_msg)]
                    )

        result_text = arg_response.get("result", "")
        error_msg = f"Could not interpret the parameters from response: {result_text[:500]}"
        return self._create_model_response_with_usage(
            arg_response, [TextPart(content=error_msg)]
        )

    def _build_argument_collection_instruction(
        self,
        schema: dict[str, Any],
        settings: ClaudeCodeSettings,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> str:
        """Build instruction for argument collection using file/folder structure."""
        working_dir = settings.get("__working_directory", "/tmp")
        output_filename = generate_output_file_path(
            working_dir, "claude_structured_output", ".json"
        )
        settings["__structured_output_file"] = output_filename

        temp_data_dir = generate_temp_directory_path(
            working_dir, "claude_data_structure", short_id=True
        )
        settings["__temp_json_dir"] = temp_data_dir

        return build_structure_instructions(
            schema, temp_data_dir, tool_name, tool_description
        )

    def _build_retry_prompt(
        self,
        messages: list[ModelMessage],
        schema: dict[str, Any],
        arg_settings: ClaudeCodeSettings,
        error_msg: str,
    ) -> str:
        """Build prompt for retry attempt after validation error."""
        working_dir = arg_settings.get("__working_directory", "/tmp")
        temp_data_dir = generate_temp_directory_path(
            working_dir, "claude_data_structure", short_id=True
        )
        arg_settings["__temp_json_dir"] = temp_data_dir

        tool_name: str | None = arg_settings.get("__tool_name")
        tool_description: str | None = arg_settings.get("__tool_description")

        instruction = self._build_argument_collection_instruction(
            schema, arg_settings, tool_name, tool_description
        )

        retry_instruction = f"""
PREVIOUS ATTEMPT HAD ERRORS:
{error_msg}

Please fix the issues above and try again. Follow the directory structure instructions carefully."""

        formatted_messages = format_messages_for_claude(
            messages, skip_system_prompt=True, working_dir=working_dir
        )
        user_request_path = Path(working_dir) / "user_request.md"
        user_request_path.parent.mkdir(parents=True, exist_ok=True)
        with open(user_request_path, "w", encoding="utf-8") as f:
            f.write(formatted_messages)

        return f"{instruction}\n\n{retry_instruction}"

    # ===== Utility Methods =====

    def _read_structured_output_file(
        self,
        file_path: str,
        schema: dict[str, Any],
        settings: ClaudeCodeSettings | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Read and validate structured output file."""
        # Try reading from directory structure first
        temp_json_dir = settings.get("__temp_json_dir") if settings else None
        if temp_json_dir and isinstance(temp_json_dir, str):
            temp_path = Path(temp_json_dir)
            if temp_path.exists():
                try:
                    parsed_data = read_structure_from_filesystem(schema, temp_path)
                    return parsed_data, None
                except RuntimeError as e:
                    return None, str(e)
                except Exception as e:
                    return None, f"Could not read the data structure: {e}"

        # Fall back to reading JSON file
        if not Path(file_path).exists():
            return None, None

        try:
            with open(file_path, encoding="utf-8") as f:
                file_content = f.read()
            parsed_data = json.loads(file_content)
            validation_error = self._validate_json_schema(parsed_data, schema)
            if validation_error:
                return None, validation_error
            return parsed_data, None
        except json.JSONDecodeError as e:
            return None, f"The file content isn't formatted correctly: {e}"
        except Exception as e:
            return None, f"Failed to read file: {e}"

    def _validate_json_schema(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> str | None:
        """Validate JSON data against schema."""
        required_fields = schema.get("required", [])
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return f"Please provide: {', '.join(missing_fields)}"

        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name in data:
                expected_type = field_schema.get("type")
                actual_value = data[field_name]

                type_valid = True
                if expected_type == "string" and not isinstance(actual_value, str):
                    type_valid = False
                elif expected_type == "integer" and not isinstance(actual_value, int):
                    type_valid = False
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    type_valid = False
                elif expected_type == "boolean" and not isinstance(actual_value, bool):
                    type_valid = False
                elif expected_type == "array" and not isinstance(actual_value, list):
                    type_valid = False
                elif expected_type == "object" and not isinstance(actual_value, dict):
                    type_valid = False

                if not type_valid:
                    return f"The value for '{field_name}' should be a {expected_type}"

        return None

    def _extract_json_robust(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Extract JSON from text using multiple strategies."""
        import re

        # Try markdown extraction
        cleaned = strip_markdown_code_fence(text)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try JSON object extraction
        json_pattern = r"\{(?:[^{}]|\{[^{}]*\})*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # Try array extraction
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

        # Try single field autowrap
        properties = schema.get("properties", {})
        if len(properties) == 1:
            field_name = list(properties.keys())[0]
            field_type = properties[field_name].get("type")
            value = text.strip()

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            converted = convert_primitive_value(value, field_type)
            if converted is not None:
                return {field_name: converted}

        raise json.JSONDecodeError("Could not extract valid JSON from response", text, 0)

    def _get_model_name(self, response: ClaudeJSONResponse) -> str:
        """Extract model name from Claude response."""
        model_name = self._model_alias
        if "modelUsage" in response and response.get("modelUsage"):
            model_names = list(response["modelUsage"].keys())
            if model_names:
                model_name = model_names[0]
        return model_name

    @staticmethod
    def _create_usage(response: ClaudeJSONResponse) -> RequestUsage:
        """Create usage info from Claude response."""
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
            input_tokens=usage_data.get("input_tokens", 0) if isinstance(usage_data, dict) else 0,
            cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0) if isinstance(usage_data, dict) else 0,
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0) if isinstance(usage_data, dict) else 0,
            output_tokens=usage_data.get("output_tokens", 0) if isinstance(usage_data, dict) else 0,
            details={
                "web_search_requests": web_search_requests,
                "total_cost_usd_cents": int(response.get("total_cost_usd", 0.0) * 100),
                "duration_ms": response.get("duration_ms", 0),
                "duration_api_ms": response.get("duration_api_ms", 0),
                "num_turns": response.get("num_turns", 0),
            },
        )
