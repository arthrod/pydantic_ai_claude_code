"""
SDK Adapter for type conversions between SDK and Pydantic AI.

Provides conversion functions for:
- Messages: SDK format <-> Pydantic AI format
- Tool calls: SDK blocks <-> Pydantic AI parts
- Usage: SDK statistics <-> Pydantic AI RequestUsage
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from .sdk_original_files.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    ContentBlock,
    Message,
    ResultMessage,
    SDKResponse,
    SDKUsage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger(__name__)


class SDKAdapter:
    """
    Adapter for converting between SDK and Pydantic AI formats.

    Handles bidirectional conversion for:
    - Messages (prompts and responses)
    - Tool calls and results
    - Usage statistics
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._tool_call_counter = 0

    def messages_to_prompt(
        self,
        messages: List[ModelMessage],
        include_system: bool = True,
        model_request_parameters: Optional[ModelRequestParameters] = None,
    ) -> str:
        """
        Convert Pydantic AI messages to prompt string.

        Args:
            messages: List of Pydantic AI messages
            include_system: Whether to include system prompts
            model_request_parameters: Additional parameters

        Returns:
            Formatted prompt string
        """
        prompt_parts: List[str] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart) and include_system:
                        prompt_parts.append(f"System: {part.content}")
                    elif isinstance(part, UserPromptPart):
                        if isinstance(part.content, str):
                            prompt_parts.append(f"Request: {part.content}")
                        # Handle binary content separately
                    elif isinstance(part, ToolReturnPart):
                        result_str = (
                            part.content if isinstance(part.content, str) else str(part.content)
                        )
                        prompt_parts.append(
                            f"Tool Result ({part.tool_name}): {result_str}"
                        )

            elif isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, TextPart):
                        prompt_parts.append(f"Assistant: {part.content}")
                    elif isinstance(part, ToolCallPart):
                        # Format tool call for context
                        import json
                        args_str = json.dumps(part.args, indent=2)
                        prompt_parts.append(
                            f"Tool Call ({part.tool_name}): {args_str}"
                        )

        # Add system prompt from parameters if provided
        if model_request_parameters and model_request_parameters.system_prompt:
            prompt_parts.insert(0, f"System: {model_request_parameters.system_prompt}")

        return "\n\n".join(prompt_parts)

    def messages_to_sdk_format(
        self,
        messages: List[ModelMessage],
    ) -> List[Message]:
        """
        Convert Pydantic AI messages to SDK format.

        Args:
            messages: List of Pydantic AI messages

        Returns:
            List of SDK messages
        """
        sdk_messages: List[Message] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                content_blocks: List[ContentBlock] = []

                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        if isinstance(part.content, str):
                            content_blocks.append(
                                TextBlock(type="text", text=part.content)
                            )
                    elif isinstance(part, ToolReturnPart):
                        result_str = (
                            part.content
                            if isinstance(part.content, str)
                            else str(part.content)
                        )
                        content_blocks.append(
                            ToolResultBlock(
                                type="tool_result",
                                tool_use_id=part.tool_call_id or "",
                                content=result_str,
                                is_error=False,
                            )
                        )

                if content_blocks:
                    sdk_messages.append(
                        UserMessage(role="user", content=content_blocks)
                    )

            elif isinstance(message, ModelResponse):
                content_blocks = []

                for part in message.parts:
                    if isinstance(part, TextPart):
                        content_blocks.append(
                            TextBlock(type="text", text=part.content)
                        )
                    elif isinstance(part, ToolCallPart):
                        import json
                        content_blocks.append(
                            ToolUseBlock(
                                type="tool_use",
                                id=part.tool_call_id or self._generate_tool_call_id(),
                                name=part.tool_name,
                                input=part.args,
                            )
                        )

                if content_blocks:
                    sdk_messages.append(
                        AssistantMessage(role="assistant", content=content_blocks)
                    )

        return sdk_messages

    def sdk_to_model_response(
        self,
        sdk_messages: List[Message],
        model_name: str = "claude-code",
    ) -> ModelResponse:
        """
        Convert SDK messages to Pydantic AI ModelResponse.

        Args:
            sdk_messages: List of SDK messages
            model_name: Model name for response

        Returns:
            Pydantic AI ModelResponse
        """
        parts: List[TextPart | ToolCallPart] = []
        usage_data: Dict[str, Any] = {}

        for message in sdk_messages:
            # Handle result messages
            if message.get("type") == "result":
                result_msg: ResultMessage = message  # type: ignore
                # Extract usage from result
                if "usage" in result_msg:
                    usage_data = result_msg["usage"]
                # Add result text if present
                if result_msg.get("result"):
                    parts.append(TextPart(content=result_msg["result"]))
                continue

            # Handle assistant messages
            role = message.get("role")
            if role == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(TextPart(content=content))
                elif isinstance(content, list):
                    for block in content:
                        block_type = block.get("type")
                        if block_type == "text":
                            parts.append(TextPart(content=block.get("text", "")))
                        elif block_type == "tool_use":
                            parts.append(
                                ToolCallPart(
                                    tool_name=block.get("name", ""),
                                    args=block.get("input", {}),
                                    tool_call_id=block.get(
                                        "id", self._generate_tool_call_id()
                                    ),
                                )
                            )

        # Build usage
        usage = self._build_usage(usage_data)

        return ModelResponse(
            parts=parts,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            usage=usage,
        )

    def sdk_response_to_model_response(
        self,
        response: SDKResponse,
        model_name: str = "claude-code",
    ) -> ModelResponse:
        """
        Convert SDKResponse to Pydantic AI ModelResponse.

        Args:
            response: SDK response object
            model_name: Model name

        Returns:
            Pydantic AI ModelResponse
        """
        return self.sdk_to_model_response(
            response.get("messages", []),
            model_name=model_name,
        )

    def _build_usage(self, usage_data: Dict[str, Any]) -> RequestUsage:
        """Build RequestUsage from SDK usage data."""
        return RequestUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        )

    def _generate_tool_call_id(self) -> str:
        """Generate unique tool call ID."""
        self._tool_call_counter += 1
        return f"call_{uuid.uuid4().hex[:16]}"

    def options_to_settings(
        self,
        options: ClaudeAgentOptions,
    ) -> Dict[str, Any]:
        """
        Convert SDK options to ClaudeCodeSettings format.

        Args:
            options: SDK agent options

        Returns:
            Settings dictionary
        """
        settings: Dict[str, Any] = {}

        if options.get("model"):
            settings["model"] = options["model"]

        if options.get("cwd"):
            settings["working_directory"] = options["cwd"]

        if options.get("allowed_tools"):
            settings["allowed_tools"] = options["allowed_tools"]

        if options.get("disallowed_tools"):
            settings["disallowed_tools"] = options["disallowed_tools"]

        if options.get("permission_mode"):
            settings["permission_mode"] = options["permission_mode"]

        if options.get("cli_path"):
            settings["claude_cli_path"] = options["cli_path"]

        if options.get("max_turns"):
            settings["max_turns"] = options["max_turns"]

        if options.get("timeout_ms"):
            settings["timeout_seconds"] = options["timeout_ms"] // 1000

        if options.get("system_prompt"):
            settings["append_system_prompt"] = options["system_prompt"]

        if options.get("verbose"):
            settings["verbose"] = options["verbose"]

        return settings

    def settings_to_options(
        self,
        settings: Dict[str, Any],
    ) -> ClaudeAgentOptions:
        """
        Convert ClaudeCodeSettings to SDK options format.

        Args:
            settings: Settings dictionary

        Returns:
            SDK agent options
        """
        options: ClaudeAgentOptions = ClaudeAgentOptions()

        if "model" in settings:
            options["model"] = settings["model"]

        if "working_directory" in settings:
            options["cwd"] = str(settings["working_directory"])

        if "allowed_tools" in settings:
            options["allowed_tools"] = settings["allowed_tools"]

        if "disallowed_tools" in settings:
            options["disallowed_tools"] = settings["disallowed_tools"]

        if "permission_mode" in settings:
            options["permission_mode"] = settings["permission_mode"]

        if "claude_cli_path" in settings:
            options["cli_path"] = settings["claude_cli_path"]

        if "timeout_seconds" in settings:
            options["timeout_ms"] = settings["timeout_seconds"] * 1000

        if "append_system_prompt" in settings:
            options["append_system_prompt"] = settings["append_system_prompt"]

        if "verbose" in settings:
            options["verbose"] = settings["verbose"]

        return options
