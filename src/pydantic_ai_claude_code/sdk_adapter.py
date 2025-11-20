"""
SDK Adapter - Conversions between SDK and Pydantic AI formats.

This module provides adapters for converting between Claude Agent SDK
message formats and Pydantic AI message formats.
"""

from typing import Any, Optional
from datetime import datetime, timezone
import uuid
import logging

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.usage import RequestUsage

logger = logging.getLogger(__name__)


class SDKAdapter:
    """
    Adapter for SDK <-> Pydantic AI conversions.

    Handles conversion between:
    - Pydantic AI ModelMessage -> SDK prompt strings
    - SDK response messages -> Pydantic AI ModelResponse
    """

    def messages_to_prompt(
        self,
        messages: list[ModelMessage],
        include_system: bool = True,
        model_request_parameters: Optional[Any] = None,
    ) -> str:
        """
        Convert Pydantic AI messages to SDK prompt string.

        Args:
            messages: List of Pydantic AI messages
            include_system: Whether to include system prompts
            model_request_parameters: Optional parameters

        Returns:
            Formatted prompt string
        """
        parts = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        if include_system:
                            parts.append(f"System: {part.content}")
                    elif isinstance(part, UserPromptPart):
                        parts.append(f"User: {part.content}")
                    elif isinstance(part, ToolReturnPart):
                        parts.append(
                            f"Tool Result ({part.tool_name}): {part.content}"
                        )
            elif hasattr(message, "parts"):
                # ModelResponse
                for part in message.parts:
                    if isinstance(part, TextPart):
                        parts.append(f"Assistant: {part.content}")
                    elif isinstance(part, ToolCallPart):
                        parts.append(
                            f"Tool Call: {part.tool_name}({part.args})"
                        )

        return "\n\n".join(parts)

    def sdk_to_model_response(
        self,
        sdk_messages: list[Any],
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
        parts = []
        usage = None

        for msg in sdk_messages:
            if msg is None:
                continue

            # Handle different message types
            msg_type = getattr(msg, "type", None) or (
                msg.get("type") if isinstance(msg, dict) else None
            )

            if msg_type == "assistant":
                # Extract text content
                content = self._extract_assistant_content(msg)
                if content:
                    parts.append(TextPart(content=content))

            elif msg_type == "result":
                # Extract result and usage
                result = self._extract_result_content(msg)
                if result:
                    parts.append(TextPart(content=result))
                usage = self._extract_usage(msg)

            elif msg_type == "tool_use":
                # Extract tool call
                tool_call = self._extract_tool_call(msg)
                if tool_call:
                    parts.append(tool_call)

        # Ensure we have at least one part
        if not parts:
            parts.append(TextPart(content=""))

        return ModelResponse(
            parts=parts,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            usage=usage,
        )

    def _extract_assistant_content(self, msg: Any) -> Optional[str]:
        """Extract text content from assistant message."""
        if isinstance(msg, dict):
            # Dict format
            content = msg.get("content", [])
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        texts.append(block)
                return "\n".join(texts) if texts else None
        else:
            # Object format
            content = getattr(msg, "content", [])
            if isinstance(content, str):
                return content
            elif hasattr(content, "__iter__"):
                texts = []
                for block in content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                    elif hasattr(block, "type") and block.type == "text":
                        texts.append(getattr(block, "text", ""))
                return "\n".join(texts) if texts else None

        return None

    def _extract_result_content(self, msg: Any) -> Optional[str]:
        """Extract result content from result message."""
        if isinstance(msg, dict):
            return msg.get("result", "")
        else:
            return getattr(msg, "result", "")

    def _extract_usage(self, msg: Any) -> Optional[RequestUsage]:
        """Extract usage information from message."""
        if isinstance(msg, dict):
            usage_data = msg.get("usage", {})
        else:
            usage_data = getattr(msg, "usage", {})

        if not usage_data:
            return None

        if isinstance(usage_data, dict):
            return RequestUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            )
        else:
            return RequestUsage(
                input_tokens=getattr(usage_data, "input_tokens", 0),
                output_tokens=getattr(usage_data, "output_tokens", 0),
            )

    def _extract_tool_call(self, msg: Any) -> Optional[ToolCallPart]:
        """Extract tool call from message."""
        if isinstance(msg, dict):
            return ToolCallPart(
                tool_name=msg.get("name", "unknown"),
                args=msg.get("input", {}),
                tool_call_id=msg.get("id", self._generate_tool_call_id()),
            )
        else:
            return ToolCallPart(
                tool_name=getattr(msg, "name", "unknown"),
                args=getattr(msg, "input", {}),
                tool_call_id=getattr(msg, "id", self._generate_tool_call_id()),
            )

    def _generate_tool_call_id(self) -> str:
        """Generate unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:16]}"

    def model_response_to_dict(self, response: ModelResponse) -> dict[str, Any]:
        """
        Convert ModelResponse to dictionary format.

        Args:
            response: Pydantic AI ModelResponse

        Returns:
            Dictionary representation
        """
        result = {
            "model_name": response.model_name,
            "timestamp": response.timestamp.isoformat() if response.timestamp else None,
            "parts": [],
        }

        for part in response.parts:
            if isinstance(part, TextPart):
                result["parts"].append({
                    "type": "text",
                    "content": part.content,
                })
            elif isinstance(part, ToolCallPart):
                result["parts"].append({
                    "type": "tool_call",
                    "tool_name": part.tool_name,
                    "args": part.args,
                    "tool_call_id": part.tool_call_id,
                })

        if response.usage:
            result["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return result


# Singleton instance for convenience
_adapter = None


def get_adapter() -> SDKAdapter:
    """Get the global SDK adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = SDKAdapter()
    return _adapter
