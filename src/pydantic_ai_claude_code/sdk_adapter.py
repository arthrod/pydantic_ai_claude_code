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
        Convert a sequence of Pydantic AI messages into a single SDK-formatted prompt string.
        
        Each message part is rendered with a role prefix:
        - System parts as "System: {content}" (included only when include_system is True)
        - User parts as "User: {content}"
        - Assistant text parts as "Assistant: {content}"
        - Tool return parts as "Tool Result ({tool_name}): {content}"
        - Tool call parts as "Tool Call: {tool_name}({args})"
        
        Parameters:
            messages (list[ModelMessage]): Messages to convert; may be ModelRequest (with prompt parts) or ModelResponse (with response parts).
            include_system (bool): Whether to include system prompt parts in the output.
            model_request_parameters (Optional[Any]): Optional model request parameters (not used in formatting, accepted for API compatibility).
        
        Returns:
            str: The prompt string with message parts joined by two newlines.
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
        Convert a sequence of SDK-formatted messages into a Pydantic AI ModelResponse.
        
        Parameters:
            sdk_messages (list[Any]): Messages from the Claude SDK; each item may be a dict or object with a `type` of "assistant", "result", or "tool_use". Relevant content, tool call data, and usage are extracted when present.
            model_name (str): Model name to set on the returned ModelResponse.
        
        Returns:
            ModelResponse: A response containing one or more Parts (TextPart or ToolCallPart), a UTC timestamp, and usage data if available.
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
        """
        Extract the assistant's textual content from an SDK-style message.
        
        Parameters:
            msg (Any): Assistant message in either dict form or an object with a `content` attribute.
                Supported shapes:
                - dict with a string `content` or a list of blocks where blocks may be strings or dicts with `"type": "text"` and `"text"`.
                - object with a string `content` or an iterable of block objects with `text` attributes or `type == "text"`.
        
        Returns:
            str | None: Concatenated text from all text blocks separated by newlines if any text is found, otherwise `None`.
        """
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
        """
        Retrieve the "result" value from a result message.
        
        Parameters:
            msg (Any): A message represented as a dict or an object; for dicts the function reads the "result" key, for objects it reads the `result` attribute.
        
        Returns:
            The result string from the message, or an empty string if no result is present.
        """
        if isinstance(msg, dict):
            return msg.get("result", "")
        else:
            return getattr(msg, "result", "")

    def _extract_usage(self, msg: Any) -> Optional[RequestUsage]:
        """
        Extract token-usage data from a message that may be a dict or an object.
        
        Returns:
            `RequestUsage` with `input_tokens` and `output_tokens` (missing fields default to 0) if usage information is present, `None` otherwise.
        """
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
        """
        Builds a ToolCallPart from an SDK tool-use message.
        
        Parameters:
        	msg (Any): Message object or dict expected to contain `name`, `input`, and `id` fields. If `name` is missing the tool name is set to "unknown"; if `input` is missing `args` defaults to an empty dict; if `id` is missing a unique tool call id is generated.
        
        Returns:
        	ToolCallPart: A ToolCallPart constructed from the message's tool name, arguments, and tool call id.
        """
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
        """
        Generate a unique identifier for a tool call.
        
        Returns:
            tool_call_id (str): Identifier string formatted as "call_<16-hex-chars>".
        """
        return f"call_{uuid.uuid4().hex[:16]}"

    def model_response_to_dict(self, response: ModelResponse) -> dict[str, Any]:
        """
        Convert a ModelResponse into a plain dictionary suitable for serialization.
        
        Returns:
            A dictionary with keys:
              - "model_name": the response.model_name.
              - "timestamp": ISO-formatted timestamp or None.
              - "parts": a list where each item is either
                  {"type": "text", "content": <str>} or
                  {"type": "tool_call", "tool_name": <str>, "args": <Any>, "tool_call_id": <str>}.
              - "usage" (optional): a dict with "input_tokens", "output_tokens", and "total_tokens".
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
    """
    Return the shared SDKAdapter singleton instance.
    
    Returns:
        SDKAdapter: The global SDKAdapter instance, creating it on first access if necessary.
    """
    global _adapter
    if _adapter is None:
        _adapter = SDKAdapter()
    return _adapter