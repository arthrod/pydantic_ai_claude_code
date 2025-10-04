"""Custom StreamedResponse implementation for Claude Code model."""

from datetime import datetime, timezone
from typing import AsyncIterator

from pydantic_ai.messages import (
    FinalResultEvent,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.usage import RequestUsage

from .types import ClaudeStreamEvent


class ClaudeCodeStreamedResponse(StreamedResponse):
    """StreamedResponse implementation for Claude Code CLI."""

    def __init__(
        self,
        model_request_parameters: ModelRequestParameters,
        model_name: str,
        event_stream: AsyncIterator[ClaudeStreamEvent],
        timestamp: datetime | None = None,
    ):
        """Initialize streamed response.

        Args:
            model_request_parameters: Model request parameters
            model_name: Name of the model
            event_stream: Async iterator of Claude stream events
            timestamp: Timestamp of the response
        """
        super().__init__(model_request_parameters)
        self._model_name = model_name
        self._event_stream = event_stream
        self._timestamp = timestamp or datetime.now(timezone.utc)
        self._usage: RequestUsage | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "claude-code"

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp."""
        return self._timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Get event iterator for streaming.

        Yields:
            Model response stream events
        """
        text_started = False
        full_text = ""

        async for event in self._event_stream:
            event_type = event.get("type")

            if event_type == "assistant":
                # Extract text from assistant message
                message = event.get("message", {})
                content = message.get("content", [])

                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")

                        if not text_started:
                            # First chunk - send PartStartEvent
                            text_started = True
                            yield PartStartEvent(
                                index=0,
                                part=TextPart(content=text),
                            )
                            full_text = text
                        else:
                            # Subsequent chunks - send delta
                            delta = text[len(full_text):]
                            if delta:
                                yield PartDeltaEvent(
                                    index=0,
                                    delta=TextPartDelta(content_delta=delta),
                                )
                                full_text = text

            elif event_type == "result":
                # Final result event
                # Extract usage if available
                usage_data = event.get("usage", {})
                self._usage = RequestUsage(
                    input_tokens=usage_data.get("input_tokens", 0),
                    cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
                    cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                )

                yield FinalResultEvent(tool_name=None, tool_call_id=None)
