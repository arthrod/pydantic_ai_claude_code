"""Custom StreamedResponse implementation for Claude Code model."""

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone

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

logger = logging.getLogger(__name__)


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
        # Initialize with zero usage, will be updated when result event arrives
        self._usage: RequestUsage = RequestUsage()

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

    def _handle_assistant_event(
        self,
        event: ClaudeStreamEvent,
        text_started: bool,
        full_text: str,
    ) -> tuple[ModelResponseStreamEvent | None, bool, str]:
        """Handle assistant message event.

        Args:
            event: Claude stream event
            text_started: Whether text streaming has started
            full_text: Full text accumulated so far

        Returns:
            Tuple of (stream_event, updated_text_started, updated_full_text)
        """
        message = event.get("message", {})
        if not isinstance(message, dict):
            return None, text_started, full_text

        content = message.get("content", [])
        if not isinstance(content, list):
            return None, text_started, full_text

        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")

                if not text_started:
                    # First chunk - send PartStartEvent
                    logger.debug("Streaming first text chunk: %d chars", len(text))
                    return (
                        PartStartEvent(index=0, part=TextPart(content=text)),
                        True,
                        text,
                    )

                # Subsequent chunks - send delta
                delta = text[len(full_text) :]
                if delta:
                    logger.debug("Streaming text delta: %d chars", len(delta))
                    return (
                        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=delta)),
                        text_started,
                        text,
                    )

        return None, text_started, full_text

    def _handle_result_event(
        self, event: ClaudeStreamEvent, event_count: int
    ) -> FinalResultEvent:
        """Handle result event.

        Args:
            event: Claude stream event
            event_count: Number of events processed

        Returns:
            Final result event
        """
        usage_data = event.get("usage", {})
        if isinstance(usage_data, dict):
            self._usage = RequestUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            )

        logger.debug(
            "Streaming completed: %d events, %d output tokens",
            event_count,
            self._usage.output_tokens if self._usage else 0,
        )
        return FinalResultEvent(tool_name=None, tool_call_id=None)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Get event iterator for streaming.

        Yields:
            Model response stream events
        """
        logger.debug("Starting streamed response event iteration")
        text_started = False
        full_text = ""
        event_count = 0

        async for event in self._event_stream:
            event_count += 1
            event_type = event.get("type")

            if event_type == "assistant":
                stream_event, text_started, full_text = self._handle_assistant_event(
                    event, text_started, full_text
                )
                if stream_event:
                    yield stream_event

            elif event_type == "result":
                yield self._handle_result_event(event, event_count)
