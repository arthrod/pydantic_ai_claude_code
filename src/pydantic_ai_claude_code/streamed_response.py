"""Custom StreamedResponse implementation for Claude Code model."""

import asyncio
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
        streaming_marker: str | None = None,
    ):
        """Initialize streamed response.

        Args:
            model_request_parameters: Model request parameters
            model_name: Name of the model
            event_stream: Async iterator of Claude stream events
            timestamp: Timestamp of the response
            streaming_marker: Marker to watch for to start streaming final response
        """
        super().__init__(model_request_parameters)
        self._model_name = model_name
        self._event_stream = event_stream
        self._timestamp = timestamp or datetime.now(timezone.utc)
        self._streaming_marker = streaming_marker
        # Initialize with zero usage, will be updated when result event arrives
        self._usage: RequestUsage = RequestUsage()
        # Buffer events as they arrive from background task
        self._buffered_events: list[ModelResponseStreamEvent] = []
        self._stream_complete = asyncio.Event()
        self._background_task: asyncio.Task[None] | None = None
        # Start background consumption immediately
        self._background_task = asyncio.create_task(self._consume_stream_background())

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

    def _process_marker_and_text(
        self, text_chunk: str, accumulated_text: str, streaming_started: bool, text_started: bool
    ) -> tuple[str, bool, bool]:
        """Process text chunk with marker detection.

        Args:
            text_chunk: New text to process
            accumulated_text: Text accumulated so far
            streaming_started: Whether streaming has started
            text_started: Whether text events have started

        Returns:
            Tuple of (updated_accumulated_text, updated_streaming_started, updated_text_started)
        """
        accumulated_text += text_chunk

        if not streaming_started and self._streaming_marker:
            if self._streaming_marker in accumulated_text:
                # Marker found - start streaming from after marker
                marker_end = accumulated_text.index(self._streaming_marker) + len(
                    self._streaming_marker
                )
                remaining_text = accumulated_text[marker_end:].lstrip()
                accumulated_text = remaining_text

                if remaining_text:
                    start_event = PartStartEvent(index=0, part=TextPart(content=""))
                    self._buffered_events.append(start_event)
                    text_started = True

                    delta_event = PartDeltaEvent(
                        index=0, delta=TextPartDelta(content_delta=remaining_text)
                    )
                    self._buffered_events.append(delta_event)
                return accumulated_text, True, text_started
        elif streaming_started:
            # Already streaming - add text events
            if not text_started:
                start_event = PartStartEvent(index=0, part=TextPart(content=""))
                self._buffered_events.append(start_event)
                text_started = True

            delta_event = PartDeltaEvent(
                index=0, delta=TextPartDelta(content_delta=text_chunk)
            )
            self._buffered_events.append(delta_event)

        return accumulated_text, streaming_started, text_started

    async def _consume_stream_background(self) -> None:
        """Consume CLI stream in background and populate buffer.

        This runs as a background task so that both __aenter__ and stream_text()
        can read from the buffer as it grows.
        """
        logger.debug("Background task started, consuming CLI stream")
        try:
            text_started = False
            streaming_started = False
            accumulated_text = ""
            event_count = 0

            async for event in self._event_stream:
                event_count += 1
                event_type = event.get("type")

                if event_type == "message_start":
                    continue

                if event_type == "content_block_delta":
                    # Only process first content block
                    if event.get("index") != 0:
                        continue

                    delta = event.get("delta", {})
                    if not (isinstance(delta, dict) and delta.get("type") == "text_delta"):
                        continue

                    text_chunk = delta.get("text", "")
                    if not text_chunk:
                        continue

                    # Process text with marker detection
                    accumulated_text, streaming_started, text_started = self._process_marker_and_text(
                        text_chunk, accumulated_text, streaming_started, text_started
                    )

                elif event_type in ("message_delta", "message_stop", "assistant"):
                    continue

                elif event_type == "result":
                    result_event = self._handle_result_event(event, event_count)
                    self._buffered_events.append(result_event)

            logger.debug("Background stream consumption complete, buffered %d events", len(self._buffered_events))
        finally:
            # Signal that stream is complete
            self._stream_complete.set()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Get event iterator for streaming.

        Yields events from the buffer as they arrive from the background task.
        Multiple consumers can call this and each will get the events.

        Yields:
            Model response stream events
        """
        logger.debug("Starting event iterator, yielding from growing buffer")
        index = 0

        # Yield events as they arrive in buffer
        while True:
            # Wait for more events or completion
            while index >= len(self._buffered_events):
                if self._stream_complete.is_set():
                    # Stream finished and we've yielded all events
                    logger.debug("Event iterator complete, yielded %d events", index)
                    return
                # Wait a bit for more events
                await asyncio.sleep(0.01)

            # Yield the next event
            event = self._buffered_events[index]
            yield event
            index += 1
