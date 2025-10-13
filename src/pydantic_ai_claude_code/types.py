"""Type definitions for Claude Code model."""

from typing import Any, Literal, TypedDict


class ClaudeUsage(TypedDict, total=False):
    """Usage statistics from Claude CLI."""

    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    web_search_requests: int
    service_tier: str


class ClaudeModelUsage(TypedDict, total=False):
    """Per-model usage statistics."""

    inputTokens: int
    outputTokens: int
    cacheReadInputTokens: int
    cacheCreationInputTokens: int
    webSearchRequests: int
    costUSD: float
    contextWindow: int


class ClaudeJSONResponse(TypedDict, total=False):
    """Response structure from Claude CLI JSON output."""

    type: Literal["result"]
    subtype: Literal["success", "error"]
    is_error: bool
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    result: str
    session_id: str
    total_cost_usd: float
    usage: ClaudeUsage
    modelUsage: dict[str, ClaudeModelUsage]
    permission_denials: list[Any]
    uuid: str
    error: str | None


class ClaudeStreamSystemEvent(TypedDict, total=False):
    """System initialization event in stream-json output."""

    type: Literal["system"]
    subtype: Literal["init"]
    cwd: str
    session_id: str
    tools: list[str]
    mcp_servers: list[dict[str, Any]]
    model: str
    permissionMode: str
    slash_commands: list[str]
    apiKeySource: str
    output_style: str
    agents: list[str]
    uuid: str


class ClaudeStreamAssistantEvent(TypedDict, total=False):
    """Assistant message event in stream-json output."""

    type: Literal["assistant"]
    message: dict[str, Any]
    parent_tool_use_id: str | None
    session_id: str
    uuid: str


class ClaudeStreamResultEvent(TypedDict, total=False):
    """Final result event in stream-json output."""

    type: Literal["result"]
    subtype: Literal["success", "error"]
    is_error: bool
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    result: str
    session_id: str
    total_cost_usd: float
    usage: ClaudeUsage
    modelUsage: dict[str, ClaudeModelUsage]
    permission_denials: list[Any]
    uuid: str
    error: str | None


ClaudeStreamEvent = (
    ClaudeStreamSystemEvent | ClaudeStreamAssistantEvent | ClaudeStreamResultEvent
)


class ClaudeCodeSettings(TypedDict, total=False):
    """Settings for Claude Code CLI execution."""

    working_directory: str | None
    allowed_tools: list[str] | None
    disallowed_tools: list[str] | None
    append_system_prompt: str | None
    permission_mode: (
        Literal["acceptEdits", "bypassPermissions", "default", "plan"] | None
    )
    model: str | None
    fallback_model: str | None
    max_turns: int | None
    session_id: str | None
    verbose: bool
    dangerously_skip_permissions: bool
    retry_on_rate_limit: bool
    __structured_output_file: str  # Internal: temp file path for structured output
    __unstructured_output_file: str  # Internal: temp file path for unstructured output
    __function_call_file: str  # Internal: temp file path for function call JSON
