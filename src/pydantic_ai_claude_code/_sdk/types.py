"""SDK Type Definitions for Claude Code Integration.

These types provide SDK-compatible interfaces while maintaining
pydantic_ai compatibility. Copied from claude_agent_sdk to avoid
external dependencies.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Union

from typing_extensions import TypedDict

# Permission modes supported by Claude Code
PermissionMode = Literal["bypassPermissions", "acceptEdits", "default", "plan"]


class TextBlock(TypedDict, total=False):
    """Text content block."""

    type: Literal["text"]
    text: str


class ToolUseBlock(TypedDict, total=False):
    """Tool use request block."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(TypedDict, total=False):
    """Tool result block."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: str
    is_error: bool


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


class UserMessage(TypedDict, total=False):
    """User message in SDK format."""

    role: Literal["user"]
    content: Union[str, list[ContentBlock]]


class AssistantMessage(TypedDict, total=False):
    """Assistant message in SDK format."""

    role: Literal["assistant"]
    content: Union[str, list[ContentBlock]]
    stop_reason: str | None


class ResultMessage(TypedDict, total=False):
    """Result message with metadata."""

    type: Literal["result"]
    subtype: Literal["success", "error"]
    is_error: bool
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    result: str
    session_id: str
    total_cost_usd: float
    usage: dict[str, Any]


Message = Union[UserMessage, AssistantMessage, ResultMessage]


class ToolPermissionContext(TypedDict, total=False):
    """Context for tool permission decisions."""

    session_id: str
    turn_number: int
    tool_history: list[dict[str, Any]]
    working_directory: str


class PermissionResultAllow(TypedDict):
    """Result allowing tool execution."""

    behavior: Literal["allow"]
    updated_input: dict[str, Any] | None


class PermissionResultDeny(TypedDict):
    """Result denying tool execution."""

    behavior: Literal["deny"]
    message: str


PermissionResult = Union[PermissionResultAllow, PermissionResultDeny]


# Callback type for tool permission decisions
CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    PermissionResult,
]


class HookMatcher(TypedDict, total=False):
    """Matcher for hook events."""

    event: str
    tool_name: str | None
    pattern: str | None


class HookEvent(TypedDict, total=False):
    """Hook event data."""

    type: str
    tool_name: str | None
    tool_input: dict[str, Any] | None
    result: str | None
    error: str | None


class HookConfig(TypedDict, total=False):
    """Hook configuration for Claude Code."""

    matcher: HookMatcher
    commands: list[str]
    timeout_ms: int | None


class ClaudeAgentOptions(TypedDict, total=False):
    """Options for Claude Agent SDK.

    These options configure the Claude Code CLI execution.
    Settings are merged in order: Provider defaults -> Agent config -> Run overrides.
    """

    # Model configuration
    model: str
    fallback_model: str | None

    # Working directory
    cwd: str | None

    # Tool permissions
    allowed_tools: list[str]
    disallowed_tools: list[str]
    permission_mode: PermissionMode
    can_use_tool: CanUseTool | None

    # Execution limits
    max_turns: int | None
    max_budget_usd: float | None
    timeout_ms: int | None

    # CLI configuration
    cli_path: str | None
    extra_args: dict[str, str | None]

    # Hooks
    hooks: list[HookConfig] | None

    # System prompt
    system_prompt: str | None
    append_system_prompt: str | None

    # Session management
    session_id: str | None
    resume_session: bool

    # Debug options
    verbose: bool
    debug: bool

    # Additional files to include
    additional_files: list[str] | None

    # Sandbox configuration
    use_sandbox_runtime: bool
    sandbox_runtime_path: str | None


class SDKUsage(TypedDict, total=False):
    """Usage statistics from SDK."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_cost_usd: float


class SDKResponse(TypedDict, total=False):
    """Response from SDK execution."""

    messages: list[Message]
    final_result: str
    usage: SDKUsage
    session_id: str
    duration_ms: int
