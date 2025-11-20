"""
SDK Type Definitions for Claude Code Integration.

These types provide SDK-compatible interfaces while maintaining
pydantic_ai compatibility.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Union
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
    input: Dict[str, Any]


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
    content: Union[str, List[ContentBlock]]


class AssistantMessage(TypedDict, total=False):
    """Assistant message in SDK format."""
    role: Literal["assistant"]
    content: Union[str, List[ContentBlock]]
    stop_reason: Optional[str]


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
    usage: Dict[str, Any]


Message = Union[UserMessage, AssistantMessage, ResultMessage]


class ToolPermissionContext(TypedDict, total=False):
    """Context for tool permission decisions."""
    session_id: str
    turn_number: int
    tool_history: List[Dict[str, Any]]
    working_directory: str


class PermissionResultAllow(TypedDict):
    """Result allowing tool execution."""
    behavior: Literal["allow"]
    updated_input: Optional[Dict[str, Any]]


class PermissionResultDeny(TypedDict):
    """Result denying tool execution."""
    behavior: Literal["deny"]
    message: str


PermissionResult = Union[PermissionResultAllow, PermissionResultDeny]


# Callback type for tool permission decisions
CanUseTool = Callable[
    [str, Dict[str, Any], ToolPermissionContext],
    PermissionResult
]


class HookMatcher(TypedDict, total=False):
    """Matcher for hook events."""
    event: str
    tool_name: Optional[str]
    pattern: Optional[str]


class HookEvent(TypedDict, total=False):
    """Hook event data."""
    type: str
    tool_name: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    result: Optional[str]
    error: Optional[str]


class PermissionUpdate(TypedDict, total=False):
    """Permission update notification."""
    tool_name: str
    permission: Literal["allow", "deny", "ask"]
    reason: Optional[str]


class ClaudeAgentOptions(TypedDict, total=False):
    """Options for Claude Agent SDK."""

    # Model configuration
    model: str
    fallback_model: Optional[str]

    # Working directory
    cwd: Optional[str]

    # Tool permissions
    allowed_tools: List[str]
    disallowed_tools: List[str]
    permission_mode: PermissionMode
    can_use_tool: Optional[CanUseTool]

    # Execution limits
    max_turns: Optional[int]
    max_budget_usd: Optional[float]
    timeout_ms: Optional[int]

    # CLI configuration
    cli_path: Optional[str]
    extra_args: Dict[str, Optional[str]]

    # Hooks
    hooks: Optional[List[Dict[str, Any]]]

    # System prompt
    system_prompt: Optional[str]
    append_system_prompt: Optional[str]

    # Session management
    session_id: Optional[str]
    resume_session: bool

    # Debug options
    verbose: bool
    debug: bool


class SDKUsage(TypedDict, total=False):
    """Usage statistics from SDK."""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_cost_usd: float


class SDKResponse(TypedDict, total=False):
    """Response from SDK execution."""
    messages: List[Message]
    final_result: str
    usage: SDKUsage
    session_id: str
    duration_ms: int
