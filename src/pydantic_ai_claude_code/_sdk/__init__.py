"""SDK core types and utilities for Claude Code integration.

This module contains copied SDK types to avoid external dependencies
while maintaining compatibility with the Claude Agent SDK patterns.
"""

from .types import (
    # Permission modes
    PermissionMode,
    # Content blocks
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ContentBlock,
    # Messages
    UserMessage,
    AssistantMessage,
    ResultMessage,
    Message,
    # Permission types
    ToolPermissionContext,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionResult,
    CanUseTool,
    # Hook types
    HookMatcher,
    HookEvent,
    HookConfig,
    # Options
    ClaudeAgentOptions,
    # Usage and response
    SDKUsage,
    SDKResponse,
)

from .errors import (
    ClaudeSDKError,
    CLIConnectionError,
    CLINotFoundError,
    ProcessError,
    TimeoutError,
    AuthenticationError,
    RateLimitError,
)

__all__ = [
    # Permission modes
    "PermissionMode",
    # Content blocks
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Messages
    "UserMessage",
    "AssistantMessage",
    "ResultMessage",
    "Message",
    # Permission types
    "ToolPermissionContext",
    "PermissionResultAllow",
    "PermissionResultDeny",
    "PermissionResult",
    "CanUseTool",
    # Hook types
    "HookMatcher",
    "HookEvent",
    "HookConfig",
    # Options
    "ClaudeAgentOptions",
    # Usage and response
    "SDKUsage",
    "SDKResponse",
    # Errors
    "ClaudeSDKError",
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "TimeoutError",
    "AuthenticationError",
    "RateLimitError",
]
