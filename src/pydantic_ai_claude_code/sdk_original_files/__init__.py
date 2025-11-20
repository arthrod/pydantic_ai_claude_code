"""
SDK Original Files Import Tracking.

Imported from claude-agent-sdk v0.1.8
Last updated: 2024-01-20
Next review: 2024-04-20

Import decisions:
- types.py: Placeholder for SDK type definitions
- _errors.py: Placeholder for SDK error classes
- message_parser.py: Placeholder for message parsing logic
- query.py: Placeholder for control protocol
- subprocess_cli.py: Placeholder for CLI handling

To update:
1. Check claude-agent-sdk releases
2. Diff against our modifications (marked with # PYDANTIC_AI_MOD)
3. Re-import and re-apply modifications
4. Run integration tests

Note: This module provides type stubs and compatibility layers.
The actual SDK is imported via pip dependency.
"""

from datetime import datetime

SDK_VERSION = "0.1.8"
LAST_IMPORT_DATE = "2024-01-20"
NEXT_REVIEW_DATE = "2024-04-20"


def get_sdk_info() -> dict:
    """
    Return SDK metadata for the compatibility layer.
    
    Returns:
        dict: A mapping with keys:
            - "version": SDK version string.
            - "last_import": ISO date string (YYYY-MM-DD) of the last import.
            - "next_review": ISO date string (YYYY-MM-DD) of the next review.
    """
    return {
        "version": SDK_VERSION,
        "last_import": LAST_IMPORT_DATE,
        "next_review": NEXT_REVIEW_DATE,
    }


# Type stubs for SDK compatibility
# These are re-exported from the actual SDK when available

try:
    from claude_agent_sdk import query as sdk_query
    from claude_agent_sdk.types import (
        Message,
        UserMessage,
        AssistantMessage,
        ResultMessage,
        SystemMessage,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    sdk_query = None
    Message = None
    UserMessage = None
    AssistantMessage = None
    ResultMessage = None
    SystemMessage = None


__all__ = [
    "SDK_VERSION",
    "LAST_IMPORT_DATE",
    "NEXT_REVIEW_DATE",
    "SDK_AVAILABLE",
    "get_sdk_info",
    "sdk_query",
    "Message",
    "UserMessage",
    "AssistantMessage",
    "ResultMessage",
    "SystemMessage",
]