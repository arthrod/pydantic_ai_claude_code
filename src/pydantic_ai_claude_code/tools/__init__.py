"""Pre-built tools for pydantic_ai_claude_code.

These are ready-to-use tool classes similar to TavilyTool in pydantic_ai.
They can be added to an Agent's tools list for specialized functionality.
"""

from .mcp import MCPTool

__all__ = ["MCPTool"]
