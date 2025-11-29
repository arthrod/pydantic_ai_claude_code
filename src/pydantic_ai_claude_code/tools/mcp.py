"""MCP (Model Context Protocol) tool for pydantic_ai.

This provides a pre-built tool class for integrating MCP servers with
Claude Code, similar to how TavilyTool works in pydantic_ai.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPTool:
    """Pre-built tool for MCP server integration.

    This tool allows you to connect MCP servers to your Claude Code agent,
    enabling access to external data sources and capabilities.

    Examples:
        >>> from pydantic_ai_claude_code.tools import MCPTool
        >>>
        >>> # Create an MCP tool for web search
        >>> search_tool = MCPTool(
        ...     server_name="brave-search",
        ...     command="brave-search-mcp",
        ...     args=["--api-key", "your-api-key"],
        ... )
        >>>
        >>> # Use with an agent
        >>> agent = Agent(
        ...     model='claude-code:sonnet',
        ...     tools=[search_tool],
        ... )

        >>> # Create a filesystem MCP server
        >>> fs_tool = MCPTool(
        ...     server_name="filesystem",
        ...     command="npx",
        ...     args=["-y", "@anthropics/mcp-server-filesystem"],
        ...     env={"HOME": "/tmp/sandbox"},
        ... )
    """

    server_name: str
    """Name of the MCP server (used as tool name)."""

    command: str
    """Command to run the MCP server."""

    args: list[str] = field(default_factory=list)
    """Arguments to pass to the MCP server command."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables for the MCP server process."""

    tool_name: str | None = None
    """Optional custom tool name (defaults to server_name)."""

    description: str | None = None
    """Optional description of what this MCP server provides."""

    @property
    def name(self) -> str:
        """Get the tool name."""
        return self.tool_name or self.server_name

    def to_mcp_config(self) -> dict[str, Any]:
        """Convert to MCP server configuration format.

        Returns:
            Configuration dictionary for the Claude Code CLI.
        """
        config: dict[str, Any] = {
            "command": self.command,
        }
        if self.args:
            config["args"] = self.args
        if self.env:
            config["env"] = self.env
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with server name and config.
        """
        return {
            self.server_name: self.to_mcp_config()
        }


def create_mcp_tools_config(tools: list[MCPTool]) -> dict[str, Any]:
    """Create combined MCP configuration from multiple tools.

    Args:
        tools: List of MCPTool instances.

    Returns:
        Combined configuration dictionary.
    """
    config: dict[str, Any] = {}
    for tool in tools:
        config.update(tool.to_dict())
    return config
