"""Tests for MCPTool pre-built class."""

import pytest

from pydantic_ai_claude_code.tools import MCPTool
from pydantic_ai_claude_code.tools.mcp import create_mcp_tools_config


class TestMCPTool:
    """Test MCPTool class."""

    def test_basic_initialization(self):
        """Test basic MCPTool initialization."""
        tool = MCPTool(
            server_name="brave-search",
            command="brave-search-mcp",
        )
        assert tool.server_name == "brave-search"
        assert tool.command == "brave-search-mcp"
        assert tool.args == []
        assert tool.env == {}
        assert tool.name == "brave-search"

    def test_initialization_with_args(self):
        """Test MCPTool with command arguments."""
        tool = MCPTool(
            server_name="filesystem",
            command="npx",
            args=["-y", "@anthropics/mcp-server-filesystem"],
        )
        assert tool.args == ["-y", "@anthropics/mcp-server-filesystem"]

    def test_initialization_with_env(self):
        """Test MCPTool with environment variables."""
        tool = MCPTool(
            server_name="github",
            command="github-mcp",
            env={"GITHUB_TOKEN": "xxx"},
        )
        assert tool.env == {"GITHUB_TOKEN": "xxx"}

    def test_custom_tool_name(self):
        """Test MCPTool with custom tool name."""
        tool = MCPTool(
            server_name="brave-search",
            command="brave-search-mcp",
            tool_name="web_search",
        )
        assert tool.tool_name == "web_search"
        assert tool.name == "web_search"

    def test_name_property_uses_server_name(self):
        """Test that name property defaults to server_name."""
        tool = MCPTool(
            server_name="my-server",
            command="my-command",
        )
        assert tool.name == "my-server"

    def test_description(self):
        """Test MCPTool with description."""
        tool = MCPTool(
            server_name="database",
            command="db-mcp",
            description="Access database queries",
        )
        assert tool.description == "Access database queries"

    def test_to_mcp_config_minimal(self):
        """Test to_mcp_config with minimal configuration."""
        tool = MCPTool(
            server_name="test",
            command="test-mcp",
        )
        config = tool.to_mcp_config()
        assert config == {"command": "test-mcp"}

    def test_to_mcp_config_with_args(self):
        """Test to_mcp_config with arguments."""
        tool = MCPTool(
            server_name="test",
            command="test-mcp",
            args=["--port", "8080"],
        )
        config = tool.to_mcp_config()
        assert config["command"] == "test-mcp"
        assert config["args"] == ["--port", "8080"]

    def test_to_mcp_config_with_env(self):
        """Test to_mcp_config with environment variables."""
        tool = MCPTool(
            server_name="test",
            command="test-mcp",
            env={"API_KEY": "secret"},
        )
        config = tool.to_mcp_config()
        assert config["command"] == "test-mcp"
        assert config["env"] == {"API_KEY": "secret"}

    def test_to_mcp_config_full(self):
        """Test to_mcp_config with all options."""
        tool = MCPTool(
            server_name="full",
            command="full-mcp",
            args=["--verbose"],
            env={"DEBUG": "true"},
        )
        config = tool.to_mcp_config()
        assert config == {
            "command": "full-mcp",
            "args": ["--verbose"],
            "env": {"DEBUG": "true"},
        }

    def test_to_dict(self):
        """Test to_dict method."""
        tool = MCPTool(
            server_name="my-server",
            command="my-command",
            args=["--flag"],
        )
        result = tool.to_dict()
        assert "my-server" in result
        assert result["my-server"]["command"] == "my-command"
        assert result["my-server"]["args"] == ["--flag"]


class TestCreateMCPToolsConfig:
    """Test create_mcp_tools_config function."""

    def test_empty_list(self):
        """Test with empty list."""
        config = create_mcp_tools_config([])
        assert config == {}

    def test_single_tool(self):
        """Test with single tool."""
        tool = MCPTool(server_name="search", command="search-mcp")
        config = create_mcp_tools_config([tool])
        assert "search" in config
        assert config["search"]["command"] == "search-mcp"

    def test_multiple_tools(self):
        """Test with multiple tools."""
        tools = [
            MCPTool(server_name="search", command="search-mcp"),
            MCPTool(server_name="filesystem", command="fs-mcp", args=["--root", "/"]),
            MCPTool(
                server_name="database",
                command="db-mcp",
                env={"DB_URL": "localhost"},
            ),
        ]
        config = create_mcp_tools_config(tools)

        assert len(config) == 3
        assert "search" in config
        assert "filesystem" in config
        assert "database" in config

        assert config["search"]["command"] == "search-mcp"
        assert config["filesystem"]["args"] == ["--root", "/"]
        assert config["database"]["env"]["DB_URL"] == "localhost"


class TestMCPToolExamples:
    """Test example use cases for MCPTool."""

    def test_brave_search_example(self):
        """Test creating Brave Search MCP tool."""
        search_tool = MCPTool(
            server_name="brave-search",
            command="brave-search-mcp",
            args=["--api-key", "your-api-key"],
        )
        assert search_tool.name == "brave-search"
        config = search_tool.to_mcp_config()
        assert "api-key" in " ".join(config["args"])

    def test_filesystem_example(self):
        """Test creating filesystem MCP server."""
        fs_tool = MCPTool(
            server_name="filesystem",
            command="npx",
            args=["-y", "@anthropics/mcp-server-filesystem"],
            env={"HOME": "/tmp/sandbox"},
        )
        assert fs_tool.name == "filesystem"
        config = fs_tool.to_mcp_config()
        assert config["env"]["HOME"] == "/tmp/sandbox"

    def test_multiple_servers_config(self):
        """Test creating config for multiple MCP servers."""
        tools = [
            MCPTool(
                server_name="search",
                command="brave-search-mcp",
                env={"API_KEY": "search-key"},
            ),
            MCPTool(
                server_name="github",
                command="github-mcp",
                env={"GITHUB_TOKEN": "gh-token"},
            ),
        ]

        config = create_mcp_tools_config(tools)

        # Can be used as MCP servers configuration
        assert len(config) == 2
        assert config["search"]["env"]["API_KEY"] == "search-key"
        assert config["github"]["env"]["GITHUB_TOKEN"] == "gh-token"


class TestMCPToolImports:
    """Test that MCPTool can be imported correctly."""

    def test_import_from_tools(self):
        """Test importing from tools module."""
        from pydantic_ai_claude_code.tools import MCPTool

        tool = MCPTool(server_name="test", command="test")
        assert tool.name == "test"

    def test_import_from_main_package(self):
        """Test importing from main package."""
        from pydantic_ai_claude_code import MCPTool

        tool = MCPTool(server_name="test", command="test")
        assert tool.name == "test"
