"""Example demonstrating DuckDuckGo web search tool with Claude Code.

This uses pydantic-ai's custom DuckDuckGo search tool which creates
tool_result_*.txt files in the working directory.
"""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

import pydantic_ai_claude_code  # noqa: F401 - registers the provider


async def main() -> None:
    """Run examples demonstrating DuckDuckGo search with claude-code:sonnet."""

    # Example 1: Basic DuckDuckGo search
    print("Example 1: DuckDuckGo Web Search")
    print("=" * 60)

    agent = Agent(
        "claude-code:sonnet",
        tools=[duckduckgo_search_tool()],
    )

    result = await agent.run(
        "What is the current stable version of Python? Search the web to find out."
    )
    print(f"Response:\n{result.output}")
    print(f"\nUsage: {result.usage()}")

    # Example 2: Search for recent news
    print("\n\nExample 2: Recent News Search")
    print("=" * 60)

    agent = Agent(
        "claude-code:sonnet",
        tools=[duckduckgo_search_tool()],
    )

    result = await agent.run(
        "What are the latest AI developments this week? Search for recent news."
    )
    print(f"Response:\n{result.output}")
    print(f"\nUsage: {result.usage()}")

    # Example 3: Technical documentation search
    print("\n\nExample 3: Technical Documentation Search")
    print("=" * 60)

    agent = Agent(
        "claude-code:sonnet",
        tools=[duckduckgo_search_tool()],
    )

    result = await agent.run(
        "How do I use Pydantic AI with custom tools? Search for documentation."
    )
    print(f"Response:\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
