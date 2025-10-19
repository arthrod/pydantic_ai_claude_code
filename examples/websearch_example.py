"""Example demonstrating web search functionality with Claude Code."""

import asyncio

from pydantic_ai import Agent, WebSearchTool

import pydantic_ai_claude_code  # noqa: F401 - registers the provider


async def main() -> None:
    """Run examples demonstrating WebSearchTool with claude-code:sonnet."""

    # Example 1: Basic web search
    print("Example 1: Basic Web Search")
    print("=" * 60)

    agent = Agent("claude-code:sonnet", builtin_tools=[WebSearchTool()])

    result = await agent.run(
        "What is the current stable version of Python? Search the web to find out."
    )
    print(f"Response:\n{result.output}")
    print(f"\nUsage: {result.usage()}")

    # Example 2: Current events search
    print("\n\nExample 2: Current Events Search")
    print("=" * 60)

    agent = Agent(
        "claude-code:sonnet",
        builtin_tools=[WebSearchTool(search_context_size="high")]
    )

    result = await agent.run(
        "What are the top AI/ML news stories this week? Give me a brief summary."
    )
    print(f"Response:\n{result.output}")
    print(f"\nUsage: {result.usage()}")

    # Example 3: Web search with domain filtering
    print("\n\nExample 3: Web Search with Blocked Domains")
    print("=" * 60)

    agent = Agent(
        "claude-code:sonnet",
        builtin_tools=[
            WebSearchTool(
                blocked_domains=["reddit.com", "twitter.com"],
            )
        ],
    )

    result = await agent.run(
        "What is Pydantic AI? Search for information about it."
    )
    print(f"Response:\n{result.output}")

    # Example 4: Limited searches with max_uses
    print("\n\nExample 4: Web Search with Max Uses Limit")
    print("=" * 60)

    agent = Agent(
        "claude-code:sonnet",
        builtin_tools=[
            WebSearchTool(max_uses=2)  # Limit to 2 search queries
        ],
    )

    result = await agent.run(
        "Compare Python and JavaScript for web development. What are the pros and cons?"
    )
    print(f"Response:\n{result.output}")

    # Example 5: Focused search with allowed domains
    print("\n\nExample 5: Web Search with Allowed Domains Only")
    print("=" * 60)

    agent = Agent(
        "claude-code:sonnet",
        builtin_tools=[
            WebSearchTool(
                allowed_domains=["python.org", "pydantic.dev", "docs.python.org"],
            )
        ],
    )

    result = await agent.run(
        "What are the key features of Pydantic? Search official sources."
    )
    print(f"Response:\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
