"""Example demonstrating tool calling and streaming with Claude Code."""

import asyncio

import pydantic_ai_claude_code  # Register the provider

from pydantic_ai import Agent, RunContext


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather description
    """
    # Simulated weather data
    weather_data = {
        "London": "Rainy, 15°C",
        "Paris": "Sunny, 22°C",
        "New York": "Cloudy, 18°C",
        "Tokyo": "Clear, 25°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result is: {result}"
        else:
            return "Invalid expression - only basic math operations allowed"
    except Exception as e:
        return f"Error calculating: {e}"


async def main():
    """Run examples demonstrating tools and streaming."""
    # Example 1: Agent with custom tools
    print("Example 1: Custom Tool Calling")
    print("=" * 60)

    agent = Agent(
        'claude-code:sonnet',
        tools=[get_weather, calculate],
        system_prompt="You are a helpful assistant with access to weather and calculator tools.",
    )

    result = await agent.run(
        "What's the weather like in Paris? Also, calculate 15 * 23 for me."
    )
    print(f"Response: {result.output}")
    print(f"\nUsage: {result.usage()}")

    # Example 2: Streaming response
    print("\n\nExample 2: Streaming Text Response")
    print("=" * 60)

    stream_agent = Agent('claude-code:sonnet')

    print("Streaming response:\n")
    async with stream_agent.run_stream(
        "Write a short haiku about programming"
    ) as result:
        async for text in result.stream_text():
            print(text, end="", flush=True)

    print(f"\n\nFinal usage: {result.usage()}")

    # Example 3: Tool calling with context
    print("\n\nExample 3: Tools with Context")
    print("=" * 60)

    def get_user_location(ctx: RunContext[str]) -> str:
        """Get the user's location from context.

        Args:
            ctx: Run context containing user location

        Returns:
            User's current location
        """
        return ctx.deps

    context_agent = Agent(
        'claude-code:sonnet',
        deps_type=str,
        tools=[get_user_location, get_weather],
        system_prompt="Help users with weather information for their location.",
    )

    result = await context_agent.run(
        "What's the weather like where I am?",
        deps="London"
    )
    print(f"Response: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
