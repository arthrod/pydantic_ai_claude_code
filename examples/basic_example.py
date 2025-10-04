"""Basic example of using Claude Code with Pydantic AI."""

import pydantic_ai_claude_code  # Register the provider

from pydantic_ai import Agent


def main():
    """Run basic examples."""
    # Example 1: Simple text query (using string format - simplest!)
    print("Example 1: Simple Query")
    print("-" * 50)
    agent = Agent('claude-code:sonnet')

    result = agent.run_sync("What is the capital of France? Just name the city.")
    print(f"Response: {result.output}\n")

    # Example 2: Math calculation
    print("Example 2: Math Calculation")
    print("-" * 50)

    result = agent.run_sync("What is 15 * 23? Just give me the number.")
    print(f"Response: {result.output}\n")

    # Example 3: Code explanation
    print("Example 3: Code Explanation")
    print("-" * 50)

    result = agent.run_sync("Explain what this Python code does in one sentence: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)")
    print(f"Response: {result.output}\n")

    # Example 4: Usage tracking
    print("Example 4: Usage Tracking")
    print("-" * 50)

    result = agent.run_sync("Write a haiku about Python programming")
    print(f"Response: {result.output}")

    usage = result.usage()
    print(f"\nUsage Stats:")
    print(f"  Input tokens: {usage.input_tokens}")
    print(f"  Output tokens: {usage.output_tokens}")
    print(f"  Total tokens: {usage.total_tokens}")
    if usage.details:
        print(f"  Cost (cents): {usage.details.get('total_cost_usd_cents', 0)}")
        print(f"  Duration: {usage.details.get('duration_ms', 0)}ms")


if __name__ == "__main__":
    main()
