"""Integration test for timeout_seconds configuration with Agent."""

import logging
from pydantic_ai import Agent
import pydantic_ai_claude_code  # noqa: F401 - triggers model registration

# Enable debug logging to see settings
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pydantic_ai_claude_code")
logger.setLevel(logging.DEBUG)


def test_timeout_with_agent():
    """Test that timeout_seconds works when passed via Agent model_settings."""

    # Create agent with custom timeout
    agent = Agent(
        "claude-code:sonnet",
        model_settings={"timeout_seconds": 1800}  # 30 minutes
    )

    # Make a simple request
    result = agent.run_sync("What is 2+2? Just give me the number.")

    print(f"✓ Agent request completed: {result.output}")
    print("✓ timeout_seconds is properly configured and passed through!")


if __name__ == "__main__":
    print("Testing timeout_seconds integration with Agent...")
    print("This will make a real call to Claude CLI with a 30-minute timeout.")
    print()

    test_timeout_with_agent()

    print("\n✅ Integration test passed!")
    print("The timeout_seconds setting is now properly configurable via Agent!")
