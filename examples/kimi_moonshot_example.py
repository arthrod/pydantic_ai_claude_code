"""Example using Moonshot/Kimi API to create a law school landing page.

This example demonstrates using an Anthropic-compatible API (Moonshot)
with pydantic_ai_claude_code. Moonshot provides an Anthropic-compatible
endpoint at https://api.moonshot.cn/anthropic/

Requirements:
    pip install python-dotenv

Usage:
    1. Create a .env file with:
       ANTHROPIC_BASE_URL=https://api.moonshot.cn/anthropic/
       ANTHROPIC_API_KEY=your_moonshot_api_key
    2. Run: python kimi_moonshot_example.py
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Enable info logging to see progress
logging.basicConfig(level=logging.INFO)
logging.getLogger("pydantic_ai_claude_code").setLevel(logging.INFO)


async def main() -> None:
    """Generate a law school landing page using Moonshot/Kimi API."""

    # Verify environment variables are set
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not base_url or not api_key:
        print("Error: Please set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY in .env file")
        print("Example .env contents:")
        print("  ANTHROPIC_BASE_URL=https://api.moonshot.cn/anthropic/")
        print("  ANTHROPIC_API_KEY=your_moonshot_api_key")
        return

    print(f"Using API endpoint: {base_url}")

    # Create provider with custom settings
    provider = ClaudeCodeProvider(
        settings={
            "timeout_seconds": 300,  # 5 minutes for longer generation
            "verbose": False,
            "use_sandbox_runtime": False,  # Use main environment credentials
        }
    )

    # Use Claude Sonnet model - Moonshot's Anthropic-compatible endpoint
    # will handle the model mapping
    model = ClaudeCodeModel("sonnet", provider=provider)

    # Create agent
    agent = Agent(model)

    # Prompt for law school landing page
    prompt = """Create a modern, professional landing page for "Apex Law School" -
a prestigious law school. Include:

1. A compelling hero section with headline and call-to-action
2. Key statistics (bar pass rate, employment rate, alumni network size)
3. Featured programs section (JD, LLM, Joint Degrees)
4. Testimonial from a successful alumnus
5. Application deadline reminder
6. Contact information and social media links

Output clean, semantic HTML with inline CSS styling. Use a professional
color scheme (navy blue, gold accents, white backgrounds). Make it
responsive and accessible."""

    print("Generating Law School Landing Page with Moonshot/Kimi...")
    print("=" * 60)

    # Run the agent asynchronously
    result = await agent.run(prompt)

    print("\nGenerated Landing Page:")
    print("-" * 60)
    print(result.output)

    # Save to examples directory
    script_dir = Path(__file__).parent
    output_path = script_dir / "kimi_law_school_landing.html"
    output_path.write_text(str(result.output))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
