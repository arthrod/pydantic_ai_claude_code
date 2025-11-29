"""Example generating a law school landing page.

This example demonstrates using pydantic_ai_claude_code to generate
a professional landing page for a law school.

Note: The Claude CLI only works with Anthropic's models. For third-party
providers like Kimi, you would need to use pydantic-ai with their native
model support (e.g., OpenAI-compatible models).
"""

import asyncio
import logging

from pydantic_ai import Agent

from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider

# Enable info logging to see progress
logging.basicConfig(level=logging.INFO)
logging.getLogger("pydantic_ai_claude_code").setLevel(logging.INFO)


async def main() -> None:
    """Generate a law school landing page using Claude."""

    # Create provider with custom settings
    provider = ClaudeCodeProvider(
        settings={
            "timeout_seconds": 300,  # 5 minutes for longer generation
            "verbose": False,
            "use_sandbox_runtime": False,  # Use main environment credentials
        }
    )

    # Use Claude Sonnet model (the CLI supports: sonnet, opus, haiku)
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

    print("Generating Law School Landing Page with Claude...")
    print("=" * 60)

    # Run the agent asynchronously
    result = await agent.run(prompt)

    print("\nGenerated Landing Page:")
    print("-" * 60)
    print(result.output)

    # Save to examples directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "law_school_landing.html")
    with open(output_path, "w") as f:
        f.write(str(result.output))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
