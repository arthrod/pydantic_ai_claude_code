"""Advanced example with custom provider configuration."""

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider


class ProjectAnalysis(BaseModel):
    """Analysis of a project structure."""

    file_count: int
    languages_used: list[str]
    summary: str


def main():
    """Run advanced examples."""
    # Example 1: Custom provider with tool restrictions
    print("Example 1: Custom Provider Configuration")
    print("-" * 50)

    provider = ClaudeCodeProvider(
        allowed_tools=["Read", "Grep", "Glob"],  # Only allow read-only tools
        verbose=False,
        use_temp_workspace=False,  # Override default - no filesystem access needed
    )

    model = ClaudeCodeModel("sonnet", provider=provider)
    agent = Agent(model)

    result = agent.run_sync("List the main components of a typical web application")
    print(f"Response: {result.output}\n")

    # Example 2: Temporary workspace
    print("Example 2: Temporary Workspace")
    print("-" * 50)

    with ClaudeCodeProvider(use_temp_workspace=True) as temp_provider:
        model = ClaudeCodeModel("sonnet", provider=temp_provider)
        agent = Agent(model)

        print(f"Working in: {temp_provider.working_directory}")
        result = agent.run_sync(
            "Create a simple hello.txt file with 'Hello, World!' in it"
        )
        print(f"Response: {result.output}")

    print("Temporary workspace cleaned up!\n")

    # Example 3: Custom system prompt
    print("Example 3: Custom System Prompt")
    print("-" * 50)

    provider = ClaudeCodeProvider(
        append_system_prompt="You are a concise assistant. Always respond in 20 words or less.",
        use_temp_workspace=False,  # Override default - no filesystem access needed
    )

    model = ClaudeCodeModel("sonnet", provider=provider)
    agent = Agent(model)

    result = agent.run_sync("Explain what Python is")
    print(f"Response: {result.output}")
    print(f"Word count: {len(str(result.output).split())}")


if __name__ == "__main__":
    main()
