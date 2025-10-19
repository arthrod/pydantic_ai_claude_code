"""Structured output example using Claude Code with Pydantic AI."""

from pydantic import BaseModel
from pydantic_ai import Agent

import pydantic_ai_claude_code  # noqa: F401 - registers the provider


class MathResult(BaseModel):
    """A mathematical calculation result."""

    answer: int
    explanation: str


class CodeAnalysis(BaseModel):
    """Code complexity analysis."""

    complexity_score: int  # 1-10
    is_recursive: bool
    description: str


def main() -> None:
    """Run structured output examples."""
    # Example 1: Math result
    print("Example 1: Structured Math Result")
    print("-" * 50)
    math_agent = Agent("claude-code:sonnet", output_type=MathResult)

    result = math_agent.run_sync("Calculate 15 * 23")
    print(f"Answer: {result.output.answer}")
    print(f"Explanation: {result.output.explanation}\n")

    # Example 2: Code analysis
    print("Example 2: Structured Code Analysis")
    print("-" * 50)
    code_agent = Agent("claude-code:sonnet", output_type=CodeAnalysis)

    result2 = code_agent.run_sync(
        "Analyze: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    )
    print(f"Complexity Score: {result2.output.complexity_score}/10")
    print(f"Is Recursive: {result2.output.is_recursive}")
    print(f"Description: {result2.output.description}\n")

    # Example 3: Usage tracking with structured output
    print("Example 3: Usage Tracking")
    print("-" * 50)
    usage = result2.usage()
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")


if __name__ == "__main__":
    main()
