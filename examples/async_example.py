"""Async example of using Claude Code with Pydantic AI."""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_claude_code import ClaudeCodeModel


class Summary(BaseModel):
    """A text summary."""

    main_points: list[str]
    word_count: int


async def main() -> None:
    """Run async examples."""
    model = ClaudeCodeModel("sonnet")

    # Example 1: Basic async query
    print("Example 1: Basic Async Query")
    print("-" * 50)
    agent = Agent(model)

    result = await agent.run("Explain what async/await does in Python in one sentence")
    print(f"Response: {result.output}\n")

    # Example 2: Multiple concurrent requests
    print("Example 2: Concurrent Requests")
    print("-" * 50)

    async def ask_question(question: str) -> str:
        agent = Agent(model)
        result = await agent.run(question)
        return str(result.output)

    questions = [
        "What is 10 + 20?",
        "What is the capital of Japan?",
        "What color is the sky?",
    ]

    # Run all questions concurrently
    answers = await asyncio.gather(*[ask_question(q) for q in questions])

    for q, a in zip(questions, answers, strict=False):
        print(f"Q: {q}")
        print(f"A: {a}\n")


if __name__ == "__main__":
    asyncio.run(main())
