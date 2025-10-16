#!/usr/bin/env python3
"""Debug script to test profile folder creation and inspect filesystem."""

import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

import pydantic_ai_claude_code  # noqa: F401 - Register provider

# Enable logging
logging.basicConfig(level=logging.INFO)

class Address(BaseModel):
    age: int
    city: str

def create_user(username: str, email: str, profile: Address) -> str:
    """Create a new user with username, email, and profile containing age and city."""
    return f"User created: {username} ({email}) - Profile: age={profile.age}, city={profile.city}"

async def main():
    agent = Agent(
        "claude-code:sonnet",
        system_prompt="You are a helpful assistant.",
        tools=[create_user],
    )

    # Run the test
    print("Running test with complex nested parameters...")
    result = await agent.run(
        "Create a user with username john_doe, email john@example.com, profile age=30, city=London"
    )

    print(f"\n{'='*80}")
    print(f"Result: {result.data}")
    print(f"{'='*80}\n")

    # Check all temp directories
    print("Checking for temp directories...")
    tmp_path = Path("/tmp")

    # Look for data structure directories
    data_dirs = list(tmp_path.glob("claude_data_structure_*"))
    print(f"Found {len(data_dirs)} data structure directories:")

    for data_dir in data_dirs:
        print(f"\n{data_dir}:")
        # List all contents recursively
        for item in sorted(data_dir.rglob("*")):
            rel_path = item.relative_to(data_dir)
            if item.is_file():
                content = item.read_text()[:100]  # First 100 chars
                print(f"  FILE: {rel_path} -> {content!r}")
            else:
                print(f"  DIR:  {rel_path}/")

if __name__ == "__main__":
    asyncio.run(main())
