"""Debug script for multiple tools test failure."""
import logging

# Enable ALL logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

import pydantic_ai_claude_code  # Register the provider
from pydantic_ai import Agent

# Track all tool calls
tool_calls = []

def add(a: int, b: int) -> int:
    """Add two numbers."""
    result = a + b
    call_info = f"add({a}, {b}) = {result}"
    tool_calls.append(call_info)
    print(f"\n>>> TOOL CALL: {call_info}\n")
    return result

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    result = a * b
    call_info = f"multiply({a}, {b}) = {result}"
    tool_calls.append(call_info)
    print(f"\n>>> TOOL CALL: {call_info}\n")
    return result

print("Creating agent...")
agent = Agent("claude-code:sonnet", tools=[add, multiply])

print("\nRunning query: 'What is 5 + 3, and what is 4 * 6?'")
result = agent.run_sync("What is 5 + 3, and what is 4 * 6?")

print("\n" + "="*80)
print("TOOL CALLS MADE:")
for call in tool_calls:
    print(f"  - {call}")

print("\nFINAL OUTPUT:")
print(result.output)

print("\n" + "="*80)
print("EXPECTED:")
print("  - 5 + 3 = 8")
print("  - 4 * 6 = 24")

print("\nVERIFICATION:")
print(f"  '8' in output: {'8' in result.output}")
print(f"  '24' in output: {'24' in result.output}")
