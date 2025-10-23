"""Example demonstrating sandbox-runtime integration.

This example shows how to run Claude Code CLI in Anthropic's sandbox-runtime for:
- Fully autonomous execution (no permission prompts)
- OS-level security isolation
- Protection against prompt injection and malicious code

Requires:
- Claude Code CLI installed and authenticated
- Sandbox-runtime installed: npm install -g @anthropic-ai/sandbox-runtime
"""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_claude_code import ClaudeCodeProvider


def basic_sandbox_example():
    """Basic example: Enable sandbox mode for autonomous execution."""
    print("\n=== Basic Sandbox Mode ===")
    print("Running Claude in sandbox with IS_SANDBOX=1 for autonomous execution\n")

    provider = ClaudeCodeProvider({
        "model": "sonnet",
        "use_sandbox_runtime": True,  # Enable sandbox wrapping
    })

    agent = Agent("claude-code:sonnet", provider=provider)

    # Claude runs autonomously inside the sandbox
    result = agent.run_sync("What is 2+2? Write a Python script to calculate it.")

    print(f"Result: {result.data}\n")


async def async_sandbox_example():
    """Async example: Long-running batch processing in sandbox."""
    print("\n=== Async Sandbox Mode ===")
    print("Processing multiple tasks autonomously in sandbox\n")

    provider = ClaudeCodeProvider({
        "model": "sonnet",
        "use_sandbox_runtime": True,
        "timeout_seconds": 1800,  # 30 minutes for batch processing
    })

    agent = Agent("claude-code:sonnet", provider=provider)

    tasks = [
        "Calculate the factorial of 10",
        "Generate the first 10 Fibonacci numbers",
        "Check if 17 is prime",
    ]

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] Processing: {task}")
        result = await agent.run(task)
        results.append(result.data)
        print(f"    Result: {result.data}")

    print(f"\nCompleted {len(results)} tasks autonomously in sandbox\n")


def custom_sandbox_path_example():
    """Example: Use custom sandbox-runtime binary path."""
    print("\n=== Custom Sandbox Path ===")
    print("Using custom srt binary location\n")

    provider = ClaudeCodeProvider({
        "model": "sonnet",
        "use_sandbox_runtime": True,
        "sandbox_runtime_path": "/usr/local/bin/srt",  # Custom path
    })

    agent = Agent("claude-code:sonnet", provider=provider)

    result = agent.run_sync("Echo 'Hello from custom sandbox'")
    print(f"Result: {result.data}\n")


def production_use_case():
    """Real-world example: Production agent with sandbox isolation."""
    print("\n=== Production Use Case ===")
    print("Running untrusted code analysis in isolated sandbox\n")

    provider = ClaudeCodeProvider({
        "model": "sonnet",
        "use_sandbox_runtime": True,
        "working_directory": "/tmp/code_analysis",
        "timeout_seconds": 3600,  # 1 hour
    })

    agent = Agent("claude-code:sonnet", provider=provider)

    # Analyze potentially untrusted code safely
    code_to_analyze = """
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
"""

    prompt = f"""Analyze this Python code for potential issues:

{code_to_analyze}

Check for:
1. Security vulnerabilities
2. Performance problems
3. Code quality issues
4. Suggested improvements

Provide a detailed report.
"""

    result = agent.run_sync(prompt)
    print(f"Analysis:\n{result.data}\n")


def comparison_example():
    """Compare standard mode vs sandbox mode."""
    print("\n=== Standard vs Sandbox Comparison ===\n")

    # Standard mode: requires permission prompts (non-interactive)
    print("1. Standard mode (--dangerously-skip-permissions):")
    standard_provider = ClaudeCodeProvider({
        "model": "sonnet",
        "use_sandbox_runtime": False,  # No sandbox
        "dangerously_skip_permissions": True,  # Skip prompts but no isolation
    })
    standard_agent = Agent("claude-code:sonnet", provider=standard_provider)
    result1 = standard_agent.run_sync("Echo 'Standard mode'")
    print(f"   Result: {result1.data}\n")

    # Sandbox mode: autonomous execution WITH OS-level isolation
    print("2. Sandbox mode (IS_SANDBOX=1 + srt isolation):")
    sandbox_provider = ClaudeCodeProvider({
        "model": "sonnet",
        "use_sandbox_runtime": True,  # Sandbox with isolation
    })
    sandbox_agent = Agent("claude-code:sonnet", provider=sandbox_provider)
    result2 = sandbox_agent.run_sync("Echo 'Sandbox mode'")
    print(f"   Result: {result2.data}\n")

    print("Key differences:")
    print("- Standard: No isolation, full system access")
    print("- Sandbox: OS-level isolation, contained environment")
    print("- Both: No permission prompts, autonomous execution")


if __name__ == "__main__":
    print("=" * 70)
    print("Sandbox-Runtime Integration Examples")
    print("=" * 70)

    # Check if srt is available
    import shutil
    if not shutil.which("srt"):
        print("\nERROR: sandbox-runtime (srt) not found!")
        print("Install with: npm install -g @anthropic-ai/sandbox-runtime")
        print("Or set SANDBOX_RUNTIME_PATH environment variable")
        exit(1)

    # Run examples
    basic_sandbox_example()
    comparison_example()
    production_use_case()
    custom_sandbox_path_example()

    # Async example
    print("Running async example...")
    asyncio.run(async_sandbox_example())

    print("\n" + "=" * 70)
    print("All sandbox examples completed successfully!")
    print("=" * 70)