"""Example demonstrating handling of long responses with Claude Code.

This example shows how the package handles responses that would exceed
typical output length limits by instructing Claude to build content gradually.

Key Features:
1. Unstructured responses: Claude builds text gradually using file appending
2. Structured responses: Claude creates a directory structure mirroring JSON
3. Streaming: Real-time progress for long responses
4. max_output_tokens: Control response length if needed
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

import pydantic_ai_claude_code  # noqa: F401 - Register the provider
from pydantic_ai_claude_code import ClaudeCodeProvider

# Constants for output display
MAX_KEY_POINTS_TO_DISPLAY = 5  # Number of key points to show before truncation


class DetailedAnalysis(BaseModel):
    """Structured output for detailed analysis with potentially large content."""

    title: str
    summary: str
    key_points: list[str]
    detailed_findings: str
    recommendations: list[str]
    conclusion: str


async def example_1_unstructured_long_response():
    """Example 1: Generate a long unstructured response.

    Behind the scenes:
    - Claude creates /tmp/claude_unstructured_output_*.txt
    - Uses Write tool for initial content
    - Uses bash 'echo >>' to append content gradually
    - Builds response incrementally without hitting limits
    """
    print("\n" + "=" * 70)
    print("Example 1: Long Unstructured Response")
    print("=" * 70)

    agent = Agent("claude-code:sonnet")

    result = await agent.run(
        "Write a comprehensive guide to machine learning that covers:\n"
        "1. Introduction to ML (what it is, history, applications)\n"
        "2. Supervised learning (concepts, algorithms, examples)\n"
        "3. Unsupervised learning (clustering, dimensionality reduction)\n"
        "4. Neural networks and deep learning\n"
        "5. Best practices and common pitfalls\n"
        "6. Real-world case studies\n"
        "Make it detailed and educational - aim for at least 2000 words."
    )

    print(f"\nGenerated {len(result.output)} characters")
    print(f"\nFirst 500 characters:\n{result.output[:500]}...")
    print(f"\nLast 200 characters:\n...{result.output[-200:]}")
    print(f"\nUsage: {result.usage()}")


async def example_2_structured_with_large_data():
    """Example 2: Structured output with large amounts of data.

    Behind the scenes:
    - Claude creates /tmp/claude_json_fields_*/
    - For each field:
      * Strings: field_name.txt (built with echo >>)
      * Arrays: field_name/0000.txt, 0001.txt, ... (one file per item)
    - Creates .complete marker when done
    - System automatically assembles valid JSON from directory structure
    """
    print("\n" + "=" * 70)
    print("Example 2: Structured Output with Large Data")
    print("=" * 70)

    agent = Agent("claude-code:sonnet", output_type=DetailedAnalysis)

    result = await agent.run(
        "Conduct a comprehensive analysis of cloud computing trends in 2025. "
        "Include:\n"
        "- At least 10 key points about current trends\n"
        "- Detailed findings (aim for 1000+ words)\n"
        "- At least 8 specific recommendations\n"
        "Make it thorough and well-researched."
    )

    analysis = result.output

    print(f"\nTitle: {analysis.title}")
    print(f"Summary: {analysis.summary[:200]}...")
    print(f"\nKey Points ({len(analysis.key_points)} total):")
    for i, point in enumerate(analysis.key_points[:MAX_KEY_POINTS_TO_DISPLAY], 1):
        print(f"  {i}. {point[:80]}...")
    if len(analysis.key_points) > MAX_KEY_POINTS_TO_DISPLAY:
        print(f"  ... and {len(analysis.key_points) - MAX_KEY_POINTS_TO_DISPLAY} more")

    print(f"\nDetailed Findings: {len(analysis.detailed_findings)} characters")
    print(f"First 300 chars: {analysis.detailed_findings[:300]}...")

    print(f"\nRecommendations ({len(analysis.recommendations)} total):")
    for i, rec in enumerate(analysis.recommendations[:3], 1):
        print(f"  {i}. {rec[:80]}...")

    print(f"\nConclusion: {analysis.conclusion[:200]}...")

    print(f"\nUsage: {result.usage()}")


async def example_3_streaming_long_response():
    """Example 3: Stream a long response in real-time.

    This provides better user experience for long-running generations.
    """
    print("\n" + "=" * 70)
    print("Example 3: Streaming Long Response")
    print("=" * 70)

    agent = Agent("claude-code:sonnet")

    print("\nGenerating comprehensive tutorial (streaming)...\n")

    async with agent.run_stream(
        "Write a comprehensive tutorial on building REST APIs with Python. "
        "Cover: Flask basics, routing, request handling, database integration, "
        "authentication, error handling, testing, deployment, and best practices. "
        "Make it detailed with code examples."
    ) as result:
        char_count = 0
        async for text in result.stream_text():
            print(text, end="", flush=True)
            char_count += len(text)

    print(f"\n\nTotal characters streamed: {char_count}")
    print(f"Usage: {result.usage()}")


async def example_4_with_max_output_tokens():
    """Example 4: Using max_output_tokens to control response length.

    This is useful when you want to limit the length of responses.
    Note: Claude CLI may not support this flag yet, but it's future-proofed.
    """
    print("\n" + "=" * 70)
    print("Example 4: Controlled Response Length with max_output_tokens")
    print("=" * 70)

    # Create provider with token limit
    provider = ClaudeCodeProvider({"max_output_tokens": 500})

    agent = Agent("claude-code:sonnet", provider=provider)

    result = await agent.run(
        "Explain quantum computing in detail with examples and applications."
    )

    print(f"\nResponse length: {len(result.output)} characters")
    print(f"\nResponse:\n{result.output}")
    print(f"\nUsage: {result.usage()}")


async def example_5_multiple_large_arrays():
    """Example 5: Structured output with multiple large arrays.

    Demonstrates how the system handles complex structures with lots of data.
    """
    print("\n" + "=" * 70)
    print("Example 5: Multiple Large Arrays in Structured Output")
    print("=" * 70)

    class ComprehensiveReport(BaseModel):
        title: str
        executive_summary: str
        findings: list[str]  # Large array
        data_points: list[str]  # Another large array
        recommendations: list[str]  # And another
        methodology: str

    agent = Agent("claude-code:sonnet", output_type=ComprehensiveReport)

    result = await agent.run(
        "Create a comprehensive market research report on electric vehicles. "
        "Include:\n"
        "- At least 15 key findings\n"
        "- At least 20 specific data points with numbers\n"
        "- At least 12 actionable recommendations\n"
        "- Detailed methodology section (500+ words)"
    )

    report = result.output

    print(f"\nTitle: {report.title}")
    print(f"Executive Summary: {report.executive_summary[:250]}...")
    print(f"\nFindings: {len(report.findings)} items")
    print(f"Data Points: {len(report.data_points)} items")
    print(f"Recommendations: {len(report.recommendations)} items")
    print(f"Methodology: {len(report.methodology)} characters")

    print("\nSample findings:")
    for i, finding in enumerate(report.findings[:3], 1):
        print(f"  {i}. {finding[:100]}...")

    print(f"\nUsage: {result.usage()}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("LONG RESPONSE HANDLING EXAMPLES")
    print("=" * 70)
    print("""
These examples demonstrate how pydantic-ai-claude-code handles
responses that exceed typical output length limits.

The key innovation: Instead of generating everything at once,
Claude builds responses GRADUALLY using file operations:

For UNSTRUCTURED outputs:
  1. Create file with Write tool
  2. Append content with bash: echo "more" >> file.txt
  3. Build up response piece by piece

For STRUCTURED outputs:
  1. Create directory structure mirroring JSON schema
  2. Each field gets its own file (field_name.txt)
  3. Arrays get directories with numbered files (0000.txt, 0001.txt, ...)
  4. Content built gradually with append operations
  5. System automatically assembles valid JSON from files

This approach allows responses of ANY SIZE without hitting limits!
""")

    try:
        # Run examples
        await example_1_unstructured_long_response()
        await example_2_structured_with_large_data()
        await example_3_streaming_long_response()
        await example_4_with_max_output_tokens()
        await example_5_multiple_large_arrays()

    except Exception as e:
        print(f"\n\nError running examples: {e}")
        print("\nNote: These examples require:")
        print("1. Claude Code CLI installed and authenticated")
        print("2. Sufficient usage limits")
        raise

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. GRADUAL BUILDING: Claude builds content piece-by-piece, not all at once
   - Avoids hitting output token limits
   - Works for responses of any length

2. FILE-BASED APPROACH:
   - Unstructured: Single file, built with append operations
   - Structured: Directory structure matching JSON schema
   - System handles JSON assembly automatically

3. STREAMING AVAILABLE:
   - Use agent.run_stream() for real-time progress
   - Better UX for long-running generations

4. STRUCTURED DATA SCALES:
   - Arrays with 100s of items? No problem!
   - Multiple large fields? Handled naturally!
   - No manual JSON syntax needed!

5. FUTURE-PROOFED:
   - max_output_tokens setting ready for CLI support
   - Fallback strategies for robustness
""")


if __name__ == "__main__":
    asyncio.run(main())
