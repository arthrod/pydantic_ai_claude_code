"""Integration tests for long response handling with actual Claude CLI calls.

These tests actually execute Claude CLI and verify the gradual file building
strategy works in practice. They may be slower than unit tests.
"""

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

# Register the claude-code provider
import pydantic_ai_claude_code  # noqa: F401

# Test constants for response validation
MIN_LONG_RESPONSE_CHARS = 500
MIN_LARGE_ARRAY_ITEMS = 20
MIN_SUMMARY_CHARS = 150
MIN_KEY_POINTS = 6
MIN_ANALYSIS_CHARS = 300
MIN_MULTI_ARRAY_ITEMS = 8
MIN_RECOMMENDATIONS = 6
MIN_EXECUTIVE_SUMMARY_CHARS = 200
MIN_FINDINGS = 10
MIN_METHODOLOGY_CHARS = 150


class LargeReport(BaseModel):
    """Report with potentially large content."""

    title: str
    summary: str
    key_points: list[str]  # Should be able to handle many items
    detailed_analysis: str  # Should be able to handle long text


class ManyItems(BaseModel):
    """Model with a large array."""

    items: list[str]  # Test with 20+ items


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow test
async def test_long_unstructured_response():
    """Test that long unstructured responses work with gradual building.

    This tests the strategy where Claude:
    1. Creates file with Write tool
    2. Appends content gradually with bash echo >>
    3. Builds response incrementally
    """
    agent = Agent("claude-code:haiku")  # Use haiku for speed

    result = await agent.run(
        "Write a comprehensive explanation of how HTTP works. "
        "Include: request methods, response codes, headers, body, "
        "connection lifecycle, and common patterns. "
        "Make it detailed - aim for at least 500 words."
    )

    # Verify we got a substantial response
    assert len(result.output) > MIN_LONG_RESPONSE_CHARS, f"Expected >{MIN_LONG_RESPONSE_CHARS} chars, got {len(result.output)}"

    # Check it contains expected content
    assert "http" in result.output.lower()
    print(f"\n✓ Generated {len(result.output)} character response")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_structured_output_with_large_array():
    """Test structured output with a large array (20+ items).

    This tests the directory structure strategy where Claude:
    1. Creates temp directory
    2. Creates field_name/ directory for arrays
    3. Creates numbered files (0000.txt, 0001.txt, ...)
    4. System assembles JSON from directory structure
    """
    agent = Agent("claude-code:haiku", output_type=ManyItems)

    result = await agent.run(
        "List 25 different programming languages. Just the names, one per item."
    )

    # Verify we got many items
    assert len(result.output.items) >= MIN_LARGE_ARRAY_ITEMS, (
        f"Expected ≥{MIN_LARGE_ARRAY_ITEMS} items, got {len(result.output.items)}"
    )
    assert all(isinstance(item, str) for item in result.output.items)

    print(f"\n✓ Generated array with {len(result.output.items)} items:")
    print(f"  First 5: {result.output.items[:5]}")
    print(f"  Last 5: {result.output.items[-5:]}")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_structured_output_with_large_text_fields():
    """Test structured output with large text fields.

    This verifies that Claude can build up long strings gradually
    in the field files using bash append operations.
    """
    agent = Agent("claude-code:haiku", output_type=LargeReport)

    result = await agent.run(
        "Create a detailed report about cloud computing. "
        "Include:\n"
        "- Title\n"
        "- A comprehensive summary (200+ words)\n"
        "- At least 8 key points\n"
        "- A detailed analysis section (400+ words)\n"
        "Make it thorough and informative."
    )

    report = result.output

    # Verify structure
    assert report.title, "Title should not be empty"
    assert len(report.summary) > MIN_SUMMARY_CHARS, f"Summary too short: {len(report.summary)} chars"
    assert len(report.key_points) >= MIN_KEY_POINTS, (
        f"Expected ≥{MIN_KEY_POINTS} key points, got {len(report.key_points)}"
    )
    assert len(report.detailed_analysis) > MIN_ANALYSIS_CHARS, (
        f"Analysis too short: {len(report.detailed_analysis)} chars"
    )

    print("\n✓ Generated structured report:")
    print(f"  Title: {report.title[:50]}...")
    print(f"  Summary: {len(report.summary)} chars")
    print(f"  Key points: {len(report.key_points)} items")
    print(f"  Analysis: {len(report.detailed_analysis)} chars")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multiple_large_arrays():
    """Test structured output with multiple large arrays.

    This tests the system's ability to handle multiple array fields,
    each with many items.
    """

    class MultiArrayReport(BaseModel):
        title: str
        pros: list[str]  # Large array 1
        cons: list[str]  # Large array 2
        recommendations: list[str]  # Large array 3

    agent = Agent("claude-code:haiku", output_type=MultiArrayReport)

    result = await agent.run(
        "Analyze remote work. Provide:\n"
        "- A title\n"
        "- At least 10 pros\n"
        "- At least 10 cons\n"
        "- At least 8 recommendations"
    )

    report = result.output

    assert report.title
    assert len(report.pros) >= MIN_MULTI_ARRAY_ITEMS, f"Expected ≥{MIN_MULTI_ARRAY_ITEMS} pros, got {len(report.pros)}"
    assert len(report.cons) >= MIN_MULTI_ARRAY_ITEMS, f"Expected ≥{MIN_MULTI_ARRAY_ITEMS} cons, got {len(report.cons)}"
    assert len(report.recommendations) >= MIN_RECOMMENDATIONS, (
        f"Expected ≥{MIN_RECOMMENDATIONS} recs, got {len(report.recommendations)}"
    )

    print("\n✓ Generated multi-array report:")
    print(f"  Pros: {len(report.pros)} items")
    print(f"  Cons: {len(report.cons)} items")
    print(f"  Recommendations: {len(report.recommendations)} items")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_combined_large_content():
    """Test combining large arrays with large text fields.

    This is the most comprehensive test - verifies that the system
    can handle complex structured output with both large arrays
    and large text fields simultaneously.
    """

    class ComprehensiveReport(BaseModel):
        executive_summary: str  # Large text
        findings: list[str]  # Large array
        methodology: str  # Large text

    agent = Agent("claude-code:haiku", output_type=ComprehensiveReport)

    result = await agent.run(
        "Create a research report on artificial intelligence. Include:\n"
        "- Executive summary (250+ words)\n"
        "- At least 12 detailed findings\n"
        "- Methodology section (200+ words)"
    )

    report = result.output

    assert len(report.executive_summary) > MIN_EXECUTIVE_SUMMARY_CHARS
    assert len(report.findings) >= MIN_FINDINGS
    assert len(report.methodology) > MIN_METHODOLOGY_CHARS

    total_content = (
        len(report.executive_summary)
        + sum(len(f) for f in report.findings)
        + len(report.methodology)
    )

    print("\n✓ Generated comprehensive report:")
    print(f"  Executive summary: {len(report.executive_summary)} chars")
    print(
        f"  Findings: {len(report.findings)} items, {sum(len(f) for f in report.findings)} total chars"
    )
    print(f"  Methodology: {len(report.methodology)} chars")
    print(f"  Total content: {total_content} chars")


if __name__ == "__main__":
    """Run integration tests manually."""
    import asyncio

    async def run_all():
        print("Running long response integration tests...\n")
        print("=" * 70)

        try:
            print("\n1. Testing long unstructured response...")
            await test_long_unstructured_response()

            print("\n2. Testing large array...")
            await test_structured_output_with_large_array()

            print("\n3. Testing large text fields...")
            await test_structured_output_with_large_text_fields()

            print("\n4. Testing multiple large arrays...")
            await test_multiple_large_arrays()

            print("\n5. Testing combined large content...")
            await test_combined_large_content()

            print("\n" + "=" * 70)
            print("✓ All integration tests passed!")
            print("=" * 70)

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            raise

    asyncio.run(run_all())
