"""Example demonstrating BinaryContent support with standard Pydantic AI interface.

This shows how to use the standard Pydantic AI BinaryContent interface which
works identically across all model providers. The implementation automatically
writes binary files to the working directory for Claude Code.
"""

from pathlib import Path

from pydantic_ai import Agent, BinaryContent


def main() -> None:
    """Demonstrate BinaryContent with standard Pydantic AI interface."""

    agent = Agent("claude-code:sonnet")

    print("=" * 80)
    print("Example 1: Single image with BinaryContent")
    print("=" * 80)

    # Create a simple test image (1x1 PNG)
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
        b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )

    # Use standard Pydantic AI BinaryContent interface
    result = agent.run_sync(
        [
            'Describe this image and tell me what you see.',
            BinaryContent(data=png_data, media_type='image/png'),
        ]
    )
    print(f"Response: {result.output}\n")

    print("=" * 80)
    print("Example 2: Multiple images for comparison")
    print("=" * 80)

    # Create two test images
    image1_data = png_data  # Reuse the PNG data
    image2_data = png_data  # For demo purposes, using same data

    result = agent.run_sync(
        [
            'Compare these two images:',
            BinaryContent(data=image1_data, media_type='image/png', identifier='first'),
            'versus',
            BinaryContent(data=image2_data, media_type='image/jpeg', identifier='second'),
            '. What are the differences?',
        ]
    )
    print(f"Response: {result.output}\n")

    print("=" * 80)
    print("Example 3: PDF document analysis")
    print("=" * 80)

    # Create a minimal PDF
    pdf_data = b'''%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 24 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000314 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
407
%%EOF'''

    result = agent.run_sync(
        [
            'What does this PDF document say?',
            BinaryContent(data=pdf_data, media_type='application/pdf'),
        ]
    )
    print(f"Response: {result.output}\n")

    print("=" * 80)
    print("Example 4: Loading from a real file")
    print("=" * 80)

    # For demonstration, create a temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        f.write("This is a test document with some sample content.\n")
        f.write("It contains multiple lines.\n")
        f.write("You can analyze this content.\n")
        temp_path = Path(f.name)

    try:
        # Read the file and send as BinaryContent
        file_data = temp_path.read_bytes()
        result = agent.run_sync(
            [
                'Summarize this text file:',
                BinaryContent(data=file_data, media_type='text/plain'),
            ]
        )
        print(f"Response: {result.output}\n")
    finally:
        temp_path.unlink()  # Clean up

    print("=" * 80)
    print("Key Points:")
    print("=" * 80)
    print("1. Uses standard Pydantic AI BinaryContent interface")
    print("2. Works identically to cloud providers (OpenAI, Anthropic, etc.)")
    print("3. Files are automatically written to working directory")
    print("4. No Claude Code-specific code needed!")
    print("5. Code is portable between different model providers")


if __name__ == "__main__":
    main()
