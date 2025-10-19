"""Integration tests for BinaryContent with actual agent execution.

Uses real test files from tests/fixtures/ directory.
"""

from pathlib import Path

from pydantic_ai import Agent, BinaryContent

# Import to trigger registration
import pydantic_ai_claude_code  # noqa: F401

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_binary_content_png_image():
    """Test sending a PNG image using BinaryContent."""
    image_path = FIXTURES_DIR / "Bert-WhiteBorder.png"
    assert image_path.exists(), f"Test fixture not found: {image_path}"

    image_data = image_path.read_bytes()

    agent = Agent("claude-code:sonnet")

    # Use standard Pydantic AI BinaryContent interface
    result = agent.run_sync(
        [
            'Describe what you see in this image in one short sentence.',
            BinaryContent(data=image_data, media_type='image/png'),
        ]
    )

    # Verify we got a response
    assert result.output is not None
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    print(f"PNG Image response: {result.output}")


def test_binary_content_jpeg_image():
    """Test sending a JPEG image using BinaryContent."""
    image_path = FIXTURES_DIR / "Bert_and_Ernie.JPG"
    assert image_path.exists(), f"Test fixture not found: {image_path}"

    image_data = image_path.read_bytes()

    agent = Agent("claude-code:sonnet")

    result = agent.run_sync(
        [
            'What do you see in this image? Answer in one short sentence.',
            BinaryContent(data=image_data, media_type='image/jpeg'),
        ]
    )

    # Verify we got a response
    assert result.output is not None
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    print(f"JPEG Image response: {result.output}")


def test_binary_content_pdf_document():
    """Test sending a PDF document using BinaryContent."""
    pdf_path = FIXTURES_DIR / "1810.04805v2.pdf"
    assert pdf_path.exists(), f"Test fixture not found: {pdf_path}"

    pdf_data = pdf_path.read_bytes()

    agent = Agent("claude-code:sonnet")

    result = agent.run_sync(
        [
            'What is the title of this PDF paper? Answer with just the title.',
            BinaryContent(data=pdf_data, media_type='application/pdf'),
        ]
    )

    # Verify we got a response
    assert result.output is not None
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    print(f"PDF response: {result.output}")


def test_binary_content_text_file():
    """Test sending a text file using BinaryContent."""
    text_path = FIXTURES_DIR / "bert_pretraining.rst.txt"
    assert text_path.exists(), f"Test fixture not found: {text_path}"

    text_data = text_path.read_bytes()

    agent = Agent("claude-code:sonnet")

    result = agent.run_sync(
        [
            'Summarize the main topic of this text file in one sentence.',
            BinaryContent(data=text_data, media_type='text/plain'),
        ]
    )

    # Verify we got a response
    assert result.output is not None
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    print(f"Text file response: {result.output}")


def test_binary_content_multiple_files():
    """Test sending multiple binary files in one message."""
    png_path = FIXTURES_DIR / "Bert-WhiteBorder.png"
    jpg_path = FIXTURES_DIR / "Bert_and_Ernie.JPG"

    assert png_path.exists() and jpg_path.exists()

    png_data = png_path.read_bytes()
    jpg_data = jpg_path.read_bytes()

    agent = Agent("claude-code:sonnet")

    result = agent.run_sync(
        [
            'I am sending you two images. How many images did I send? Answer with just the number.',
            BinaryContent(data=png_data, media_type='image/png', identifier='bert_white'),
            BinaryContent(data=jpg_data, media_type='image/jpeg', identifier='bert_ernie'),
        ]
    )

    # Verify we got a response
    assert result.output is not None
    assert isinstance(result.output, str)
    # Should mention "2" or "two"
    assert '2' in result.output or 'two' in result.output.lower()
    print(f"Multiple files response: {result.output}")


def test_binary_content_preserves_data():
    """Test that binary data is preserved correctly through the pipeline."""
    import tempfile

    # Use a real file to ensure data integrity
    source_path = FIXTURES_DIR / "Bert-WhiteBorder.png"
    assert source_path.exists()

    original_data = source_path.read_bytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = Agent("claude-code:sonnet")

        # Send binary content with a specific working directory so we can inspect it
        result = agent.run_sync(
            [
                'Acknowledge receipt of the image file.',
                BinaryContent(data=original_data, media_type='image/png'),
            ],
            model_settings={"working_directory": tmpdir}
        )

        # Find the created subdirectory
        subdirs = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
        assert len(subdirs) >= 1

        # Find the PNG file created in the working directory
        png_files = list(subdirs[0].glob('*.png'))
        assert len(png_files) >= 1, "No PNG file found in working directory"

        # Verify the data is preserved
        written_data = png_files[0].read_bytes()
        assert written_data == original_data, "Binary data was not preserved correctly"

        # Verify we got a response
        assert result.output is not None
        print(f"Data preservation response: {result.output}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
