"""Tests for BinaryContent handling in messages."""

import tempfile
from pathlib import Path

from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

from pydantic_ai_claude_code.messages import format_messages_for_claude


def test_format_binary_content_image():
    """Test formatting a message with an image BinaryContent."""
    # Create a fake PNG image (just some bytes)
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'

    with tempfile.TemporaryDirectory() as tmpdir:
        binary_content = BinaryContent(data=png_data, media_type='image/png')

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[
                        'What is in this image?',
                        binary_content,
                    ])
                ]
            )
        ]

        result = format_messages_for_claude(messages, working_dir=tmpdir)

        # Check that the prompt contains a file reference using @ syntax
        assert '@' in result
        assert '.png' in result
        assert 'What is in this image?' in result

        # Check that the file was created
        files = list(Path(tmpdir).glob('*.png'))
        assert len(files) == 1
        assert files[0].read_bytes() == png_data


def test_format_binary_content_pdf():
    """Test formatting a message with a PDF BinaryContent."""
    # Create a minimal PDF
    pdf_data = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'

    with tempfile.TemporaryDirectory() as tmpdir:
        binary_content = BinaryContent(data=pdf_data, media_type='application/pdf')

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[
                        'Summarize this document:',
                        binary_content,
                    ])
                ]
            )
        ]

        result = format_messages_for_claude(messages, working_dir=tmpdir)

        # Check that the prompt contains a file reference using @ syntax
        assert '@' in result
        assert '.pdf' in result
        assert 'Summarize this document:' in result

        # Check that the file was created
        files = list(Path(tmpdir).glob('*.pdf'))
        assert len(files) == 1
        assert files[0].read_bytes() == pdf_data


def test_format_multiple_binary_content():
    """Test formatting a message with multiple BinaryContent items."""
    img1 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    img2 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02'

    with tempfile.TemporaryDirectory() as tmpdir:
        binary1 = BinaryContent(data=img1, media_type='image/png', identifier='image1')
        binary2 = BinaryContent(data=img2, media_type='image/jpeg', identifier='image2')

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[
                        'Compare these two images:',
                        binary1,
                        'and',
                        binary2,
                    ])
                ]
            )
        ]

        result = format_messages_for_claude(messages, working_dir=tmpdir)

        # Check that the prompt contains both file references using @ syntax
        assert result.count('@') == 2
        assert 'image1' in result
        assert 'image2' in result
        assert 'Compare these two images:' in result
        assert ' and ' in result

        # Check that both files were created
        files = list(Path(tmpdir).glob('*'))
        assert len(files) == 2


def test_format_binary_content_with_text_only():
    """Test that text-only messages still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Just a simple text message')
                ]
            )
        ]

        result = format_messages_for_claude(messages, working_dir=tmpdir)

        assert result == 'Request: Just a simple text message'

        # No files should be created
        files = list(Path(tmpdir).glob('*'))
        assert len(files) == 0


def test_format_binary_content_identifier_sanitization():
    """Test that file identifiers are sanitized for safe filenames."""
    png_data = b'\x89PNG\r\n\x1a\n'

    with tempfile.TemporaryDirectory() as tmpdir:
        # Identifier with unsafe characters
        binary_content = BinaryContent(
            data=png_data,
            media_type='image/png',
            identifier='my/unsafe:file<name>'
        )

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=['Check this:', binary_content])
                ]
            )
        ]

        result = format_messages_for_claude(messages, working_dir=tmpdir)

        # Check that unsafe characters were replaced using @ syntax
        assert '@my_unsafe_file_name_' in result

        # File should exist with sanitized name
        files = list(Path(tmpdir).glob('*.png'))
        assert len(files) == 1
        assert files[0].read_bytes() == png_data
