"""Tests for utility modules (_utils package)."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from pydantic_ai_claude_code._utils import (
    copy_additional_files,
    get_next_call_subdirectory,
    strip_markdown_code_fence,
    convert_primitive_value,
)
from pydantic_ai_claude_code._utils.json_utils import extract_json_from_text
from pydantic_ai_claude_code._utils.type_utils import get_type_description


class TestFileUtils:
    """Tests for file_utils module."""

    def test_get_next_call_subdirectory_creates_first_subdir(self, tmp_path):
        """Test creating the first numbered subdirectory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        result = get_next_call_subdirectory(str(base_dir))

        assert result == base_dir / "1"
        assert result.exists()
        assert result.is_dir()

    def test_get_next_call_subdirectory_increments_correctly(self, tmp_path):
        """Test that subdirectories increment numerically."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Create first subdirectory
        first = get_next_call_subdirectory(str(base_dir))
        assert first == base_dir / "1"

        # Create second subdirectory
        second = get_next_call_subdirectory(str(base_dir))
        assert second == base_dir / "2"

        # Create third subdirectory
        third = get_next_call_subdirectory(str(base_dir))
        assert third == base_dir / "3"

    def test_get_next_call_subdirectory_with_existing_numbered_dirs(self, tmp_path):
        """Test that it correctly identifies next number when dirs exist."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Create some existing numbered directories
        (base_dir / "1").mkdir()
        (base_dir / "2").mkdir()
        (base_dir / "5").mkdir()  # Gap in numbering

        # Should create "4" (total count + 1)
        result = get_next_call_subdirectory(str(base_dir))
        assert result == base_dir / "4"

    def test_get_next_call_subdirectory_ignores_non_numeric_dirs(self, tmp_path):
        """Test that non-numeric directories are ignored."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Create some non-numeric directories
        (base_dir / "foo").mkdir()
        (base_dir / "bar").mkdir()
        (base_dir / "1").mkdir()

        # Should create "2" (only count numeric dirs)
        result = get_next_call_subdirectory(str(base_dir))
        assert result == base_dir / "2"

    def test_copy_additional_files_basic(self, tmp_path):
        """Test copying a single file."""
        source_file = tmp_path / "source.txt"
        source_file.write_text("test content")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        copy_additional_files(str(dest_dir), {"output.txt": source_file})

        dest_file = dest_dir / "output.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "test content"

    def test_copy_additional_files_multiple(self, tmp_path):
        """Test copying multiple files."""
        source1 = tmp_path / "file1.txt"
        source1.write_text("content 1")
        source2 = tmp_path / "file2.txt"
        source2.write_text("content 2")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        copy_additional_files(
            str(dest_dir), {"out1.txt": source1, "out2.txt": source2}
        )

        assert (dest_dir / "out1.txt").read_text() == "content 1"
        assert (dest_dir / "out2.txt").read_text() == "content 2"

    def test_copy_additional_files_with_subdirectories(self, tmp_path):
        """Test copying files into subdirectories."""
        source_file = tmp_path / "source.txt"
        source_file.write_text("nested content")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        copy_additional_files(str(dest_dir), {"sub/dir/output.txt": source_file})

        dest_file = dest_dir / "sub" / "dir" / "output.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "nested content"

    def test_copy_additional_files_missing_source(self, tmp_path):
        """Test error when source file doesn't exist."""
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="not found"):
            copy_additional_files(str(dest_dir), {"output.txt": nonexistent})

    def test_copy_additional_files_source_is_directory(self, tmp_path):
        """Test error when source is a directory."""
        source_dir = tmp_path / "source_dir"
        source_dir.mkdir()

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        with pytest.raises(ValueError, match="not a file"):
            copy_additional_files(str(dest_dir), {"output": source_dir})


class TestJsonUtils:
    """Tests for json_utils module."""

    def test_strip_markdown_code_fence_with_json_prefix(self):
        """Test stripping ```json code fence."""
        text = "```json\n{\"key\": \"value\"}\n```"
        result = strip_markdown_code_fence(text)
        assert result == '{"key": "value"}'

    def test_strip_markdown_code_fence_with_plain_backticks(self):
        """Test stripping plain ``` code fence."""
        text = "```\n{\"key\": \"value\"}\n```"
        result = strip_markdown_code_fence(text)
        assert result == '{"key": "value"}'

    def test_strip_markdown_code_fence_no_fence(self):
        """Test that text without fences is returned as-is."""
        text = '{"key": "value"}'
        result = strip_markdown_code_fence(text)
        assert result == text

    def test_strip_markdown_code_fence_only_leading(self):
        """Test stripping only leading fence."""
        text = "```json\n{\"key\": \"value\"}"
        result = strip_markdown_code_fence(text)
        assert result == '{"key": "value"}'

    def test_strip_markdown_code_fence_only_trailing(self):
        """Test stripping only trailing fence."""
        text = "{\"key\": \"value\"}\n```"
        result = strip_markdown_code_fence(text)
        assert result == '{"key": "value"}'

    def test_extract_json_from_text_direct_parse(self):
        """Test direct JSON parsing after markdown stripping."""
        text = "```json\n{\"name\": \"test\", \"value\": 42}\n```"
        result = extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_from_text_embedded_in_text(self):
        """Test extracting JSON from surrounding text."""
        text = "Here is the data: {\"result\": \"success\"} and that's it"
        result = extract_json_from_text(text)
        assert result == {"result": "success"}

    def test_extract_json_from_text_with_schema_single_field(self):
        """Test single-field wrapping when extraction fails."""
        text = "just some plain text"
        schema = {"properties": {"message": {"type": "string"}}}
        result = extract_json_from_text(text, schema)
        assert result == {"message": "just some plain text"}

    def test_extract_json_from_text_invalid_json_no_schema(self):
        """Test that None is returned for invalid JSON without schema."""
        text = "not json at all"
        result = extract_json_from_text(text)
        assert result is None

    def test_extract_json_from_text_nested_braces(self):
        """Test extraction with nested JSON objects."""
        text = 'Some text {"outer": {"inner": "value"}} more text'
        result = extract_json_from_text(text)
        assert result == {"outer": {"inner": "value"}}


class TestTypeUtils:
    """Tests for type_utils module."""

    def test_convert_primitive_value_integer(self):
        """Test converting string to integer."""
        assert convert_primitive_value("42", "integer") == 42
        assert convert_primitive_value("-10", "integer") == -10
        assert convert_primitive_value("0", "integer") == 0

    def test_convert_primitive_value_number_float(self):
        """Test converting string to float."""
        assert convert_primitive_value("3.14", "number") == 3.14
        assert convert_primitive_value("-2.5", "number") == -2.5
        assert convert_primitive_value("1e-3", "number") == 0.001

    def test_convert_primitive_value_number_integer_format(self):
        """Test that integers stay as integers for 'number' type."""
        result = convert_primitive_value("42", "number")
        assert result == 42
        assert isinstance(result, int)

    def test_convert_primitive_value_boolean_true(self):
        """Test converting strings to True."""
        assert convert_primitive_value("true", "boolean") is True
        assert convert_primitive_value("True", "boolean") is True
        assert convert_primitive_value("TRUE", "boolean") is True
        assert convert_primitive_value("1", "boolean") is True
        assert convert_primitive_value("yes", "boolean") is True

    def test_convert_primitive_value_boolean_false(self):
        """Test converting strings to False."""
        assert convert_primitive_value("false", "boolean") is False
        assert convert_primitive_value("no", "boolean") is False
        assert convert_primitive_value("0", "boolean") is False

    def test_convert_primitive_value_string(self):
        """Test string type returns string as-is."""
        assert convert_primitive_value("hello", "string") == "hello"
        assert convert_primitive_value("123", "string") == "123"

    def test_convert_primitive_value_invalid_integer(self):
        """Test that invalid integer conversion returns None."""
        assert convert_primitive_value("not a number", "integer") is None
        assert convert_primitive_value("3.14", "integer") is None

    def test_convert_primitive_value_invalid_number(self):
        """Test that invalid number conversion returns None."""
        assert convert_primitive_value("not a number", "number") is None

    def test_convert_primitive_value_unknown_type(self):
        """Test that unknown types return None."""
        assert convert_primitive_value("value", "unknown_type") is None

    def test_get_type_description_known_types(self):
        """Test descriptions for known types."""
        assert get_type_description("string") == "Text value"
        assert get_type_description("integer") == "Whole number"
        assert get_type_description("number") == "Numeric value"
        assert get_type_description("boolean") == "True/false value"

    def test_get_type_description_unknown_type(self):
        """Test default description for unknown types."""
        assert get_type_description("unknown") == "Value"
        assert get_type_description("custom_type") == "Value"