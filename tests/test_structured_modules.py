"""Tests for structured output modules (structured package)."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pydantic_ai_claude_code.structured import (
    create_structured_output_path,
    create_unstructured_output_path,
    read_structured_output,
    read_unstructured_output,
    build_function_selection_prompt,
    parse_function_selection,
    build_argument_collection_prompt,
)
from pydantic_ai_claude_code.structured.file_handler import cleanup_output_file
from pydantic_ai_claude_code.structured.function_selector import build_retry_prompt


class TestFileHandler:
    """Tests for file_handler module."""

    def test_create_structured_output_path_format(self):
        """Test that structured output path follows expected format."""
        path = create_structured_output_path()
        
        assert path.startswith("/tmp/claude_data_structure_")
        assert len(path) > len("/tmp/claude_data_structure_")

    def test_create_structured_output_path_unique(self):
        """Test that multiple calls create unique paths."""
        path1 = create_structured_output_path()
        path2 = create_structured_output_path()
        path3 = create_structured_output_path()
        
        assert path1 != path2
        assert path2 != path3
        assert path1 != path3

    def test_create_unstructured_output_path_format(self):
        """Test that unstructured output path follows expected format."""
        path = create_unstructured_output_path()
        
        assert path.startswith("/tmp/claude_unstructured_output_")
        assert path.endswith(".txt")

    def test_create_unstructured_output_path_unique(self):
        """Test that multiple calls create unique paths."""
        path1 = create_unstructured_output_path()
        path2 = create_unstructured_output_path()
        
        assert path1 != path2

    def test_read_structured_output_success(self, tmp_path):
        """Test reading structured output from directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create simple structure
        (output_dir / "name.txt").write_text("John")
        (output_dir / "age.txt").write_text("30")
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        with patch("pydantic_ai_claude_code.structured.file_handler.read_structure_from_filesystem") as mock_read:
            mock_read.return_value = {"name": "John", "age": 30}
            
            result = read_structured_output(schema, str(output_dir))
            
            assert result == {"name": "John", "age": 30}
            mock_read.assert_called_once()

    def test_read_structured_output_directory_not_found(self, tmp_path):
        """Test error when output directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        schema = {"type": "object", "properties": {}}
        
        with pytest.raises(RuntimeError, match="not found"):
            read_structured_output(schema, str(nonexistent))

    def test_read_unstructured_output_success(self, tmp_path):
        """Test reading unstructured output from file."""
        output_file = tmp_path / "output.txt"
        output_file.write_text("This is the output content")
        
        result = read_unstructured_output(str(output_file))
        
        assert result == "This is the output content"

    def test_read_unstructured_output_file_not_found(self, tmp_path):
        """Test error when output file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.txt"
        
        with pytest.raises(RuntimeError, match="not found"):
            read_unstructured_output(str(nonexistent))

    def test_cleanup_output_file_removes_file(self, tmp_path):
        """Test cleaning up output file."""
        output_file = tmp_path / "output.txt"
        output_file.write_text("content")
        
        cleanup_output_file(str(output_file))
        
        assert not output_file.exists()

    def test_cleanup_output_file_removes_directory(self, tmp_path):
        """Test cleaning up output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "file.txt").write_text("content")
        
        cleanup_output_file(str(output_dir))
        
        assert not output_dir.exists()

    def test_cleanup_output_file_nonexistent(self, tmp_path):
        """Test that cleanup doesn't error on nonexistent path."""
        nonexistent = tmp_path / "nonexistent"
        
        # Should not raise an error
        cleanup_output_file(str(nonexistent))


class TestFunctionSelector:
    """Tests for function_selector module."""

    def test_build_function_selection_prompt_single_function(self):
        """Test building selection prompt with one function."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}}
                }
            }
        ]
        
        prompt, functions = build_function_selection_prompt(tools)
        
        assert "get_weather" in prompt
        assert "Get weather for a location" in prompt
        assert "CHOICE:" in prompt
        assert "get_weather" in functions
        assert functions["get_weather"]["description"] == "Get weather for a location"

    def test_build_function_selection_prompt_multiple_functions(self):
        """Test building selection prompt with multiple functions."""
        tools = [
            {"name": "func1", "description": "First function"},
            {"name": "func2", "description": "Second function"},
            {"name": "func3", "description": "Third function"},
        ]
        
        prompt, functions = build_function_selection_prompt(tools)
        
        assert "func1" in prompt
        assert "func2" in prompt
        assert "func3" in prompt
        assert "none" in prompt
        assert len(functions) == 3

    def test_build_function_selection_prompt_includes_none_option(self):
        """Test that prompt includes 'none' option."""
        tools = [{"name": "test_func", "description": "Test"}]
        
        prompt, _ = build_function_selection_prompt(tools)
        
        assert "none" in prompt.lower()
        assert "Answer directly without calling a function" in prompt or "without" in prompt.lower()

    def test_parse_function_selection_valid_choice(self):
        """Test parsing valid function selection."""
        response = "CHOICE: get_weather"
        
        result = parse_function_selection(response)
        
        assert result == "get_weather"

    def test_parse_function_selection_none_choice(self):
        """Test parsing 'none' selection."""
        response = "CHOICE: none"
        
        result = parse_function_selection(response)
        
        assert result == "none"

    def test_parse_function_selection_case_insensitive(self):
        """Test that parsing is case insensitive."""
        response = "choice: MyFunction"
        
        result = parse_function_selection(response)
        
        assert result == "myfunction"

    def test_parse_function_selection_with_surrounding_text(self):
        """Test parsing when choice is embedded in text."""
        response = "Based on the request, I will select:\nCHOICE: calculate\nThis seems appropriate."
        
        result = parse_function_selection(response)
        
        assert result == "calculate"

    def test_parse_function_selection_invalid_format(self):
        """Test that invalid format returns None."""
        response = "I think we should use get_weather"
        
        result = parse_function_selection(response)
        
        assert result is None

    def test_build_argument_collection_prompt_structure(self):
        """Test building argument collection prompt."""
        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
        
        with patch("pydantic_ai_claude_code.structured.function_selector.build_structure_instructions") as mock_build:
            mock_build.return_value = "Structure instructions"
            
            prompt = build_argument_collection_prompt(
                "get_weather",
                "Get weather information",
                schema,
                "/tmp/output"
            )
            
            assert prompt == "Structure instructions"
            mock_build.assert_called_once_with(
                schema,
                "/tmp/output",
                tool_name="get_weather",
                tool_description="Get weather information"
            )

    def test_build_retry_prompt_includes_error(self):
        """Test that retry prompt includes error message."""
        schema = {"type": "object", "properties": {"field": {"type": "string"}}}
        error_msg = "Field 'location' is required"
        
        with patch("pydantic_ai_claude_code.structured.function_selector.build_structure_instructions") as mock_build:
            mock_build.return_value = "Instructions"
            
            prompt = build_retry_prompt(
                "original prompt",
                schema,
                "/tmp/output",
                error_msg
            )
            
            assert "Retry" in prompt
            assert error_msg in prompt
            assert "failed" in prompt.lower()

    def test_build_retry_prompt_rebuilds_instructions(self):
        """Test that retry prompt rebuilds structure instructions."""
        schema = {"type": "object", "properties": {}}
        
        with patch("pydantic_ai_claude_code.structured.function_selector.build_structure_instructions") as mock_build:
            mock_build.return_value = "New instructions"
            
            prompt = build_retry_prompt(
                "original",
                schema,
                "/tmp/new_output",
                "error"
            )
            
            mock_build.assert_called_once_with(schema, "/tmp/new_output")
            assert "New instructions" in prompt


class TestConverterReexports:
    """Tests for converter module re-exports."""

    def test_converter_exports_structure_functions(self):
        """Test that converter module exports expected functions."""
        from pydantic_ai_claude_code.structured.converter import (
            write_structure_to_filesystem,
            read_structure_from_filesystem,
            build_structure_instructions,
        )
        
        # Just verify they're importable and callable
        assert callable(write_structure_to_filesystem)
        assert callable(read_structure_from_filesystem)
        assert callable(build_structure_instructions)