"""Tests for long response handling with gradual file building."""

import tempfile
from pathlib import Path

import pytest

from pydantic_ai_claude_code.model import ClaudeCodeModel

# Test constants
LARGE_ARRAY_SIZE = 100  # Number of items for large array tests


class TestJSONAssembly:
    """Test JSON assembly from directory structure."""

    def test_assemble_scalar_fields(self):
        """Test assembling JSON with scalar fields."""
        model = ClaudeCodeModel()

        # Create temp directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create field files
            (tmp_path / "name.txt").write_text("Alice")
            (tmp_path / "age.txt").write_text("30")
            (tmp_path / "score.txt").write_text("95.5")
            (tmp_path / "active.txt").write_text("true")

            # Schema
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "score": {"type": "number"},
                    "active": {"type": "boolean"},
                },
            }

            # Assemble
            result = model._assemble_json_from_directory(tmp_path, schema)

            assert result == {
                "name": "Alice",
                "age": 30,
                "score": 95.5,
                "active": True,
            }

    def test_assemble_array_field(self):
        """Test assembling JSON with array fields."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create array directory
            array_dir = tmp_path / "items"
            array_dir.mkdir()

            # Create numbered files
            (array_dir / "0000.txt").write_text("first")
            (array_dir / "0001.txt").write_text("second")
            (array_dir / "0002.txt").write_text("third")

            # Schema
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = model._assemble_json_from_directory(tmp_path, schema)

            assert result == {"items": ["first", "second", "third"]}

    def test_assemble_mixed_fields(self):
        """Test assembling JSON with both scalar and array fields."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create scalar fields
            (tmp_path / "title.txt").write_text("My Report")
            (tmp_path / "count.txt").write_text("42")

            # Create array field
            tags_dir = tmp_path / "tags"
            tags_dir.mkdir()
            (tags_dir / "0000.txt").write_text("python")
            (tags_dir / "0001.txt").write_text("ai")
            (tags_dir / "0002.txt").write_text("testing")

            # Schema
            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "count": {"type": "integer"},
                    "tags": {"type": "array"},
                },
            }

            # Assemble
            result = model._assemble_json_from_directory(tmp_path, schema)

            assert result == {
                "title": "My Report",
                "count": 42,
                "tags": ["python", "ai", "testing"],
            }

    def test_assemble_with_multiline_content(self):
        """Test assembling JSON with multiline field content."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create field with multiline content
            (tmp_path / "description.txt").write_text(
                "This is line 1\nThis is line 2\nThis is line 3"
            )

            # Schema
            schema = {
                "type": "object",
                "properties": {"description": {"type": "string"}},
            }

            # Assemble
            result = model._assemble_json_from_directory(tmp_path, schema)

            assert (
                result["description"]
                == "This is line 1\nThis is line 2\nThis is line 3"
            )

    def test_assemble_array_sorting(self):
        """Test that array items are sorted correctly by filename."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create array directory with files in non-sequential order
            array_dir = tmp_path / "items"
            array_dir.mkdir()

            # Write files out of order
            (array_dir / "0005.txt").write_text("sixth")
            (array_dir / "0000.txt").write_text("first")
            (array_dir / "0003.txt").write_text("fourth")
            (array_dir / "0001.txt").write_text("second")
            (array_dir / "0004.txt").write_text("fifth")
            (array_dir / "0002.txt").write_text("third")

            # Schema
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = model._assemble_json_from_directory(tmp_path, schema)

            # Should be sorted by filename
            assert result["items"] == [
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
                "sixth",
            ]

    def test_assemble_boolean_variations(self):
        """Test different boolean value formats."""
        model = ClaudeCodeModel()

        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
        ]

        for input_val, expected in test_cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                (tmp_path / "flag.txt").write_text(input_val)

                schema = {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                }

                result = model._assemble_json_from_directory(tmp_path, schema)
                assert result["flag"] == expected, f"Failed for input: {input_val}"

    def test_assemble_missing_field_error(self):
        """Test error when required field is missing."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Only create one field, but schema expects two
            (tmp_path / "name.txt").write_text("Alice")

            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},  # Missing!
                },
            }

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Missing field file"):
                model._assemble_json_from_directory(tmp_path, schema)

    def test_assemble_missing_array_directory_error(self):
        """Test error when array directory is missing."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Don't create the array directory

            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Missing array directory"):
                model._assemble_json_from_directory(tmp_path, schema)

    def test_assemble_empty_array(self):
        """Test assembling an empty array."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create empty array directory
            array_dir = tmp_path / "items"
            array_dir.mkdir()

            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = model._assemble_json_from_directory(tmp_path, schema)

            assert result == {"items": []}

    def test_assemble_large_array(self):
        """Test assembling a large array with many items."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create array directory with 100 items
            array_dir = tmp_path / "items"
            array_dir.mkdir()

            for i in range(LARGE_ARRAY_SIZE):
                (array_dir / f"{i:04d}.txt").write_text(f"item_{i}")

            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = model._assemble_json_from_directory(tmp_path, schema)

            assert len(result["items"]) == LARGE_ARRAY_SIZE
            assert result["items"][0] == "item_0"
            assert result["items"][LARGE_ARRAY_SIZE - 1] == f"item_{LARGE_ARRAY_SIZE - 1}"

    def test_structured_output_instruction_format(self):
        """Test that structured output instruction includes correct format."""
        model = ClaudeCodeModel()

        from pydantic_ai_claude_code.types import ClaudeCodeSettings

        settings: ClaudeCodeSettings = {}

        # Mock output tool
        class MockTool:
            parameters_json_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {"type": "array"},
                    "count": {"type": "integer"},
                },
            }

        instruction = model._build_structured_output_instruction(MockTool(), settings)

        # Check that instruction includes key elements
        assert "mkdir -p /tmp/claude_json_fields_" in instruction
        assert "name.txt" in instruction
        assert "tags/" in instruction
        assert "0000.txt, 0001.txt" in instruction
        assert ".complete" in instruction
        assert "DO NOT manually create JSON" in instruction

    def test_unstructured_output_instruction_format(self):
        """Test that unstructured output instruction includes gradual appending."""
        model = ClaudeCodeModel()

        from pydantic_ai_claude_code.types import ClaudeCodeSettings

        settings: ClaudeCodeSettings = {}

        instruction = model._build_unstructured_output_instruction(settings)

        # Check that instruction includes key elements
        assert "Write tool" in instruction
        assert "bash" in instruction
        assert "echo" in instruction
        assert ">>" in instruction  # Append operator
        assert "cat <<" in instruction  # Heredoc
        assert "incrementally" in instruction


class TestValidationErrors:
    """Test validation error handling for assembled JSON."""

    def test_type_mismatch_in_assembled_json(self):
        """Test that type mismatches are caught and reported."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create fields with wrong types
            (tmp_path / "name.txt").write_text("Alice")
            (tmp_path / "age.txt").write_text("not_a_number")  # Should be integer!
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }

            from pydantic_ai_claude_code.types import ClaudeCodeSettings

            settings: ClaudeCodeSettings = {"__temp_json_dir": str(tmp_path)}

            # Read and validate - should catch type error
            parsed_data, error_msg = model._read_structured_output_file(
                "/tmp/dummy.json", schema, settings
            )

            assert parsed_data is None
            assert error_msg is not None
            assert "type" in error_msg.lower() or "invalid" in error_msg.lower()

    def test_missing_required_field_validation(self):
        """Test that missing required fields are caught."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Only create one field, schema requires two
            (tmp_path / "name.txt").write_text("Alice")
            # Missing: age.txt
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }

            from pydantic_ai_claude_code.types import ClaudeCodeSettings

            settings: ClaudeCodeSettings = {"__temp_json_dir": str(tmp_path)}

            # Should catch missing field during assembly
            parsed_data, error_msg = model._read_structured_output_file(
                "/tmp/dummy.json", schema, settings
            )

            assert parsed_data is None
            assert error_msg is not None
            assert "missing" in error_msg.lower() or "age" in error_msg.lower()

    def test_validation_error_returns_as_text(self):
        """Test that validation errors are returned as TextPart for retry."""
        model = ClaudeCodeModel()

        # Mock response with validation error scenario
        from pydantic_ai_claude_code.types import ClaudeJSONResponse

        response: ClaudeJSONResponse = {
            "result": "some text",
            "is_error": False,
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "total_cost_usd": 0.001,
        }

        # Mock an output tool
        class MockOutputTool:
            name = "test_output"
            parameters_json_schema = {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }

        # Test with a file that will fail validation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_file = tmp_path / "output.json"
            output_file.write_text('{"value": "not_an_int"}')  # Invalid!

            from pydantic_ai_claude_code.types import ClaudeCodeSettings

            settings: ClaudeCodeSettings = {
                "__structured_output_file": str(output_file)
            }

            result = model._convert_response(
                response, output_tools=[MockOutputTool()], settings=settings
            )

            # Should return TextPart with error message
            assert len(result.parts) == 1
            from pydantic_ai.messages import TextPart

            assert isinstance(result.parts[0], TextPart)
            # Error message should mention the type issue
            error_text = result.parts[0].content
            assert "value" in error_text.lower()
            assert "type" in error_text.lower() or "int" in error_text.lower()
