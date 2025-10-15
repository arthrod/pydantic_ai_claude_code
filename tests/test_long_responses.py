"""Tests for long response handling with gradual file building."""

import tempfile
from pathlib import Path

import pytest

from pydantic_ai_claude_code.model import ClaudeCodeModel
from pydantic_ai_claude_code.structure_converter import read_structure_from_filesystem

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
            (tmp_path / ".complete").touch()

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
            result = read_structure_from_filesystem(schema, tmp_path)

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
            (tmp_path / ".complete").touch()

            # Schema
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

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
            (tmp_path / ".complete").touch()

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
            result = read_structure_from_filesystem(schema, tmp_path)

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
            (tmp_path / ".complete").touch()

            # Schema
            schema = {
                "type": "object",
                "properties": {"description": {"type": "string"}},
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

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
            (tmp_path / ".complete").touch()

            # Schema
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

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
                (tmp_path / ".complete").touch()

                schema = {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                }

                result = read_structure_from_filesystem(schema, tmp_path)
                assert result["flag"] == expected, f"Failed for input: {input_val}"

    def test_assemble_missing_field_error(self):
        """Test error when required field is missing."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Only create one field, but schema expects two
            (tmp_path / "name.txt").write_text("Alice")
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},  # Missing!
                },
            }

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Missing file:"):
                read_structure_from_filesystem(schema, tmp_path)

    def test_assemble_missing_array_directory_error(self):
        """Test error when array directory is missing."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Don't create the array directory
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Missing directory:"):
                read_structure_from_filesystem(schema, tmp_path)

    def test_assemble_empty_array(self):
        """Test assembling an empty array."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create empty array directory
            array_dir = tmp_path / "items"
            array_dir.mkdir()
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

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
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

            assert len(result["items"]) == LARGE_ARRAY_SIZE
            assert result["items"][0] == "item_0"
            assert result["items"][LARGE_ARRAY_SIZE - 1] == f"item_{LARGE_ARRAY_SIZE - 1}"

    def test_assemble_array_with_object_items(self):
        """Test assembling array with nested objects (e.g., list[BaseModel])."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create array directory with object items
            array_dir = tmp_path / "recommendations"
            array_dir.mkdir()

            # Create numbered subdirectories for each object
            # Item 0
            item_0 = array_dir / "0000"
            item_0.mkdir()
            (item_0 / "priority.txt").write_text("1")
            criteria_0 = item_0 / "criteria_addressed"
            criteria_0.mkdir()
            (criteria_0 / "0000.txt").write_text("clarity")
            (criteria_0 / "0001.txt").write_text("impact")
            (item_0 / "current_weakness.txt").write_text("Lacks detail")
            (item_0 / "specific_action.txt").write_text("Add examples")
            (item_0 / ".complete").touch()

            # Item 1
            item_1 = array_dir / "0001"
            item_1.mkdir()
            (item_1 / "priority.txt").write_text("2")
            criteria_1 = item_1 / "criteria_addressed"
            criteria_1.mkdir()
            (criteria_1 / "0000.txt").write_text("feasibility")
            (item_1 / "current_weakness.txt").write_text("Budget unclear")
            (item_1 / "specific_action.txt").write_text("Include cost breakdown")
            (item_1 / ".complete").touch()

            # Item 2
            item_2 = array_dir / "0002"
            item_2.mkdir()
            (item_2 / "priority.txt").write_text("3")
            criteria_2 = item_2 / "criteria_addressed"
            criteria_2.mkdir()
            (criteria_2 / "0000.txt").write_text("innovation")
            (item_2 / "current_weakness.txt").write_text("Incremental improvements")
            (item_2 / "specific_action.txt").write_text("Highlight novel approach")
            (item_2 / ".complete").touch()

            (tmp_path / ".complete").touch()

            # Schema matching ImprovementRecommendation from the bug report
            schema = {
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "priority": {"type": "integer"},
                                "criteria_addressed": {"type": "array"},
                                "current_weakness": {"type": "string"},
                                "specific_action": {"type": "string"},
                            },
                        },
                    }
                },
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

            # Verify structure
            assert "recommendations" in result
            assert len(result["recommendations"]) == 3

            # Verify first item is a dict (not a string!)
            assert isinstance(result["recommendations"][0], dict)
            assert result["recommendations"][0]["priority"] == 1
            assert isinstance(result["recommendations"][0]["criteria_addressed"], list)
            assert result["recommendations"][0]["criteria_addressed"] == ["clarity", "impact"]
            assert result["recommendations"][0]["current_weakness"] == "Lacks detail"
            assert result["recommendations"][0]["specific_action"] == "Add examples"

            # Verify second item
            assert isinstance(result["recommendations"][1], dict)
            assert result["recommendations"][1]["priority"] == 2

            # Verify third item
            assert isinstance(result["recommendations"][2], dict)
            assert result["recommendations"][2]["priority"] == 3

    def test_assemble_array_with_integer_items(self):
        """Test assembling array with integer items."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create array directory with integer items
            array_dir = tmp_path / "scores"
            array_dir.mkdir()

            (array_dir / "0000.txt").write_text("95")
            (array_dir / "0001.txt").write_text("87")
            (array_dir / "0002.txt").write_text("92")
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                },
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

            assert result == {"scores": [95, 87, 92]}
            # Verify types
            assert all(isinstance(x, int) for x in result["scores"])

    def test_assemble_array_with_number_items(self):
        """Test assembling array with float items."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create array directory with float items
            array_dir = tmp_path / "temperatures"
            array_dir.mkdir()

            (array_dir / "0000.txt").write_text("98.6")
            (array_dir / "0001.txt").write_text("99.2")
            (array_dir / "0002.txt").write_text("97.8")
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {
                    "temperatures": {
                        "type": "array",
                        "items": {"type": "number"},
                    }
                },
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

            assert result == {"temperatures": [98.6, 99.2, 97.8]}
            # Verify types
            assert all(isinstance(x, float) for x in result["temperatures"])

    def test_assemble_array_with_boolean_items(self):
        """Test assembling array with boolean items."""
        model = ClaudeCodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create array directory with boolean items
            array_dir = tmp_path / "flags"
            array_dir.mkdir()

            (array_dir / "0000.txt").write_text("true")
            (array_dir / "0001.txt").write_text("false")
            (array_dir / "0002.txt").write_text("1")
            (array_dir / "0003.txt").write_text("0")
            (tmp_path / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {
                    "flags": {
                        "type": "array",
                        "items": {"type": "boolean"},
                    }
                },
            }

            # Assemble
            result = read_structure_from_filesystem(schema, tmp_path)

            assert result == {"flags": [True, False, True, False]}
            # Verify types
            assert all(isinstance(x, bool) for x in result["flags"])

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
        assert "mkdir -p /tmp/claude_data_structure_" in instruction
        assert "name.txt" in instruction
        assert "tags/" in instruction
        assert "0000.txt, 0001.txt" in instruction
        # Completion marker removed as unnecessary (CLI execution is synchronous)
        assert ".complete" not in instruction
        assert "Task: Organize your response" in instruction
        assert "Information to provide:" in instruction
        assert "Required information:" in instruction

    def test_unstructured_output_instruction_format(self):
        """Test that unstructured output instruction uses Write tool."""
        model = ClaudeCodeModel()

        from pydantic_ai_claude_code.types import ClaudeCodeSettings

        settings: ClaudeCodeSettings = {}

        instruction = model._build_unstructured_output_instruction(settings)

        # Check that instruction includes key elements
        assert "Write tool" in instruction
        assert "/tmp/claude_unstructured_output_" in instruction
        assert "complete response" in instruction
        assert "ONLY your direct answer" in instruction


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
