"""Tests for type-preserving structure conversion between data and filesystem.

These tests verify that:
1. Data → Filesystem conversion works correctly
2. Filesystem → Data conversion preserves all types
3. Round-trip conversions maintain exact equality
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from pydantic_ai_claude_code.structure_converter import (
    build_structure_instructions,
    read_structure_from_filesystem,
    write_structure_to_filesystem,
)


def test_simple_scalar_types_round_trip():
    """Test round-trip conversion with simple scalar types."""
    schema = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "score": {"type": "number"},
            "active": {"type": "boolean"},
        },
        "required": ["name", "age"],
    }

    original_data = {
        "name": "Alice Smith",
        "age": 30,
        "score": 95.5,
        "active": True,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify files exist
        assert (base_path / "name.txt").exists()
        assert (base_path / "age.txt").exists()
        assert (base_path / "score.txt").exists()
        assert (base_path / "active.txt").exists()

        # Verify file contents
        assert (base_path / "name.txt").read_text() == "Alice Smith"
        assert (base_path / "age.txt").read_text() == "30"
        assert (base_path / "score.txt").read_text() == "95.5"
        assert (base_path / "active.txt").read_text() == "true"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify exact equality
        assert loaded_data == original_data
        assert type(loaded_data["name"]) is str
        assert type(loaded_data["age"]) is int
        assert type(loaded_data["score"]) is float
        assert type(loaded_data["active"]) is bool

        # Data → Filesystem again
        base_path2 = Path(tmpdir) / "data2"
        write_structure_to_filesystem(loaded_data, schema, base_path2)

        # Verify file contents are identical
        assert (base_path2 / "name.txt").read_text() == (base_path / "name.txt").read_text()
        assert (base_path2 / "age.txt").read_text() == (base_path / "age.txt").read_text()
        assert (base_path2 / "score.txt").read_text() == (base_path / "score.txt").read_text()
        assert (base_path2 / "active.txt").read_text() == (base_path / "active.txt").read_text()


def test_array_of_primitives_round_trip():
    """Test round-trip conversion with arrays of primitive types."""
    schema = {
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scores": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "ratings": {
                "type": "array",
                "items": {"type": "number"},
            },
            "flags": {
                "type": "array",
                "items": {"type": "boolean"},
            },
        },
        "required": ["tags"],
    }

    original_data = {
        "tags": ["python", "ai", "testing", "automation"],
        "scores": [100, 95, 87, 92],
        "ratings": [4.5, 3.8, 4.9, 4.2],
        "flags": [True, False, True, False],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify directory structure
        assert (base_path / "tags").is_dir()
        assert (base_path / "scores").is_dir()
        assert (base_path / "ratings").is_dir()
        assert (base_path / "flags").is_dir()

        # Verify array files
        assert (base_path / "tags" / "0000.txt").read_text() == "python"
        assert (base_path / "tags" / "0001.txt").read_text() == "ai"
        assert (base_path / "tags" / "0002.txt").read_text() == "testing"
        assert (base_path / "tags" / "0003.txt").read_text() == "automation"

        assert (base_path / "scores" / "0000.txt").read_text() == "100"
        assert (base_path / "scores" / "0001.txt").read_text() == "95"

        assert (base_path / "ratings" / "0000.txt").read_text() == "4.5"
        assert (base_path / "ratings" / "0001.txt").read_text() == "3.8"

        assert (base_path / "flags" / "0000.txt").read_text() == "true"
        assert (base_path / "flags" / "0001.txt").read_text() == "false"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify exact equality
        assert loaded_data == original_data
        assert all(type(t) is str for t in loaded_data["tags"])
        assert all(type(s) is int for s in loaded_data["scores"])
        assert all(type(r) is float for r in loaded_data["ratings"])
        assert all(type(f) is bool for f in loaded_data["flags"])

        # Data → Filesystem again
        base_path2 = Path(tmpdir) / "data2"
        write_structure_to_filesystem(loaded_data, schema, base_path2)

        # Verify identical file contents
        for i in range(4):
            assert (base_path2 / "tags" / f"{i:04d}.txt").read_text() == (
                base_path / "tags" / f"{i:04d}.txt"
            ).read_text()
            assert (base_path2 / "scores" / f"{i:04d}.txt").read_text() == (
                base_path / "scores" / f"{i:04d}.txt"
            ).read_text()


def test_nested_object_round_trip():
    """Test round-trip conversion with nested objects."""
    schema = {
        "properties": {
            "title": {"type": "string"},
            "author": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "version": {"type": "integer"},
                    "published": {"type": "boolean"},
                },
            },
        },
        "required": ["title"],
    }

    original_data = {
        "title": "Python AI Programming",
        "author": {
            "first_name": "Alice",
            "last_name": "Johnson",
            "age": 35,
            "email": "alice@example.com",
        },
        "metadata": {
            "version": 2,
            "published": True,
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify nested structure
        assert (base_path / "title.txt").read_text() == "Python AI Programming"
        assert (base_path / "author").is_dir()
        assert (base_path / "author" / "first_name.txt").read_text() == "Alice"
        assert (base_path / "author" / "last_name.txt").read_text() == "Johnson"
        assert (base_path / "author" / "age.txt").read_text() == "35"
        assert (base_path / "author" / "email.txt").read_text() == "alice@example.com"

        assert (base_path / "metadata").is_dir()
        assert (base_path / "metadata" / "version.txt").read_text() == "2"
        assert (base_path / "metadata" / "published.txt").read_text() == "true"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify exact equality
        assert loaded_data == original_data
        assert type(loaded_data["author"]["age"]) is int
        assert type(loaded_data["metadata"]["version"]) is int
        assert type(loaded_data["metadata"]["published"]) is bool

        # Data → Filesystem again
        base_path2 = Path(tmpdir) / "data2"
        write_structure_to_filesystem(loaded_data, schema, base_path2)

        # Verify identical structure
        assert (base_path2 / "title.txt").read_text() == (base_path / "title.txt").read_text()
        assert (base_path2 / "author" / "first_name.txt").read_text() == (
            base_path / "author" / "first_name.txt"
        ).read_text()
        assert (base_path2 / "metadata" / "version.txt").read_text() == (
            base_path / "metadata" / "version.txt"
        ).read_text()


def test_array_of_objects_round_trip():
    """Test round-trip conversion with arrays of objects."""
    schema = {
        "properties": {
            "title": {"type": "string"},
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "title": {"type": "string"},
                        "pages": {"type": "integer"},
                        "completed": {"type": "boolean"},
                    },
                },
            },
        },
        "required": ["title", "chapters"],
    }

    original_data = {
        "title": "Advanced Python Guide",
        "chapters": [
            {"number": 1, "title": "Introduction", "pages": 15, "completed": True},
            {"number": 2, "title": "Getting Started", "pages": 28, "completed": True},
            {"number": 3, "title": "Advanced Topics", "pages": 42, "completed": False},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify array of objects structure
        assert (base_path / "chapters").is_dir()
        assert (base_path / "chapters" / "0000").is_dir()
        assert (base_path / "chapters" / "0001").is_dir()
        assert (base_path / "chapters" / "0002").is_dir()

        # Verify first chapter
        assert (base_path / "chapters" / "0000" / "number.txt").read_text() == "1"
        assert (base_path / "chapters" / "0000" / "title.txt").read_text() == "Introduction"
        assert (base_path / "chapters" / "0000" / "pages.txt").read_text() == "15"
        assert (base_path / "chapters" / "0000" / "completed.txt").read_text() == "true"

        # Verify second chapter
        assert (base_path / "chapters" / "0001" / "number.txt").read_text() == "2"
        assert (base_path / "chapters" / "0001" / "title.txt").read_text() == "Getting Started"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify exact equality
        assert loaded_data == original_data
        assert len(loaded_data["chapters"]) == 3
        assert all(type(ch["number"]) is int for ch in loaded_data["chapters"])
        assert all(type(ch["pages"]) is int for ch in loaded_data["chapters"])
        assert all(type(ch["completed"]) is bool for ch in loaded_data["chapters"])

        # Data → Filesystem again
        base_path2 = Path(tmpdir) / "data2"
        write_structure_to_filesystem(loaded_data, schema, base_path2)

        # Verify identical structure
        for i in range(3):
            chapter_dir = f"{i:04d}"
            assert (base_path2 / "chapters" / chapter_dir / "number.txt").read_text() == (
                base_path / "chapters" / chapter_dir / "number.txt"
            ).read_text()
            assert (base_path2 / "chapters" / chapter_dir / "title.txt").read_text() == (
                base_path / "chapters" / chapter_dir / "title.txt"
            ).read_text()


def test_complex_deeply_nested_round_trip():
    """Test round-trip conversion with complex deeply nested structures."""
    schema = {
        "properties": {
            "course_name": {"type": "string"},
            "instructor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "years_experience": {"type": "integer"},
                },
            },
            "students": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "integer"},
                        "name": {"type": "string"},
                        "grades": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "passed": {"type": "boolean"},
                    },
                },
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["course_name"],
    }

    original_data = {
        "course_name": "Machine Learning Fundamentals",
        "instructor": {
            "name": "Dr. Sarah Chen",
            "email": "sarah.chen@university.edu",
            "years_experience": 12,
        },
        "students": [
            {
                "student_id": 1001,
                "name": "Bob Wilson",
                "grades": [88.5, 92.0, 85.5],
                "passed": True,
            },
            {
                "student_id": 1002,
                "name": "Carol Davis",
                "grades": [95.0, 98.5, 94.0],
                "passed": True,
            },
            {
                "student_id": 1003,
                "name": "David Lee",
                "grades": [72.5, 68.0, 75.5],
                "passed": False,
            },
        ],
        "tags": ["machine-learning", "ai", "python", "beginner"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify complex nested structure
        assert (base_path / "course_name.txt").read_text() == "Machine Learning Fundamentals"
        assert (base_path / "instructor" / "name.txt").read_text() == "Dr. Sarah Chen"
        assert (base_path / "instructor" / "years_experience.txt").read_text() == "12"

        # Verify students array with nested grades array
        assert (base_path / "students" / "0000" / "student_id.txt").read_text() == "1001"
        assert (base_path / "students" / "0000" / "name.txt").read_text() == "Bob Wilson"
        assert (base_path / "students" / "0000" / "grades").is_dir()
        assert (base_path / "students" / "0000" / "grades" / "0000.txt").read_text() == "88.5"
        assert (base_path / "students" / "0000" / "grades" / "0001.txt").read_text() == "92.0"
        assert (base_path / "students" / "0000" / "grades" / "0002.txt").read_text() == "85.5"
        assert (base_path / "students" / "0000" / "passed.txt").read_text() == "true"

        assert (base_path / "students" / "0002" / "student_id.txt").read_text() == "1003"
        assert (base_path / "students" / "0002" / "passed.txt").read_text() == "false"

        # Verify tags
        assert (base_path / "tags" / "0000.txt").read_text() == "machine-learning"
        assert (base_path / "tags" / "0003.txt").read_text() == "beginner"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify exact equality
        assert loaded_data == original_data
        assert type(loaded_data["instructor"]["years_experience"]) is int
        assert len(loaded_data["students"]) == 3
        assert all(type(s["student_id"]) is int for s in loaded_data["students"])
        assert all(type(s["passed"]) is bool for s in loaded_data["students"])
        assert all(len(s["grades"]) == 3 for s in loaded_data["students"])
        assert all(type(g) is float for s in loaded_data["students"] for g in s["grades"])

        # Data → Filesystem again
        base_path2 = Path(tmpdir) / "data2"
        write_structure_to_filesystem(loaded_data, schema, base_path2)

        # Verify identical structure (spot check key files)
        assert (base_path2 / "course_name.txt").read_text() == (
            base_path / "course_name.txt"
        ).read_text()
        assert (base_path2 / "instructor" / "years_experience.txt").read_text() == (
            base_path / "instructor" / "years_experience.txt"
        ).read_text()
        assert (base_path2 / "students" / "0000" / "grades" / "0001.txt").read_text() == (
            base_path / "students" / "0000" / "grades" / "0001.txt"
        ).read_text()
        assert (base_path2 / "students" / "0002" / "passed.txt").read_text() == (
            base_path / "students" / "0002" / "passed.txt"
        ).read_text()


def test_integer_vs_float_preservation():
    """Test that integer vs float types are preserved correctly."""
    schema = {
        "properties": {
            "int_value": {"type": "integer"},
            "float_value": {"type": "number"},
            "int_array": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "float_array": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
        "required": [],
    }

    original_data = {
        "int_value": 42,
        "float_value": 42.0,
        "int_array": [1, 2, 3],
        "float_array": [1.0, 2.5, 3.0],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify type preservation
        assert loaded_data == original_data
        assert type(loaded_data["int_value"]) is int
        assert type(loaded_data["float_value"]) in (int, float)  # 42.0 might load as int
        assert all(type(v) is int for v in loaded_data["int_array"])
        # Float array should preserve float types even for whole numbers
        assert any(type(v) is float for v in loaded_data["float_array"])


def test_empty_arrays():
    """Test handling of empty arrays."""
    schema = {
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scores": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": [],
    }

    original_data = {
        "tags": [],
        "scores": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify empty directories are created
        assert (base_path / "tags").is_dir()
        assert (base_path / "scores").is_dir()
        assert list((base_path / "tags").glob("*.txt")) == []
        assert list((base_path / "scores").glob("*.txt")) == []

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify exact equality
        assert loaded_data == original_data
        assert loaded_data["tags"] == []
        assert loaded_data["scores"] == []


def test_optional_fields_missing_from_filesystem():
    """Test that optional fields missing from filesystem don't cause errors.

    This tests the fix for the bug where optional array/object fields
    that don't exist on the filesystem would cause RuntimeError even though
    they weren't in the schema's 'required' list.
    """
    schema = {
        "properties": {
            "needs_research": {"type": "boolean", "description": "Whether research is needed"},
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional search queries",
            },
            "rationale": {"type": "string", "description": "Explanation"},
            "metadata": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "description": "Optional metadata",
            },
            "score": {"type": "number", "description": "Optional score"},
        },
        "required": ["needs_research", "rationale"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"
        base_path.mkdir(parents=True, exist_ok=True)

        # Create filesystem with only required fields
        # (simulating Claude not creating optional fields)
        (base_path / "needs_research.txt").write_text("false")
        (base_path / "rationale.txt").write_text("This is basic math, no research needed")

        # Filesystem → Data (should NOT raise error for missing optional fields)
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify only required fields are present
        assert loaded_data == {
            "needs_research": False,
            "rationale": "This is basic math, no research needed",
        }
        # Optional fields should not be in the result dict
        assert "queries" not in loaded_data
        assert "metadata" not in loaded_data
        assert "score" not in loaded_data


def test_optional_fields_partially_present():
    """Test that some optional fields can be present while others are missing."""
    schema = {
        "properties": {
            "name": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "version": {"type": "integer"},
                },
            },
            "score": {"type": "number"},
        },
        "required": ["name"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"
        base_path.mkdir(parents=True, exist_ok=True)

        # Create filesystem with required field + some optional fields
        (base_path / "name.txt").write_text("Test")
        (base_path / "tags").mkdir()
        (base_path / "tags" / "0000.txt").write_text("tag1")
        (base_path / "tags" / "0001.txt").write_text("tag2")
        # metadata and score are missing

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify partial optional fields
        assert loaded_data == {
            "name": "Test",
            "tags": ["tag1", "tag2"],
        }
        assert "metadata" not in loaded_data
        assert "score" not in loaded_data


def test_required_fields_still_raise_errors():
    """Test that missing required fields still raise errors as expected."""
    schema = {
        "properties": {
            "name": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "tags"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"
        base_path.mkdir(parents=True, exist_ok=True)

        # Only create name, not tags (which is required)
        (base_path / "name.txt").write_text("Test")

        # Should raise RuntimeError for missing required array field
        with pytest.raises(RuntimeError, match="Missing directory.*tags"):
            read_structure_from_filesystem(schema, base_path)


def test_pydantic_generated_schema_with_ref_references():
    """Test that Pydantic-generated schemas with $ref references work correctly.

    This is a regression test for the bug where $ref references in array items
    were not resolved, causing arrays of nested models to be treated as arrays
    of strings.
    """

    class NestedModel(BaseModel):
        """A nested model with multiple fields."""
        priority: int = Field(description="Priority level (1-10)")
        action: str = Field(description="Action to take")
        details: str | None = Field(default=None, description="Optional details")

    class ParentModel(BaseModel):
        """Parent model containing a list of nested models."""
        summary: str = Field(description="Summary text")
        items: list[NestedModel] = Field(description="List of nested items")
        tags: list[str] = Field(description="Simple string list for comparison")

    # Generate schema using Pydantic (this will contain $ref references)
    schema = ParentModel.model_json_schema()

    # Verify the schema contains $ref (this is what was causing the bug)
    assert "$ref" in schema["properties"]["items"]["items"]
    assert schema["properties"]["items"]["items"]["$ref"] == "#/$defs/NestedModel"

    # Test data
    original_data = {
        "summary": "Test summary",
        "items": [
            {"priority": 1, "action": "First action", "details": "Some details"},
            {"priority": 2, "action": "Second action", "details": None},
        ],
        "tags": ["tag1", "tag2"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify nested object array structure (not primitive files!)
        assert (base_path / "items").is_dir()
        assert (base_path / "items" / "0000").is_dir()  # Should be directory, not .txt file
        assert (base_path / "items" / "0001").is_dir()  # Should be directory, not .txt file
        assert not (base_path / "items" / "0000.txt").exists()  # Should NOT be .txt file
        assert not (base_path / "items" / "0001.txt").exists()  # Should NOT be .txt file

        # Verify nested object fields
        assert (base_path / "items" / "0000" / "priority.txt").read_text() == "1"
        assert (base_path / "items" / "0000" / "action.txt").read_text() == "First action"
        assert (base_path / "items" / "0000" / "details.txt").read_text() == "Some details"

        assert (base_path / "items" / "0001" / "priority.txt").read_text() == "2"
        assert (base_path / "items" / "0001" / "action.txt").read_text() == "Second action"
        # details is None for second item, so file shouldn't exist
        assert not (base_path / "items" / "0001" / "details.txt").exists()

        # Verify primitive array structure (should be .txt files)
        assert (base_path / "tags" / "0000.txt").read_text() == "tag1"
        assert (base_path / "tags" / "0001.txt").read_text() == "tag2"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify data (note: details=None in item[1] is omitted since it's optional)
        assert len(loaded_data["items"]) == 2
        assert loaded_data["items"][0]["priority"] == 1
        assert loaded_data["items"][0]["action"] == "First action"
        assert loaded_data["items"][0]["details"] == "Some details"
        assert loaded_data["items"][1]["priority"] == 2
        assert loaded_data["items"][1]["action"] == "Second action"
        # details is optional and None, so it's omitted from result
        assert "details" not in loaded_data["items"][1]


def test_build_instructions_with_ref_references():
    """Test that build_structure_instructions correctly handles $ref references.

    This ensures the instructions tell Claude to create subdirectories for nested
    objects, not .txt files for primitives.
    """

    class NestedModel(BaseModel):
        priority: int = Field(description="Priority level")
        action: str = Field(description="Action to take")

    class ParentModel(BaseModel):
        summary: str = Field(description="Summary text")
        items: list[NestedModel] = Field(description="List of nested items")

    schema = ParentModel.model_json_schema()

    # Verify schema has $ref
    assert "$ref" in schema["properties"]["items"]["items"]

    # Generate instructions
    instructions = build_structure_instructions(schema, "/tmp/test")

    # Verify instructions mention subdirectories for nested objects
    assert "numbered subdirectories" in instructions
    assert "priority, action" in instructions  # Should list nested fields

    # Verify instructions don't incorrectly say "numbered files (.txt)" for objects
    # (This was the bug - treating nested objects as primitives)
    items_section = instructions.split("- items:")[1].split("-")[0]
    assert "0000.txt" not in items_section  # Should NOT mention .txt files for nested objects
    assert "0000/" in items_section or "subdirectories" in items_section  # Should mention subdirectories


def test_none_vs_empty_string_distinction():
    """Test that None values and empty strings are handled distinctly.

    This is critical: None = no file, empty string = empty file.
    """
    schema = {
        "properties": {
            "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "description": {"type": "string"},
            "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["name", "description"],
    }

    original_data = {
        "name": "Alice",
        "description": "",  # Empty string
        "notes": None,  # None value
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify: name has content
        assert (base_path / "name.txt").exists()
        assert (base_path / "name.txt").read_text() == "Alice"

        # Verify: description is empty file (empty string)
        assert (base_path / "description.txt").exists()
        assert (base_path / "description.txt").read_text() == ""

        # Verify: notes has no file (None)
        assert not (base_path / "notes.txt").exists()

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify distinction is preserved
        assert loaded_data["name"] == "Alice"
        assert loaded_data["description"] == ""  # Empty string preserved
        assert "notes" not in loaded_data  # None means field omitted (optional)


def test_none_in_primitive_arrays_creates_gaps():
    """Test that None values in arrays create gaps in numbering."""
    schema = {
        "properties": {
            "values": {
                "type": "array",
                "items": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
        "required": ["values"],
    }

    original_data = {
        "values": ["first", None, "third", None, "fifth"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify gaps in numbering
        assert (base_path / "values" / "0000.txt").exists()
        assert not (base_path / "values" / "0001.txt").exists()  # Gap for None
        assert (base_path / "values" / "0002.txt").exists()
        assert not (base_path / "values" / "0003.txt").exists()  # Gap for None
        assert (base_path / "values" / "0004.txt").exists()

        # Verify contents
        assert (base_path / "values" / "0000.txt").read_text() == "first"
        assert (base_path / "values" / "0002.txt").read_text() == "third"
        assert (base_path / "values" / "0004.txt").read_text() == "fifth"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify None values restored in correct positions
        assert loaded_data == original_data
        assert loaded_data["values"][0] == "first"
        assert loaded_data["values"][1] is None
        assert loaded_data["values"][2] == "third"
        assert loaded_data["values"][3] is None
        assert loaded_data["values"][4] == "fifth"


def test_none_in_object_arrays_creates_gaps():
    """Test that None values in object arrays create gaps in subdirectories."""
    schema = {
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                            },
                        },
                        {"type": "null"},
                    ]
                },
            },
        },
        "required": ["items"],
    }

    original_data = {
        "items": [
            {"id": 1, "name": "First"},
            None,  # Gap
            {"id": 3, "name": "Third"},
            None,  # Gap
            {"id": 5, "name": "Fifth"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"

        # Data → Filesystem
        write_structure_to_filesystem(original_data, schema, base_path)

        # Verify gaps in subdirectory numbering
        assert (base_path / "items" / "0000").is_dir()
        assert not (base_path / "items" / "0001").exists()  # Gap for None
        assert (base_path / "items" / "0002").is_dir()
        assert not (base_path / "items" / "0003").exists()  # Gap for None
        assert (base_path / "items" / "0004").is_dir()

        # Verify contents of existing subdirectories
        assert (base_path / "items" / "0000" / "id.txt").read_text() == "1"
        assert (base_path / "items" / "0000" / "name.txt").read_text() == "First"
        assert (base_path / "items" / "0002" / "id.txt").read_text() == "3"
        assert (base_path / "items" / "0004" / "name.txt").read_text() == "Fifth"

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # Verify None values restored in correct positions
        assert loaded_data == original_data
        assert loaded_data["items"][0] == {"id": 1, "name": "First"}
        assert loaded_data["items"][1] is None
        assert loaded_data["items"][2] == {"id": 3, "name": "Third"}
        assert loaded_data["items"][3] is None
        assert loaded_data["items"][4] == {"id": 5, "name": "Fifth"}


def test_required_nullable_field_missing_returns_none():
    """Test that required nullable fields missing from filesystem return None."""
    schema = {
        "properties": {
            "name": {"type": "string"},
            "middle_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["name", "middle_name"],  # Both required!
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data"
        base_path.mkdir(parents=True, exist_ok=True)

        # Create only name, not middle_name
        (base_path / "name.txt").write_text("Alice")
        # middle_name.txt does NOT exist

        # Filesystem → Data
        loaded_data = read_structure_from_filesystem(schema, base_path)

        # middle_name is required AND nullable, so missing file = None
        assert loaded_data == {
            "name": "Alice",
            "middle_name": None,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
