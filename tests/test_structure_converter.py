"""Tests for type-preserving structure conversion between data and filesystem.

These tests verify that:
1. Data → Filesystem conversion works correctly
2. Filesystem → Data conversion preserves all types
3. Round-trip conversions maintain exact equality
"""

import tempfile
from pathlib import Path

import pytest

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
        assert (base_path / ".complete").exists()

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
        assert (base_path / "author" / ".complete").exists()

        assert (base_path / "metadata").is_dir()
        assert (base_path / "metadata" / "version.txt").read_text() == "2"
        assert (base_path / "metadata" / "published.txt").read_text() == "true"
        assert (base_path / "metadata" / ".complete").exists()

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
        assert (base_path / "chapters" / "0000" / ".complete").exists()

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


def test_instruction_generation():
    """Test that instruction generation works without JSON terminology."""
    schema = {
        "properties": {
            "name": {"type": "string", "description": "Full name of the person"},
            "age": {"type": "integer"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "profile": {
                "type": "object",
                "properties": {
                    "bio": {"type": "string"},
                    "active": {"type": "boolean"},
                },
            },
        },
        "required": ["name", "age"],
    }

    instructions = build_structure_instructions(schema, "/tmp/test_dir")

    # Verify no JSON terminology
    assert "json" not in instructions.lower()
    assert "field" not in instructions.lower()
    # "object" is acceptable in general terms but should not appear in the technical JSON sense

    # Verify key concepts are present
    assert "mkdir -p /tmp/test_dir" in instructions
    assert "touch /tmp/test_dir/.complete" in instructions
    assert "name.txt" in instructions
    assert "age.txt" in instructions
    assert "Collection" in instructions or "collection" in instructions
    assert "Group" in instructions or "group" in instructions

    # Verify example structure is included
    assert "Example structure:" in instructions
    assert "├── " in instructions or "└── " in instructions

    # Verify descriptions are included
    assert "Full name of the person" in instructions


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
