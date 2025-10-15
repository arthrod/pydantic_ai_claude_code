"""Demonstrate the new structure instructions for Claude CLI prompts."""

import tempfile
from pathlib import Path

from pydantic_ai_claude_code.structure_converter import (
    build_structure_instructions,
    read_structure_from_filesystem,
    write_structure_to_filesystem,
)


def show_simple_example():
    """Simple schema with scalar types."""
    print("=" * 80)
    print("EXAMPLE 1: Simple Schema with Scalar Types")
    print("=" * 80)

    schema = {
        "properties": {
            "name": {"type": "string", "description": "Full name of the person"},
            "age": {"type": "integer", "description": "Age in years"},
            "score": {"type": "number", "description": "Test score"},
            "active": {"type": "boolean", "description": "Account status"},
        },
        "required": ["name", "age"],
    }

    instructions = build_structure_instructions(schema, "/tmp/claude_data_structure_abc123")
    print(instructions)
    print("\n")


def show_array_example():
    """Schema with arrays."""
    print("=" * 80)
    print("EXAMPLE 2: Schema with Arrays")
    print("=" * 80)

    schema = {
        "properties": {
            "title": {"type": "string", "description": "Book title"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Topic tags",
            },
            "ratings": {
                "type": "array",
                "items": {"type": "number"},
                "description": "User ratings",
            },
        },
        "required": ["title"],
    }

    instructions = build_structure_instructions(schema, "/tmp/claude_data_structure_xyz789")
    print(instructions)
    print("\n")


def show_nested_object_example():
    """Schema with nested objects."""
    print("=" * 80)
    print("EXAMPLE 3: Schema with Nested Objects")
    print("=" * 80)

    schema = {
        "properties": {
            "book_title": {"type": "string"},
            "author": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "description": "Author information",
            },
        },
        "required": ["book_title"],
    }

    instructions = build_structure_instructions(schema, "/tmp/claude_data_structure_nested1")
    print(instructions)
    print("\n")


def show_array_of_objects_example():
    """Schema with array of objects."""
    print("=" * 80)
    print("EXAMPLE 4: Schema with Array of Objects (e.g., Function Arguments)")
    print("=" * 80)

    schema = {
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "operator": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
                "description": "Filter conditions",
            },
            "limit": {"type": "integer", "description": "Maximum results"},
        },
        "required": ["query"],
    }

    instructions = build_structure_instructions(schema, "/tmp/claude_data_structure_func123")
    print(instructions)
    print("\n")


def show_complex_nested_example():
    """Complex deeply nested schema."""
    print("=" * 80)
    print("EXAMPLE 5: Complex Deeply Nested Schema")
    print("=" * 80)

    schema = {
        "properties": {
            "course_name": {"type": "string"},
            "instructor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
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
                    },
                },
            },
        },
        "required": ["course_name"],
    }

    instructions = build_structure_instructions(schema, "/tmp/claude_data_structure_complex")
    print(instructions)
    print("\n")


def show_validation_errors():
    """Show error messages Claude would receive for invalid structures."""
    print("=" * 80)
    print("EXAMPLE 6: Validation Error Feedback")
    print("=" * 80)
    print("\nThese are the error messages Claude CLI receives when the structure is invalid:\n")

    schema = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "age"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Error 1: Missing completion marker
        print("ERROR CASE 1: Missing completion marker")
        print("-" * 80)
        base_path = Path(tmpdir) / "test1"
        base_path.mkdir()
        (base_path / "name.txt").write_text("Alice")
        (base_path / "age.txt").write_text("30")
        # Forgot to create .complete marker

        try:
            read_structure_from_filesystem(schema, base_path)
        except RuntimeError as e:
            print(f"Error message:\n{e}\n")

        # Error 2: Missing required field
        print("ERROR CASE 2: Missing required field 'age'")
        print("-" * 80)
        base_path = Path(tmpdir) / "test2"
        base_path.mkdir()
        (base_path / "name.txt").write_text("Bob")
        # Missing age.txt
        (base_path / ".complete").touch()

        try:
            read_structure_from_filesystem(schema, base_path)
        except RuntimeError as e:
            print(f"Error message:\n{e}\n")

        # Error 3: Invalid type (text instead of number)
        print("ERROR CASE 3: Invalid type (text instead of number)")
        print("-" * 80)
        base_path = Path(tmpdir) / "test3"
        base_path.mkdir()
        (base_path / "name.txt").write_text("Carol")
        (base_path / "age.txt").write_text("thirty")  # Should be a number!
        (base_path / ".complete").touch()

        try:
            read_structure_from_filesystem(schema, base_path)
        except (RuntimeError, ValueError) as e:
            print(f"Error message:\n{e}\n")

        # Error 4: Missing directory for array
        print("ERROR CASE 4: Missing directory for collection 'tags'")
        print("-" * 80)
        base_path = Path(tmpdir) / "test4"
        base_path.mkdir()
        (base_path / "name.txt").write_text("David")
        (base_path / "age.txt").write_text("25")
        # Should create tags/ directory, but didn't
        (base_path / ".complete").touch()

        try:
            read_structure_from_filesystem(schema, base_path)
        except RuntimeError as e:
            print(f"Error message:\n{e}\n")

        # Error 5: Created file instead of directory for object
        print("ERROR CASE 5: Created file instead of directory for nested group")
        print("-" * 80)
        schema_with_object = {
            "properties": {
                "title": {"type": "string"},
                "author": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                },
            },
            "required": ["title"],
        }

        base_path = Path(tmpdir) / "test5"
        base_path.mkdir()
        (base_path / "title.txt").write_text("Book Title")
        (base_path / "author.txt").write_text("John Doe")  # Should be a directory!
        (base_path / ".complete").touch()

        try:
            read_structure_from_filesystem(schema_with_object, base_path)
        except RuntimeError as e:
            print(f"Error message:\n{e}\n")

    print("=" * 80)
    print("KEY POINTS:")
    print("=" * 80)
    print("1. Error messages are specific and actionable")
    print("2. They tell Claude exactly what's missing or wrong")
    print("3. Claude can retry with corrections based on these messages")
    print("4. The system validates both structure and types")
    print("\n")


if __name__ == "__main__":
    show_simple_example()
    show_array_example()
    show_nested_object_example()
    show_array_of_objects_example()
    show_complex_nested_example()
    show_validation_errors()
