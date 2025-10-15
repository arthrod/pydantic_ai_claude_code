"""Type-preserving conversion between JSON schemas and file/folder structures.

This module provides utilities to convert structured data to filesystem representations
and back, maintaining exact type fidelity throughout the round-trip conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_structure_to_filesystem(
    data: dict[str, Any],
    schema: dict[str, Any],
    base_path: Path,
) -> None:
    """Write structured data to filesystem representation.

    Args:
        data: Data dictionary to write
        schema: JSON schema defining structure and types
        base_path: Base directory path to write to

    Raises:
        ValueError: If data doesn't match schema
    """
    base_path.mkdir(parents=True, exist_ok=True)
    properties = schema.get("properties", {})

    for field_name, field_schema in properties.items():
        if field_name not in data:
            continue

        field_value = data[field_name]
        field_type = field_schema.get("type", "string")

        if field_type == "array":
            _write_array_field(field_name, field_value, field_schema, base_path)
        elif field_type == "object":
            _write_object_field(field_name, field_value, field_schema, base_path)
        else:
            _write_scalar_field(field_name, field_value, field_type, base_path)


def _write_scalar_field(
    field_name: str,
    value: Any,
    field_type: str,
    base_path: Path,
) -> None:
    """Write scalar field to .txt file."""
    file_path = base_path / f"{field_name}.txt"

    if field_type == "boolean":
        content = "true" if value else "false"
    elif field_type in ("integer", "number"):
        content = str(value)
    else:  # string
        content = str(value)

    file_path.write_text(content)


def _write_array_field(
    field_name: str,
    value: list[Any],
    field_schema: dict[str, Any],
    base_path: Path,
) -> None:
    """Write array field to directory with numbered files/subdirs."""
    array_dir = base_path / field_name
    array_dir.mkdir(parents=True, exist_ok=True)

    items_schema = field_schema.get("items", {})
    item_type = items_schema.get("type", "string")

    for idx, item in enumerate(value):
        item_name = f"{idx:04d}"

        if item_type == "object":
            # Array of objects: create numbered subdirectories
            item_dir = array_dir / item_name
            write_structure_to_filesystem(item, items_schema, item_dir)
        else:
            # Array of primitives: create numbered .txt files
            item_file = array_dir / f"{item_name}.txt"
            if item_type == "boolean":
                content = "true" if item else "false"
            elif item_type in ("integer", "number"):
                content = str(item)
            else:  # string
                content = str(item)
            item_file.write_text(content)


def _write_object_field(
    field_name: str,
    value: dict[str, Any],
    field_schema: dict[str, Any],
    base_path: Path,
) -> None:
    """Write object field to subdirectory."""
    object_dir = base_path / field_name
    write_structure_to_filesystem(value, field_schema, object_dir)


def read_structure_from_filesystem(
    schema: dict[str, Any],
    base_path: Path,
) -> dict[str, Any]:
    """Read structured data from filesystem representation.

    Args:
        schema: JSON schema defining structure and types
        base_path: Base directory path to read from

    Returns:
        Assembled data dictionary

    Raises:
        RuntimeError: If filesystem structure doesn't match schema
    """
    if not base_path.exists():
        raise RuntimeError(
            f"Working directory not found.\n"
            f"Expected: {base_path}\n"
            f"Please create it with: mkdir -p {base_path}"
        )

    properties = schema.get("properties", {})
    result: dict[str, Any] = {}

    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "string")

        if field_type == "array":
            result[field_name] = _read_array_field(field_name, field_schema, base_path)
        elif field_type == "object":
            result[field_name] = _read_object_field(field_name, field_schema, base_path)
        else:
            result[field_name] = _read_scalar_field(field_name, field_type, base_path)

    return result


def _read_scalar_field(
    field_name: str,
    field_type: str,
    base_path: Path,
) -> Any:
    """Read scalar field from .txt file."""
    file_path = base_path / f"{field_name}.txt"

    if not file_path.exists():
        type_desc = _get_type_description(field_type)
        raise RuntimeError(
            f"Missing file: {file_path}\n"
            f"Expected content: {type_desc}\n"
            f"Create the file with the appropriate content."
        )

    content = file_path.read_text().strip()

    try:
        if field_type == "integer":
            return int(content)
        elif field_type == "number":
            # Preserve integer vs float distinction
            if "." in content or "e" in content.lower():
                return float(content)
            return int(content)
        elif field_type == "boolean":
            return content.lower() in ("true", "1", "yes")
        else:  # string
            return content
    except ValueError as e:
        type_desc = _get_type_description(field_type)
        raise RuntimeError(
            f"Invalid content in file: {file_path}\n"
            f"Expected: {type_desc}\n"
            f"Found: '{content}'\n"
            f"Fix the file content to match the expected format."
        ) from e


def _read_array_of_objects(
    array_dir: Path,
    items_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read array of objects from numbered subdirectories.

    Args:
        array_dir: Directory containing numbered subdirectories
        items_schema: Schema for array items

    Returns:
        List of parsed objects
    """
    subdirs = sorted([d for d in array_dir.iterdir() if d.is_dir()])
    items: list[dict[str, Any]] = []
    for subdir in subdirs:
        item_data = read_structure_from_filesystem(items_schema, subdir)
        items.append(item_data)
    return items


def _read_array_of_primitives(
    array_dir: Path,
    item_type: str,
) -> list[Any]:
    """Read array of primitives from numbered .txt files.

    Args:
        array_dir: Directory containing numbered .txt files
        item_type: Type of array items (string, integer, number, boolean)

    Returns:
        List of parsed primitive values

    Raises:
        RuntimeError: If file content is invalid for the specified type
    """
    files = sorted(array_dir.glob("*.txt"))
    items: list[Any] = []

    for file_path in files:
        content = file_path.read_text().strip()

        try:
            if item_type == "integer":
                items.append(int(content))
            elif item_type == "number":
                if "." in content or "e" in content.lower():
                    items.append(float(content))
                else:
                    items.append(int(content))
            elif item_type == "boolean":
                items.append(content.lower() in ("true", "1", "yes"))
            else:  # string
                items.append(content)
        except ValueError as e:
            type_desc = _get_type_description(item_type)
            raise RuntimeError(
                f"Invalid content in file: {file_path}\n"
                f"Expected: {type_desc}\n"
                f"Found: '{content}'\n"
                f"Fix the file content to match the expected format."
            ) from e

    return items


def _read_array_field(
    field_name: str,
    field_schema: dict[str, Any],
    base_path: Path,
) -> list[Any]:
    """Read array field from directory with numbered files/subdirs."""
    array_dir = base_path / field_name

    if not array_dir.exists():
        items_schema = field_schema.get("items", {})
        item_type = items_schema.get("type", "string")
        if item_type == "object":
            raise RuntimeError(
                f"Missing directory: {array_dir}\n"
                f"This should contain numbered subdirectories (0000/, 0001/, etc.)\n"
                f"Create it with: mkdir -p {array_dir}"
            )
        else:
            raise RuntimeError(
                f"Missing directory: {array_dir}\n"
                f"This should contain numbered files (0000.txt, 0001.txt, etc.)\n"
                f"Create it with: mkdir -p {array_dir}"
            )

    if not array_dir.is_dir():
        raise RuntimeError(
            f"Expected directory but found file: {array_dir}\n"
            f"Remove the file and create a directory instead:\n"
            f"rm {array_dir} && mkdir -p {array_dir}"
        )

    items_schema = field_schema.get("items", {})
    item_type = items_schema.get("type", "string")

    if item_type == "object":
        return _read_array_of_objects(array_dir, items_schema)
    else:
        return _read_array_of_primitives(array_dir, item_type)


def _read_object_field(
    field_name: str,
    field_schema: dict[str, Any],
    base_path: Path,
) -> dict[str, Any]:
    """Read object field from subdirectory."""
    object_dir = base_path / field_name

    if not object_dir.exists():
        nested_props = field_schema.get("properties", {})
        fields_list = ", ".join(nested_props.keys()) if nested_props else "nested files"
        raise RuntimeError(
            f"Missing directory: {object_dir}\n"
            f"This should contain: {fields_list}\n"
            f"Create it with: mkdir -p {object_dir}"
        )

    if not object_dir.is_dir():
        raise RuntimeError(
            f"Expected directory but found file: {object_dir}\n"
            f"Remove the file and create a directory instead:\n"
            f"rm {object_dir} && mkdir -p {object_dir}"
        )

    return read_structure_from_filesystem(field_schema, object_dir)


def build_structure_instructions(schema: dict[str, Any], temp_dir: str) -> str:
    """Build human-readable instructions for creating filesystem structure.

    Args:
        schema: JSON schema defining structure
        temp_dir: Temporary directory path to use

    Returns:
        Instruction string without JSON terminology
    """
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Build field descriptions
    field_descriptions = _build_field_descriptions(properties)

    # Build example structure
    example_structure = _build_example_structure(properties)

    instructions = f"""Task: Organize your response into separate files and directories.

IMPORTANT: Create files and directories exactly as described. Do not write any structured text formats.

Working directory:
mkdir -p {temp_dir}

How to organize information:

1. Text values: Write to a .txt file
   Example: name.txt contains "Alice"

2. Numbers: Write to a .txt file (just the number)
   Example: age.txt contains "25"
   Example: score.txt contains "98.5"

3. True/false values: Write to a .txt file (just "true" or "false")
   Example: active.txt contains "true"

4. Collections of items: Create a directory, then numbered files or subdirectories
   Example for simple items: tags/0000.txt, tags/0001.txt, tags/0002.txt
   Example for complex items: chapters/0000/, chapters/0001/ (each contains files)

5. Nested groups: Create a subdirectory containing files
   Example: author/first_name.txt, author/last_name.txt

Information to provide:
{chr(10).join(field_descriptions)}

Example structure:
{example_structure}

Required information: {", ".join(required_fields) if required_fields else "all listed above"}"""

    return instructions


def _build_field_descriptions(
    properties: dict[str, Any], prefix: str = ""
) -> list[str]:
    """Build field descriptions without JSON terminology."""
    descriptions = []

    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "string")
        field_path = f"{prefix}{field_name}"
        field_desc = field_schema.get("description", "")
        desc_suffix = f" - {field_desc}" if field_desc else ""

        if field_type == "array":
            items_schema = field_schema.get("items", {})
            item_type = items_schema.get("type", "string")

            if item_type == "object":
                items_props = items_schema.get("properties", {})
                nested_fields = ", ".join(items_props.keys())
                descriptions.append(
                    f"- {field_path}: Collection of items. Create directory '{field_path}/', "
                    f"then numbered subdirectories (0000/, 0001/, ...) each containing: {nested_fields}{desc_suffix}"
                )
            else:
                type_desc = _get_type_description(item_type)
                descriptions.append(
                    f"- {field_path}: Collection of {type_desc}. Create directory '{field_path}/', "
                    f"then numbered files (0000.txt, 0001.txt, ...){desc_suffix}"
                )
        elif field_type == "object":
            nested_props = field_schema.get("properties", {})
            if nested_props:
                nested_fields = ", ".join(nested_props.keys())
                descriptions.append(
                    f"- {field_path}: Group containing: {nested_fields}. Create directory '{field_path}/', "
                    f"then a file for each item inside{desc_suffix}"
                )
            else:
                descriptions.append(
                    f"- {field_path}: Group. Create directory '{field_path}/'{desc_suffix}"
                )
        else:
            type_desc = _get_type_description(field_type)
            descriptions.append(
                f"- {field_path}: {type_desc}. Write to '{field_path}.txt'{desc_suffix}"
            )

    return descriptions


def _get_type_description(field_type: str) -> str:
    """Get human-readable type description."""
    type_map = {
        "string": "Text value",
        "integer": "Whole number",
        "number": "Numeric value",
        "boolean": "True/false value",
    }
    return type_map.get(field_type, "Value")


def _build_array_of_objects_example(
    indent_str: str,
    items_props: dict[str, Any],
) -> list[str]:
    """Build example tree lines for array of objects.

    Args:
        indent_str: Indentation string for current level
        items_props: Properties of each object in the array

    Returns:
        List of tree lines showing example array structure
    """
    lines = []
    lines.append(f"{indent_str}    ├── 0000/")
    for sub_idx, (sub_name, _) in enumerate(items_props.items()):
        sub_is_last = sub_idx == len(items_props) - 1
        sub_branch = "└── " if sub_is_last else "├── "
        lines.append(f"{indent_str}    │   {sub_branch}{sub_name}.txt")
    lines.append(f"{indent_str}    └── 0001/")
    for sub_idx, (sub_name, _) in enumerate(items_props.items()):
        sub_is_last = sub_idx == len(items_props) - 1
        sub_branch = "└── " if sub_is_last else "├── "
        lines.append(f"{indent_str}        {sub_branch}{sub_name}.txt")
    return lines


def _build_array_of_primitives_example(indent_str: str) -> list[str]:
    """Build example tree lines for array of primitives.

    Args:
        indent_str: Indentation string for current level

    Returns:
        List of tree lines showing example array structure
    """
    return [
        f"{indent_str}    ├── 0000.txt",
        f"{indent_str}    ├── 0001.txt",
        f"{indent_str}    └── 0002.txt",
    ]


def _build_object_example(
    indent_str: str,
    is_last: bool,
    nested_props: dict[str, Any],
) -> list[str]:
    """Build example tree lines for object field.

    Args:
        indent_str: Indentation string for current level
        is_last: Whether this is the last field at current level
        nested_props: Properties of the nested object

    Returns:
        List of tree lines showing example object structure
    """
    lines = []
    for sub_idx, (sub_name, _) in enumerate(nested_props.items()):
        sub_is_last = sub_idx == len(nested_props) - 1
        sub_branch = "└── " if sub_is_last else "├── "
        if is_last:
            lines.append(f"{indent_str}    {sub_branch}{sub_name}.txt")
        else:
            lines.append(f"{indent_str}│   {sub_branch}{sub_name}.txt")
    return lines


def _build_example_structure(
    properties: dict[str, Any], prefix: str = "", indent: int = 0
) -> str:
    """Build example directory tree structure."""
    lines = []
    indent_str = "  " * indent

    for idx, (field_name, field_schema) in enumerate(properties.items()):
        field_type = field_schema.get("type", "string")
        is_last = idx == len(properties) - 1
        branch = "└── " if is_last else "├── "

        if field_type == "array":
            items_schema = field_schema.get("items", {})
            item_type = items_schema.get("type", "string")

            lines.append(f"{indent_str}{branch}{field_name}/")

            if item_type == "object":
                items_props = items_schema.get("properties", {})
                lines.extend(_build_array_of_objects_example(indent_str, items_props))
            else:
                lines.extend(_build_array_of_primitives_example(indent_str))
        elif field_type == "object":
            lines.append(f"{indent_str}{branch}{field_name}/")
            nested_props = field_schema.get("properties", {})
            lines.extend(_build_object_example(indent_str, is_last, nested_props))
        else:
            lines.append(f"{indent_str}{branch}{field_name}.txt")

    return "\n".join(lines)
