"""Type-preserving conversion between JSON schemas and file/folder structures.

This module provides utilities to convert structured data to filesystem representations
and back, maintaining exact type fidelity throughout the round-trip conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve_schema_ref(field_schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref references in JSON schema.

    Args:
        field_schema: Field schema that may contain $ref
        root_schema: Root schema containing $defs

    Returns:
        Resolved schema with $ref expanded
    """
    if "$ref" not in field_schema:
        return field_schema

    ref = field_schema["$ref"]
    # Handle references like "#/$defs/UserProfile"
    if ref.startswith("#/$defs/"):
        def_name = ref.split("/")[-1]
        defs = root_schema.get("$defs", {})
        if def_name in defs:
            return defs[def_name]

    # If we can't resolve, return original
    return field_schema


def write_structure_to_filesystem(
    data: dict[str, Any],
    schema: dict[str, Any],
    base_path: Path,
    root_schema: dict[str, Any] | None = None,
) -> None:
    """Write structured data to filesystem representation.

    Args:
        data: Data dictionary to write
        schema: JSON schema defining structure and types
        base_path: Base directory path to write to
        root_schema: Root schema for resolving $ref (defaults to schema)

    Raises:
        ValueError: If data doesn't match schema
    """
    if root_schema is None:
        root_schema = schema

    base_path.mkdir(parents=True, exist_ok=True)
    properties = schema.get("properties", {})

    for field_name, field_schema in properties.items():
        if field_name not in data:
            continue

        # Resolve $ref if present
        field_schema = _resolve_schema_ref(field_schema, root_schema)
        field_value = data[field_name]
        field_type = field_schema.get("type", "string")

        if field_type == "array":
            _write_array_field(field_name, field_value, field_schema, base_path, root_schema)
        elif field_type == "object":
            _write_object_field(field_name, field_value, field_schema, base_path, root_schema)
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
    root_schema: dict[str, Any],
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
            write_structure_to_filesystem(item, items_schema, item_dir, root_schema)
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
    root_schema: dict[str, Any],
) -> None:
    """Write object field to subdirectory."""
    object_dir = base_path / field_name
    write_structure_to_filesystem(value, field_schema, object_dir, root_schema)


def read_structure_from_filesystem(
    schema: dict[str, Any],
    base_path: Path,
    root_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read structured data from filesystem representation.

    Args:
        schema: JSON schema defining structure and types
        base_path: Base directory path to read from
        root_schema: Root schema for resolving $ref (defaults to schema)

    Returns:
        Assembled data dictionary

    Raises:
        RuntimeError: If filesystem structure doesn't match schema
    """
    if root_schema is None:
        root_schema = schema

    if not base_path.exists():
        raise RuntimeError(
            f"Working directory not found.\n"
            f"Expected: {base_path}\n"
            f"Please create it with: mkdir -p {base_path}"
        )

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    result: dict[str, Any] = {}

    for field_name, field_schema in properties.items():
        # Resolve $ref if present
        field_schema = _resolve_schema_ref(field_schema, root_schema)
        field_type = field_schema.get("type", "string")
        is_required = field_name in required_fields

        # Check if path exists for this field
        if field_type == "array":
            field_path = base_path / field_name
        elif field_type == "object":
            field_path = base_path / field_name
        else:
            field_path = base_path / f"{field_name}.txt"

        # Skip optional fields that don't exist on filesystem
        if not field_path.exists() and not is_required:
            continue

        # Read the field (will raise error if required but missing)
        if field_type == "array":
            result[field_name] = _read_array_field(field_name, field_schema, base_path, root_schema)
        elif field_type == "object":
            result[field_name] = _read_object_field(field_name, field_schema, base_path, root_schema)
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
    root_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read array of objects from numbered subdirectories.

    Args:
        array_dir: Directory containing numbered subdirectories
        items_schema: Schema for array items
        root_schema: Root schema for resolving $ref

    Returns:
        List of parsed objects
    """
    subdirs = sorted([d for d in array_dir.iterdir() if d.is_dir()])
    items: list[dict[str, Any]] = []
    for subdir in subdirs:
        item_data = read_structure_from_filesystem(items_schema, subdir, root_schema)
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
    root_schema: dict[str, Any],
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
        return _read_array_of_objects(array_dir, items_schema, root_schema)
    else:
        return _read_array_of_primitives(array_dir, item_type)


def _read_object_field(
    field_name: str,
    field_schema: dict[str, Any],
    base_path: Path,
    root_schema: dict[str, Any],
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

    return read_structure_from_filesystem(field_schema, object_dir, root_schema)


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
    field_descriptions = _build_field_descriptions(properties, schema)

    # Build example structure
    example_structure = _build_example_structure(properties, schema)

    instructions = f"""# Task: Organize Information into File Structure

## Working Directory

```bash
mkdir -p {temp_dir}
```

---

## How to Organize Information

### 1. Text Values
Write to a `.txt` file containing the text content.

**Example:**
```
name.txt contains "Alice"
```

### 2. Numbers
Write to a `.txt` file containing just the number.

**Examples:**
```
age.txt contains "25"
score.txt contains "98.5"
```

### 3. True/False Values
Write to a `.txt` file containing just `"true"` or `"false"`.

**Example:**
```
active.txt contains "true"
```

### 4. Ordered Items
Create a subfolder, then numbered files for values **OR** numbered subdirectories for items.

**Examples:**
- **For values:** `tags/0000.txt`, `tags/0001.txt`, `tags/0002.txt`
- **For items:** `chapters/0000/`, `chapters/0001/` (each directory contains its own files/subfolders)

> **IMPORTANT:** Subfolders for **ordered** items can be empty if there are no values or items to include.

### 5. Labelled Items
Create a subfolder, then create appropriately named files for values **OR** subfolders for items.

**Examples:**
- **For values:** `author/first_name.txt`, `author/last_name.txt`
- **For items:** `author/profile/`, `author/bibliography/` (each contains its own structure according to ordered items or labelled items)

> **IMPORTANT:** Subfolders for **labelled** items **CANNOT BE EMPTY** - each must contain files for values or subfolders for items.

---

## Information to Provide

{chr(10).join(field_descriptions)}

---

## Example Structure

```
{example_structure}
```

---

## Required Information

{", ".join(required_fields) if required_fields else "All fields listed above"}

---

> **CRITICAL:**
> - Read the request below carefully
> - Extract **ALL** necessary values, names, and data from the request text
> - Create the **COMPLETE** file/folder structure with ALL required information
> - Do **NOT** leave any subfolders for labelled items empty or files missing
> - Do **NOT** write any structured text formats (like JSON, YAML, etc.) - use the file/folder structure only"""

    return instructions


def _build_field_descriptions(
    properties: dict[str, Any], root_schema: dict[str, Any], prefix: str = ""
) -> list[str]:
    """Build field descriptions without JSON terminology."""
    descriptions = []

    for field_name, field_schema in properties.items():
        # Resolve $ref if present
        field_schema = _resolve_schema_ref(field_schema, root_schema)
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
                    f"- {field_path}: Subfolder containing ordered subfolders and/or values (cannot be empty). "
                    f"Create directory '{field_path}/', then numbered subdirectories (0000/, 0001/, ...) "
                    f"each containing: {nested_fields}{desc_suffix}"
                )
            else:
                type_desc = _get_type_description(item_type)
                descriptions.append(
                    f"- {field_path}: Subfolder containing ordered subfolders and/or values (cannot be empty). "
                    f"Create directory '{field_path}/', then numbered files (0000.txt, 0001.txt, ...) "
                    f"containing each {type_desc.lower()}{desc_suffix}"
                )
        elif field_type == "object":
            nested_props = field_schema.get("properties", {})
            if nested_props:
                nested_fields = ", ".join(f"{name}.txt" for name in nested_props.keys())
                descriptions.append(
                    f"- {field_path}: Subfolder containing labelled subfolders and/or values (cannot be empty). "
                    f"Create directory '{field_path}/', then create files: {nested_fields}{desc_suffix}"
                )
            else:
                descriptions.append(
                    f"- {field_path}: Subfolder containing labelled subfolders and/or values (cannot be empty). "
                    f"Create directory '{field_path}/', then create appropriately named files for each value inside{desc_suffix}"
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
    root_schema: dict[str, Any],
) -> list[str]:
    """Build example tree lines for array of objects.

    Args:
        indent_str: Indentation string for current level
        items_props: Properties of each object in the array
        root_schema: Root schema for resolving $ref

    Returns:
        List of tree lines showing example array structure
    """
    lines = []
    lines.append(f"{indent_str}    ├── [item_0]/")
    for sub_idx, (sub_name, sub_schema) in enumerate(items_props.items()):
        # Resolve $ref if present
        sub_schema = _resolve_schema_ref(sub_schema, root_schema)
        sub_desc = sub_schema.get("description", "")
        sub_is_last = sub_idx == len(items_props) - 1
        sub_branch = "└── " if sub_is_last else "├── "
        desc_comment = f"  # {sub_desc}" if sub_desc else ""
        lines.append(f"{indent_str}    │   {sub_branch}{sub_name}.txt{desc_comment}")
    lines.append(f"{indent_str}    ├── [item_1]/")
    for sub_idx, (sub_name, sub_schema) in enumerate(items_props.items()):
        # Resolve $ref if present
        sub_schema = _resolve_schema_ref(sub_schema, root_schema)
        sub_desc = sub_schema.get("description", "")
        sub_is_last = sub_idx == len(items_props) - 1
        sub_branch = "└── " if sub_is_last else "├── "
        desc_comment = f"  # {sub_desc}" if sub_desc else ""
        lines.append(f"{indent_str}    │   {sub_branch}{sub_name}.txt{desc_comment}")
    lines.append(f"{indent_str}    └── ...")
    return lines


def _build_array_of_primitives_example(indent_str: str) -> list[str]:
    """Build example tree lines for array of primitives.

    Args:
        indent_str: Indentation string for current level

    Returns:
        List of tree lines showing example array structure
    """
    return [
        f"{indent_str}    ├── [item_0].txt",
        f"{indent_str}    ├── [item_1].txt",
        f"{indent_str}    └── ...",
    ]


def _build_object_example(
    indent_str: str,
    is_last: bool,
    nested_props: dict[str, Any],
    root_schema: dict[str, Any],
) -> list[str]:
    """Build example tree lines for object field.

    Args:
        indent_str: Indentation string for current level
        is_last: Whether this is the last field at current level
        nested_props: Properties of the nested object
        root_schema: Root schema for resolving $ref

    Returns:
        List of tree lines showing example object structure
    """
    lines = []

    if nested_props:
        # Object has defined properties - show them
        for sub_idx, (sub_name, sub_schema) in enumerate(nested_props.items()):
            # Resolve $ref if present
            sub_schema = _resolve_schema_ref(sub_schema, root_schema)
            sub_desc = sub_schema.get("description", "")
            sub_is_last = sub_idx == len(nested_props) - 1
            sub_branch = "└── " if sub_is_last else "├── "
            desc_comment = f"  # {sub_desc}" if sub_desc else ""
            if is_last:
                lines.append(f"{indent_str}    {sub_branch}{sub_name}.txt{desc_comment}")
            else:
                lines.append(f"{indent_str}│   {sub_branch}{sub_name}.txt{desc_comment}")
    else:
        # Object has no defined properties - show template examples with "cannot be empty" indicator
        if is_last:
            lines.append(f"{indent_str}    ├── [value_name_extracted_from_request].txt")
            lines.append(f"{indent_str}    └── [item_name_extracted_from_request]/")
            lines.append(f"{indent_str}        └── ...")
        else:
            lines.append(f"{indent_str}│   ├── [value_name_extracted_from_request].txt")
            lines.append(f"{indent_str}│   └── [item_name_extracted_from_request]/")
            lines.append(f"{indent_str}│       └── ...")

    return lines


def _build_example_structure(
    properties: dict[str, Any], root_schema: dict[str, Any], indent: int = 0
) -> str:
    """Build example directory tree structure."""
    lines = []
    indent_str = "  " * indent

    for idx, (field_name, field_schema) in enumerate(properties.items()):
        # Resolve $ref if present
        field_schema = _resolve_schema_ref(field_schema, root_schema)
        field_type = field_schema.get("type", "string")
        field_desc = field_schema.get("description", "")
        is_last = idx == len(properties) - 1
        branch = "└── " if is_last else "├── "
        desc_comment = f"  # {field_desc}" if field_desc else ""

        if field_type == "array":
            items_schema = field_schema.get("items", {})
            item_type = items_schema.get("type", "string")

            lines.append(f"{indent_str}{branch}{field_name}/{desc_comment}")

            if item_type == "object":
                items_props = items_schema.get("properties", {})
                lines.extend(_build_array_of_objects_example(indent_str, items_props, root_schema))
            else:
                lines.extend(_build_array_of_primitives_example(indent_str))
        elif field_type == "object":
            lines.append(f"{indent_str}{branch}{field_name}/{desc_comment}")
            nested_props = field_schema.get("properties", {})
            lines.extend(_build_object_example(indent_str, is_last, nested_props, root_schema))
        else:
            lines.append(f"{indent_str}{branch}{field_name}.txt{desc_comment}")

    return "\n".join(lines)
