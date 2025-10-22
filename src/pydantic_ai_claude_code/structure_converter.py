"""Type-preserving conversion between JSON schemas and file/folder structures.

This module provides utilities to convert structured data to filesystem representations
and back, maintaining exact type fidelity throughout the round-trip conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


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
            return cast(dict[str, Any], defs[def_name])

    # If we can't resolve, return original
    return field_schema


def _is_nullable(field_schema: dict[str, Any]) -> bool:
    """Check if a field schema allows null/None values.

    Args:
        field_schema: Field schema to check

    Returns:
        True if the field can be None/null
    """
    # Check for explicit null type
    if field_schema.get("type") == "null":
        return True

    # Check for anyOf/oneOf containing null
    for key in ("anyOf", "oneOf"):
        if key in field_schema:
            for option in field_schema[key]:
                if option.get("type") == "null":
                    return True

    return False


def _get_non_null_type(field_schema: dict[str, Any]) -> str | None:
    """Get the non-null type from a schema that may use anyOf/oneOf.

    Args:
        field_schema: Field schema

    Returns:
        The non-null type, or None if not found
    """
    # Direct type
    if "type" in field_schema and field_schema["type"] != "null":
        return cast(str, field_schema["type"])

    # Check anyOf/oneOf for non-null type
    for key in ("anyOf", "oneOf"):
        if key in field_schema:
            for option in field_schema[key]:
                if option.get("type") and option["type"] != "null":
                    return cast(str, option["type"])

    return None


def _get_non_null_schema(field_schema: dict[str, Any]) -> dict[str, Any]:
    """Get the non-null schema from a schema that may use anyOf/oneOf.

    For schemas with anyOf/oneOf containing null, this returns the non-null option.
    Otherwise returns the schema as-is.

    Args:
        field_schema: Field schema

    Returns:
        The non-null schema
    """
    # If no anyOf/oneOf, return as-is
    if "anyOf" not in field_schema and "oneOf" not in field_schema:
        return field_schema

    # Check anyOf/oneOf for non-null schema
    for key in ("anyOf", "oneOf"):
        if key in field_schema:
            for option in field_schema[key]:
                if option.get("type") != "null":
                    return cast(dict[str, Any], option)

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
    """Write scalar field to .txt file.

    None values are not written (no file created).
    """
    # None/null values -> don't create file
    if value is None:
        return

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
    items_schema = _resolve_schema_ref(items_schema, root_schema)
    # Get non-null schema (handles anyOf with null)
    items_schema_non_null = _get_non_null_schema(items_schema)
    item_type = _get_non_null_type(items_schema) or "string"

    for idx, item in enumerate(value):
        item_name = f"{idx:04d}"

        if item_type == "object":
            # Array of objects: None items don't create subdirectories (creates gaps)
            if item is None:
                continue
            item_dir = array_dir / item_name
            write_structure_to_filesystem(item, items_schema_non_null, item_dir, root_schema)
        else:
            # Array of primitives: None items don't create files (creates gaps)
            if item is None:
                continue

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
        is_nullable = _is_nullable(field_schema)

        # Check if path exists for this field
        if field_type in ("array", "object"):
            field_path = base_path / field_name
        else:
            field_path = base_path / f"{field_name}.txt"

        # Handle missing files/directories
        if not field_path.exists():
            if is_required and is_nullable:
                # Required but nullable - missing file means None
                result[field_name] = None
                continue
            elif not is_required:
                # Optional field - skip it
                continue
            # else: required and not nullable - will raise error in read functions below

        # Read the field (will raise error if required but missing and not nullable)
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
) -> list[dict[str, Any] | None]:
    """Read array of objects from numbered subdirectories.

    Missing numbered subdirectories are treated as None values (creates gaps in array).

    Args:
        array_dir: Directory containing numbered subdirectories
        items_schema: Schema for array items
        root_schema: Root schema for resolving $ref

    Returns:
        List of parsed objects (with None for missing subdirectories)
    """
    subdirs = [d for d in array_dir.iterdir() if d.is_dir()]

    # If no subdirs, return empty array
    if not subdirs:
        return []

    # Find highest index to determine array length
    max_idx = -1
    dir_map: dict[int, Path] = {}
    for subdir in subdirs:
        # Extract index from dirname (e.g., "0042" -> 42)
        try:
            idx = int(subdir.name)
            dir_map[idx] = subdir
            max_idx = max(max_idx, idx)
        except ValueError:
            # Skip directories that don't follow numbering pattern
            continue

    # Initialize array with None values
    items: list[dict[str, Any] | None] = [None] * (max_idx + 1)

    # Fill in values from existing subdirectories
    for idx, subdir in dir_map.items():
        item_data = read_structure_from_filesystem(items_schema, subdir, root_schema)
        items[idx] = item_data

    return items


def _read_array_of_primitives(
    array_dir: Path,
    item_type: str,
) -> list[Any]:
    """Read array of primitives from numbered .txt files.

    Missing numbered files are treated as None values (creates gaps in array).
    Empty file content is treated as empty string for string types.

    Args:
        array_dir: Directory containing numbered .txt files
        item_type: Type of array items (string, integer, number, boolean)

    Returns:
        List of parsed primitive values (with None for missing files)

    Raises:
        RuntimeError: If file content is invalid for the specified type
    """
    files = sorted(array_dir.glob("*.txt"))

    # If no files, return empty array
    if not files:
        return []

    # Find highest index to determine array length
    max_idx = -1
    file_map: dict[int, Path] = {}
    for file_path in files:
        # Extract index from filename (e.g., "0042.txt" -> 42)
        try:
            idx = int(file_path.stem)
            file_map[idx] = file_path
            max_idx = max(max_idx, idx)
        except ValueError:
            # Skip files that don't follow numbering pattern
            continue

    # Initialize array with None values
    items: list[Any] = [None] * (max_idx + 1)

    # Fill in values from existing files
    for idx, file_path in file_map.items():
        content = file_path.read_text().strip()

        try:
            if item_type == "integer":
                items[idx] = int(content)
            elif item_type == "number":
                if "." in content or "e" in content.lower():
                    items[idx] = float(content)
                else:
                    items[idx] = int(content)
            elif item_type == "boolean":
                items[idx] = content.lower() in ("true", "1", "yes")
            else:  # string
                items[idx] = content
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
        items_schema = _resolve_schema_ref(items_schema, root_schema)
        item_type = _get_non_null_type(items_schema) or "string"
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
    items_schema = _resolve_schema_ref(items_schema, root_schema)
    # Get non-null schema (handles anyOf with null)
    items_schema_non_null = _get_non_null_schema(items_schema)
    item_type = _get_non_null_type(items_schema) or "string"

    if item_type == "object":
        return _read_array_of_objects(array_dir, items_schema_non_null, root_schema)
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


def build_structure_instructions(
    schema: dict[str, Any],
    temp_dir: str,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> str:
    """Build human-readable instructions for creating filesystem structure.

    Args:
        schema: JSON schema defining structure
        temp_dir: Temporary directory path to use
        tool_name: Optional function/tool name (for argument collection context)
        tool_description: Optional function/tool description (for argument collection context)

    Returns:
        Instruction string without JSON terminology
    """
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Build field descriptions
    field_descriptions = _build_field_descriptions(properties, schema)

    # Build example structure
    example_structure = _build_example_structure(properties, schema)

    # Build function context section if provided
    function_context = ""
    if tool_name and tool_description:
        function_context = f"""
---

## Function Context

**You are collecting arguments for the function: `{tool_name}()`**

**Purpose:** {tool_description}

**Important:** The information you extract from the request below should be the INPUT PARAMETERS for this function, NOT the expected output or results.

---
"""

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

**Empty strings:**
```
description.txt contains "" (empty file for empty string)
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

### 4. None/Null Values
**Do NOT create a file or directory** for None/null values.

**Examples:**
- `middle_name` is None → don't create `middle_name.txt`
- `items[2]` is None in array → don't create `items/0002.txt` (skip that index)

**Important:** Empty string (`""`) is DIFFERENT from None:
- Empty string → create empty file
- None/null → don't create file at all

### 5. Ordered Items
Create a subfolder, then numbered files for values **OR** numbered subdirectories for items.

**Examples:**
- **For values:** `tags/0000.txt`, `tags/0001.txt`, `tags/0002.txt`
- **For items:** `chapters/0000/`, `chapters/0001/` (each directory contains its own files/subfolders)
- **With None values:** `tags/0000.txt`, `tags/0002.txt` (skip 0001 if that value is None)

> **IMPORTANT:** Subfolders for **ordered** items can be empty if there are no values or items to include.

### 6. Labelled Items
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
{function_context}
> **CRITICAL:**
> - Read the request below carefully
> - Extract **ALL** necessary values, names, and data from the request text
> - Create the **COMPLETE** file/folder structure with ALL required information
> - Do **NOT** leave any subfolders for labelled items empty or files missing
> - Do **NOT** write any structured text formats (like JSON, YAML, etc.) - use the file/folder structure only
> - **After creating files/folders, review your output** to ensure it's consistent, valid, and complete according to the requirements and information provided

---

## The User's Request

**Extract information FROM the following request and organize it into the file structure:**

"""

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
            items_schema = _resolve_schema_ref(items_schema, root_schema)
            items_schema_non_null = _get_non_null_schema(items_schema)
            item_type = _get_non_null_type(items_schema) or "string"

            if item_type == "object":
                items_props = items_schema_non_null.get("properties", {})
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
                nested_fields = ", ".join(f"{name}.txt" for name in nested_props)
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


def _format_field_tree_lines(
    prefix: str,
    field_name: str,
    field_schema: dict[str, Any],
    root_schema: dict[str, Any],
    is_last: bool,
) -> list[str]:
    """Format tree lines for a single field.

    Args:
        prefix: Prefix string for indentation
        field_name: Name of the field
        field_schema: Schema for the field
        root_schema: Root schema for resolving $ref
        is_last: Whether this is the last field

    Returns:
        List of formatted tree lines
    """
    lines = []
    field_desc = field_schema.get("description", "")
    field_type = field_schema.get("type", "string")
    branch = "└── " if is_last else "├── "
    desc_comment = f"  # {field_desc}" if field_desc else ""

    if field_type == "array":
        lines.append(f"{prefix}{branch}{field_name}/{desc_comment}")
        items_schema = field_schema.get("items", {})
        items_schema = _resolve_schema_ref(items_schema, root_schema)
        item_type = _get_non_null_type(items_schema) or "string"

        if item_type == "object":
            lines.append(f"{prefix}│   ├── 0000/")
            lines.append(f"{prefix}│   │   └── ...")
            lines.append(f"{prefix}{'│' if not is_last else ' '}   └── ...")
        else:
            lines.append(f"{prefix}│   ├── 0000.txt")
            lines.append(f"{prefix}{'│' if not is_last else ' '}   └── ...")
    elif field_type == "object":
        lines.append(f"{prefix}{branch}{field_name}/{desc_comment}")
        lines.append(f"{prefix}{'│' if not is_last else ' '}   └── ...")
    else:
        lines.append(f"{prefix}{branch}{field_name}.txt{desc_comment}")

    return lines


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

    # Helper function to build field lines using the common formatter
    def build_field_lines(prefix: str) -> None:
        for sub_idx, (sub_name, sub_schema) in enumerate(items_props.items()):
            sub_schema = _resolve_schema_ref(sub_schema, root_schema)
            sub_is_last = sub_idx == len(items_props) - 1
            field_lines = _format_field_tree_lines(prefix, sub_name, sub_schema, root_schema, sub_is_last)
            lines.extend(field_lines)

    lines.append(f"{indent_str}    ├── [item_0]/")
    build_field_lines(f"{indent_str}    │   ")

    lines.append(f"{indent_str}    ├── [item_1]/")
    build_field_lines(f"{indent_str}    │   ")

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
    prefix = f"{indent_str}    " if is_last else f"{indent_str}│   "

    if nested_props:
        # Object has defined properties - show them using common formatter
        for sub_idx, (sub_name, sub_schema) in enumerate(nested_props.items()):
            sub_schema = _resolve_schema_ref(sub_schema, root_schema)
            sub_is_last = sub_idx == len(nested_props) - 1
            field_lines = _format_field_tree_lines(prefix, sub_name, sub_schema, root_schema, sub_is_last)
            lines.extend(field_lines)
    # Object has no defined properties - show template examples with "cannot be empty" indicator
    elif is_last:
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
            items_schema = _resolve_schema_ref(items_schema, root_schema)
            items_schema_non_null = _get_non_null_schema(items_schema)
            item_type = _get_non_null_type(items_schema) or "string"

            lines.append(f"{indent_str}{branch}{field_name}/{desc_comment}")

            if item_type == "object":
                items_props = items_schema_non_null.get("properties", {})
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
