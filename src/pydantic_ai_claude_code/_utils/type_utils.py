"""Type utilities for Claude Code model.

This module provides type conversion utilities for converting string values
to typed primitives based on JSON schema types.
"""

import logging

logger = logging.getLogger(__name__)


def convert_primitive_value(
    value: str, field_type: str
) -> int | float | bool | str | None:
    """Convert string value to typed primitive.

    Centralized type conversion used throughout the codebase for consistent
    handling of JSON schema type conversion.

    Args:
        value: String value to convert
        field_type: Target type (integer, number, boolean, string)

    Returns:
        Converted value or None if conversion fails

    Examples:
        >>> convert_primitive_value("42", "integer")
        42
        >>> convert_primitive_value("3.14", "number")
        3.14
        >>> convert_primitive_value("true", "boolean")
        True
        >>> convert_primitive_value("hello", "string")
        'hello'
    """
    try:
        if field_type == "integer":
            return int(value)
        elif field_type == "number":
            # Preserve integer vs float distinction
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        elif field_type == "boolean":
            return value.lower() in ("true", "1", "yes")
        elif field_type == "string":
            return value
    except (ValueError, AttributeError):
        pass

    return None


def get_type_description(field_type: str) -> str:
    """Get human-readable type description.

    Args:
        field_type: JSON schema type string

    Returns:
        Human-readable description
    """
    type_map = {
        "string": "Text value",
        "integer": "Whole number",
        "number": "Numeric value",
        "boolean": "True/false value",
    }
    return type_map.get(field_type, "Value")
