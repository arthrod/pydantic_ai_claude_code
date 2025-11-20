"""Type utilities for Claude Code model.

This module provides type conversion utilities for converting string values
to typed primitives based on JSON schema types.
"""

import logging

logger = logging.getLogger(__name__)


def convert_primitive_value(
    value: str, field_type: str
) -> int | float | bool | str | None:
    """
    Convert a string value to the target JSON Schema primitive.
    
    Parameters:
        value (str): The string to convert.
        field_type (str): Target JSON Schema type; supported values are "integer", "number", "boolean", and "string".
    
    Returns:
        int, float, bool, or str: The converted value corresponding to the requested type.
        None: If conversion fails or the field_type is unsupported.
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
    """
    Provide a human-readable description for a JSON Schema primitive type.
    
    Parameters:
        field_type (str): JSON Schema type name such as "string", "integer", "number", or "boolean".
    
    Returns:
        description (str): A short human-readable description for the given type; returns "Value" for unknown types.
    """
    type_map = {
        "string": "Text value",
        "integer": "Whole number",
        "number": "Numeric value",
        "boolean": "True/false value",
    }
    return type_map.get(field_type, "Value")