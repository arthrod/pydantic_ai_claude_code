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
    Convert a string to a primitive value according to a JSON Schema type.
    
    Supported target types are "integer", "number", "boolean", and "string". For
    "number", an integer is returned when the input contains no decimal point or
    exponent; otherwise a float is returned. For "boolean", the values "true",
    "1", and "yes" (case-insensitive) are interpreted as True; all other values
    are interpreted as False.
    
    Parameters:
        value (str): The string to convert.
        field_type (str): Target type; one of "integer", "number", "boolean", or "string".
    
    Returns:
        int | float | bool | str | None: The converted value, or `None` if conversion
        fails or `field_type` is not supported.
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
        field_type (str): JSON Schema type name; expected values include "string", "integer", "number", or "boolean".
    
    Returns:
        str: A short human-readable description for the given type (defaults to "Value" for unknown types).
    """
    type_map = {
        "string": "Text value",
        "integer": "Whole number",
        "number": "Numeric value",
        "boolean": "True/false value",
    }
    return type_map.get(field_type, "Value")