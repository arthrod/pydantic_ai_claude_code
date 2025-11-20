"""JSON utilities for Claude Code model.

This module provides JSON-related utility functions for parsing and cleaning
JSON content from Claude CLI responses.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def strip_markdown_code_fence(text: str) -> str:
    """Remove markdown code fence markers from text.

    Strips ```json, ```, and trailing ``` from text before parsing.

    Args:
        text: Text potentially wrapped in markdown code fences

    Returns:
        Cleaned text with code fences removed
    """
    cleaned = text.strip()

    # Remove starting code fence
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    # Remove ending code fence
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def extract_json_from_text(text: str, schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Extract JSON from text using multiple strategies.

    Tries various methods to extract valid JSON from Claude's response:
    1. Direct parsing of stripped markdown
    2. Search for JSON object pattern
    3. Single-field wrapping for simple responses

    Args:
        text: Text containing JSON
        schema: Optional schema to guide extraction

    Returns:
        Parsed JSON dict or None if extraction fails
    """
    # Strategy 1: Direct parse after stripping markdown
    cleaned = strip_markdown_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object in text
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 3: If schema has single field, try wrapping
    if schema and "properties" in schema:
        props = schema.get("properties", {})
        if len(props) == 1:
            field_name = list(props.keys())[0]
            try:
                return {field_name: cleaned}
            except Exception:
                pass

    logger.warning("Failed to extract JSON from text")
    return None
