"""Debug saving functionality for Claude Code model.

This module provides debug saving functionality that the Claude Agent SDK
doesn't have, allowing prompts and responses to be saved for debugging purposes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Counter for sequential numbering of prompts/responses
_debug_counter = 0


def get_debug_dir(settings: dict[str, Any] | None) -> Path | None:
    """Get debug directory path if debug saving is enabled.

    Args:
        settings: Settings dict

    Returns:
        Path to debug directory or None if disabled
    """
    if not settings:
        return None

    debug_setting = settings.get("debug_save_prompts")
    if not debug_setting:
        return None

    if debug_setting is True:
        debug_dir = Path("/tmp/claude_debug")
    else:
        debug_dir = Path(str(debug_setting))

    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def save_prompt_debug(prompt: str, settings: dict[str, Any] | None) -> None:
    """Save prompt to debug file if enabled.

    Args:
        prompt: Prompt text to save
        settings: Settings dict
    """
    debug_dir = get_debug_dir(settings)
    if not debug_dir:
        return

    global _debug_counter
    _debug_counter += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_debug_counter:03d}_{timestamp}_prompt.md"
    filepath = debug_dir / filename

    filepath.write_text(prompt)
    logger.info("Saved prompt to: %s", filepath)


def save_response_debug(response: dict[str, Any], settings: dict[str, Any] | None) -> None:
    """Save response to debug file if enabled.

    Args:
        response: Claude response to save
        settings: Settings dict
    """
    debug_dir = get_debug_dir(settings)
    if not debug_dir:
        return

    global _debug_counter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_debug_counter:03d}_{timestamp}_response.json"
    filepath = debug_dir / filename

    filepath.write_text(json.dumps(response, indent=2))
    logger.info("Saved response to: %s", filepath)


def save_raw_response_to_working_dir(
    response: dict[str, Any], settings: dict[str, Any] | None
) -> None:
    """Save raw response to working directory (always-on feature).

    Args:
        response: Claude response to save
        settings: Settings dict containing __response_file_path
    """
    if not settings:
        return

    response_file = settings.get("__response_file_path")
    if not response_file:
        logger.debug("No response file path configured, skipping save")
        return

    try:
        response_path = Path(response_file)
        response_path.write_text(json.dumps(response, indent=2))
        logger.info("Saved raw response to: %s", response_path)
    except Exception as e:
        logger.warning("Failed to save raw response to working directory: %s", e)
