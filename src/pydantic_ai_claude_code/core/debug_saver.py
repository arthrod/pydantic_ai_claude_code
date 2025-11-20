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
    """
    Determine and create the debug directory when debug saving is enabled.
    
    Parameters:
        settings (dict[str, Any] | None): Configuration dict that may include the key
            "debug_save_prompts". If the value is `True`, the function uses
            "/tmp/claude_debug". If the value is a string, that string is used as
            the directory path. Falsy or missing values disable debug saving.
    
    Returns:
        Path | None: Path to the created (or existing) debug directory when enabled,
        or `None` if debug saving is disabled or `settings` is falsy.
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
    """
    Save a prompt string to the configured debug directory when debug saving is enabled.
    
    If debug saving is disabled in settings, the function returns without writing a file. When enabled, the prompt is written to a file inside the debug directory and the module-level debug counter is incremented; the file name includes a sequential counter and a timestamp.
    
    Parameters:
        prompt (str): Prompt text to save.
        settings (dict[str, Any] | None): Settings mapping; debug saving is enabled when settings["debug_save_prompts"] is truthy. If this key is True the default debug directory is used, otherwise its string value is treated as the debug directory path.
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
    """
    Save a response dictionary as a timestamped JSON file in the debug directory when debug saving is enabled.
    
    Parameters:
        response (dict[str, Any]): The response object to serialize and save.
        settings (dict[str, Any] | None): Application settings used to determine the debug directory (e.g., the `debug_save_prompts` setting).
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
    """
    Save the provided response as pretty-printed JSON to the working-file path specified in settings.
    
    If settings contains a truthy value for "__response_file_path", the response is serialized with indentation and written to that path. Failures during writing are caught and logged; this function does not raise on I/O errors.
    
    Parameters:
        response: The response dictionary to serialize and save.
        settings: Configuration dictionary; must contain "__response_file_path" with the target file path to enable saving.
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