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
    Determine the directory path used for saving debug prompts when enabled.
    
    If the provided settings contain a truthy "debug_save_prompts" value this returns the corresponding Path and ensures the directory exists. If "debug_save_prompts" is True the path "/tmp/claude_debug" is used; if it is a string that string is used as the directory path. If settings is None or "debug_save_prompts" is missing or falsy, saving is considered disabled.
    
    Parameters:
        settings (dict[str, Any] | None): Settings mapping that may contain the "debug_save_prompts" key.
    
    Returns:
        Path | None: Path to the debug directory when saving is enabled, or `None` when disabled.
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
    Save the given prompt to a timestamped debug file when debug saving is enabled.
    
    Parameters:
        prompt: The prompt text to persist.
        settings: Configuration mapping that enables or configures debug saving. If the key
            "debug_save_prompts" is truthy, prompts are saved; if its value is a string it is
            treated as the directory path to save files, otherwise a default debug directory is used.
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
    Save a Claude response dictionary to a timestamped JSON file when debug saving is enabled.
    
    If debug saving is disabled (get_debug_dir returns None) this function does nothing. When enabled, the response is serialized as pretty-printed JSON and written into the debug directory using the filename pattern "<counter>_<YYYYMMDD_HHMMSS>_response.json" where <counter> is the current three-digit debug counter.
    
    Parameters:
        response (dict[str, Any]): Claude response object to persist.
        settings (dict[str, Any] | None): Optional settings used to determine the debug directory; if None or debug saving is disabled, no file is written.
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
    Persist the raw Claude response to a file path taken from settings.
    
    If settings is None or does not contain the "__response_file_path" key, the function returns without action. When a path is provided, the function serializes `response` as JSON (indent=2) and writes it to that path; success and failure are logged.
    
    Parameters:
        response (dict[str, Any]): The raw response object to serialize and save.
        settings (dict[str, Any] | None): Configuration dictionary that must include the "__response_file_path"
            key with the filesystem path (string) where the response should be written.
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