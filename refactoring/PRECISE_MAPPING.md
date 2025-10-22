# Precise Function Mapping for Restructuring

**Date:** 2025-10-22
**Purpose:** Exact function-by-function mapping for IDE refactoring

---

## Part 1: Split `utils.py` → `cli/` Package

### File: `cli/commands.py`

**Purpose:** CLI command building and workspace setup

**Functions to move (14 total):**

```python
# Line 27: Type conversion (could also go in types.py or output/)
def convert_primitive_value(value: str, field_type: str) -> int | float | bool | str | None

# Line 96: Subprocess creation
async def create_subprocess_async(cmd: list[str], cwd: str | None = None) -> asyncio.subprocess.Process

# Line 139: CLI path resolution
def resolve_claude_cli_path(settings: ClaudeCodeSettings | None = None) -> str

# Line 344: Internal helper for command building
def _add_tool_permission_flags(cmd: list[str], settings: ClaudeCodeSettings) -> None

# Line 357: Internal helper for command building
def _add_model_flags(cmd: list[str], settings: ClaudeCodeSettings) -> None

# Line 372: Internal helper for command building
def _add_settings_flags(cmd: list[str], settings: ClaudeCodeSettings) -> None

# Line 399: Main command builder
def build_claude_command(
    *,
    settings: ClaudeCodeSettings | None = None,
    input_format: str = "text",
    output_format: str = "json",
) -> list[str]

# Line 443: Working directory management
def _get_next_call_subdirectory(base_dir: str) -> Path

# Line 463: File copying
def _copy_additional_files(cwd: str, additional_files: dict[str, Path]) -> None

# Line 501: Working directory determination
def _determine_working_directory(settings: ClaudeCodeSettings | None) -> str

# Line 524: Logging helper
def _log_prompt_info(prompt_file: Path, prompt: str) -> None

# Line 536: Working directory setup
def _setup_working_directory_and_prompt(
    prompt: str, settings: ClaudeCodeSettings | None
) -> str

# Line 1166: Debug directory helper
def _get_debug_dir(settings: ClaudeCodeSettings | None) -> Path | None

# Line 1191: Debug prompt saving
def _save_prompt_debug(prompt: str, settings: ClaudeCodeSettings | None) -> None
```

**Imports needed:**
```python
import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import cast

from ..types import ClaudeCodeSettings
```

**Constants to copy:**
```python
logger = logging.getLogger(__name__)
_debug_counter = 0  # Global counter for debug files
```

---

### File: `cli/error_handling.py`

**Purpose:** Error detection, classification, and handling

**Functions to move (11 total):**

```python
# Line 119: Error message formatting
def _format_cli_error_message(elapsed: float, returncode: int, stderr_text: str) -> str

# Line 185: Rate limit detection
def detect_rate_limit(error_output: str) -> tuple[bool, str | None]

# Line 207: Wait time calculation
def calculate_wait_time(reset_time_str: str) -> int

# Line 254: Infrastructure failure detection
def detect_cli_infrastructure_failure(stderr: str) -> bool

# Line 275: OAuth error detection
def detect_oauth_error(stdout: str, stderr: str) -> tuple[bool, str | None]

# Line 645: Rate limit checking
def _check_rate_limit(
    stdout_text: str, stderr_text: str, returncode: int, retry_enabled: bool
) -> tuple[bool, int]

# Line 672: Command failure handling
def _handle_command_failure(
    stdout_text: str,
    stderr_text: str,
    returncode: int,
    elapsed: float,
    prompt_len: int,
    cwd: str,
) -> None

# Line 717: JSON response parsing
def _parse_json_response(raw_stdout: str) -> ClaudeJSONResponse

# Line 743: Response validation
def _validate_claude_response(response: ClaudeJSONResponse) -> None

# Line 765: Error classification (NEW - from Phase 2.1)
def _classify_execution_error(
    stdout_text: str,
    stderr_text: str,
    returncode: int,
    elapsed: float,
    retry_enabled: bool,
    cwd: str,
) -> tuple[str, float]

# Line 823: Response processing (NEW - from Phase 2.1)
def _process_successful_response(
    stdout_text: str,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse
```

**Imports needed:**
```python
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import cast

from ..exceptions import ClaudeOAuthError
from ..types import ClaudeCodeSettings, ClaudeJSONResponse
```

**Constants to copy:**
```python
logger = logging.getLogger(__name__)
LONG_RUNTIME_THRESHOLD_SECONDS = 600  # 10 minutes
```

---

### File: `cli/execution.py`

**Purpose:** Sync/async command execution and retry logic

**Functions to move (7 total):**

```python
# Line 603: Sync command execution
def _execute_sync_command(
    cmd: list[str], cwd: str, timeout_seconds: int
) -> subprocess.CompletedProcess[str]

# Line 967: Async command execution
async def _execute_async_command(
    cmd: list[str], cwd: str, timeout_seconds: int
) -> tuple[bytes, bytes, int]

# Line 844: Sync execution with rate limit retry
def _try_sync_execution_with_rate_limit_retry(
    cmd: list[str],
    cwd: str,
    timeout_seconds: int,
    retry_enabled: bool,
    settings: ClaudeCodeSettings | None = None,
) -> tuple[ClaudeJSONResponse | None, bool]

# Line 1015: Async execution with rate limit retry
async def _try_async_execution_with_rate_limit_retry(
    cmd: list[str],
    cwd: str,
    timeout_seconds: int,
    retry_enabled: bool,
    settings: ClaudeCodeSettings | None = None,
) -> tuple[ClaudeJSONResponse | None, bool]

# Line 886: Main sync execution
def run_claude_sync(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse

# Line 1058: Main async execution
async def run_claude_async(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse

# Line 1213: Response debug saving
def _save_response_debug(response: ClaudeJSONResponse, settings: ClaudeCodeSettings | None) -> None
```

**Imports needed:**
```python
import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import cast

from ..types import ClaudeCodeSettings, ClaudeJSONResponse
from .commands import build_claude_command, _setup_working_directory_and_prompt, _get_debug_dir
from .error_handling import (
    _classify_execution_error,
    _process_successful_response,
    detect_cli_infrastructure_failure,
)
```

**Constants to copy:**
```python
logger = logging.getLogger(__name__)
MAX_CLI_RETRIES = 3
RETRY_BACKOFF_BASE = 2
_debug_counter = 0  # Share with commands.py
```

---

### File: `cli/__init__.py`

**Public API exports:**

```python
"""CLI command building and execution."""

from .commands import (
    build_claude_command,
    resolve_claude_cli_path,
    convert_primitive_value,  # If kept here
)
from .error_handling import (
    detect_oauth_error,
    detect_rate_limit,
    detect_cli_infrastructure_failure,
)
from .execution import (
    run_claude_sync,
    run_claude_async,
)

__all__ = [
    "build_claude_command",
    "resolve_claude_cli_path",
    "convert_primitive_value",
    "detect_oauth_error",
    "detect_rate_limit",
    "detect_cli_infrastructure_failure",
    "run_claude_sync",
    "run_claude_async",
]
```

---

### What stays in `utils.py` (or moves elsewhere)

**Move to `streaming/parser.py`:**
```python
# Line 1139: Stream JSON parsing
def parse_stream_json_line(line: str) -> ClaudeStreamEvent | None
```

**Move to `output/structure_converter.py`:**
```python
# Line 70: Markdown fence stripping
def strip_markdown_code_fence(text: str) -> str
```

**Move to new `cli/debug.py` OR keep in execution.py:**
```python
# Line 1234: Save raw response to working directory
def _save_raw_response_to_working_dir(
    response: ClaudeJSONResponse, settings: ClaudeCodeSettings | None
) -> None
```

**DELETE `utils.py` after moving all functions**

---

## Part 2: Split `model.py` → `model/` Package

### File: `model/base.py`

**Purpose:** Main ClaudeCodeModel class and request methods

**Class and methods to keep (9 total):**

```python
# Line 54: Main class
class ClaudeCodeModel(Model):

    # Line 65: Constructor
    def __init__(self, model_name: str, *, settings: ClaudeCodeSettings | None = None)

    # Line 84: Settings preservation
    def _preserve_user_settings(self, settings: ClaudeCodeSettings) -> ClaudeCodeSettings

    # Line 106: Property
    def model_name(self) -> str

    # Line 533: Working directory prep
    def _prepare_working_directory(self, settings: ClaudeCodeSettings) -> None

    # Line 1139: Main request method
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> ModelResponse

    # Line 1223: Streaming request method
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[StreamResponse]

    # Line 1606: Cleanup helper
    def _cleanup_temp_file(self, file_path: str | Path) -> None

    # Line 995: Request execution helper
    async def _run_and_log_claude_request(
        self, prompt: str, settings: ClaudeCodeSettings
    ) -> ClaudeJSONResponse
```

**Imports will include:**
```python
from pydantic_ai.models import Model, ModelResponse, StreamResponse
from pydantic_ai.messages import ModelMessage, ModelSettings
from ..types import ClaudeCodeSettings, ClaudeJSONResponse
from .prompts import PromptBuilder
from .responses import ResponseHandler
from .function_calling import FunctionCallHandler
from ..cli import run_claude_async
```

---

### File: `model/prompts.py`

**Purpose:** System prompt building

**Methods to extract (10 total):**

```python
class PromptBuilder:
    """Handles system prompt construction for different request types."""

    # Line 111: System property getter
    @staticmethod
    def get_system_prompt(model: ClaudeCodeModel) -> str

    # Line 115: Unstructured output instructions
    @staticmethod
    def build_unstructured_output_instruction(
        output_file: str, schema: dict[str, Any]
    ) -> str

    # Line 168: Structured output instructions
    @staticmethod
    def build_structured_output_instruction(
        output_file: str, schema: dict[str, Any], tool_name: str
    ) -> str

    # Line 217: XML to markdown converter
    @staticmethod
    def _xml_to_markdown(xml_text: str) -> str

    # Line 247: Function option descriptions
    @staticmethod
    def build_function_option_descriptions(tools: list[ToolDefinition]) -> str

    # Line 283: Function tools prompt
    @staticmethod
    def build_function_tools_prompt(
        tools: list[ToolDefinition],
        output_file: str,
        has_tool_results: bool,
    ) -> str

    # Line 362: System prompt parts builder
    @staticmethod
    def build_system_prompt_parts(
        model: ClaudeCodeModel,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> list[str]

    # Line 445: Final prompt assembly
    @staticmethod
    def assemble_final_prompt(
        system_parts: list[str],
        messages: list[ModelMessage],
    ) -> str

    # Line 683: Retry prompt building
    @staticmethod
    def build_retry_prompt(
        previous_result: str, error_message: str, attempt: int
    ) -> str

    # Line 1012: Argument collection instruction
    @staticmethod
    def build_argument_collection_instruction(
        tool_name: str, function_schema: dict[str, Any], output_file: str
    ) -> str
```

---

### File: `model/responses.py`

**Purpose:** Response handling and conversion

**Methods to extract (7 total):**

```python
class ResponseHandler:
    """Handles response parsing and conversion."""

    # Line 510: Create model response with usage
    @staticmethod
    def create_model_response_with_usage(
        content: str | list[ModelResponsePart],
        response: ClaudeJSONResponse,
    ) -> ModelResponse

    # Line 1568: Response conversion
    @staticmethod
    def convert_response(
        model: ClaudeCodeModel,
        response: ClaudeJSONResponse,
        model_settings: ModelSettings | None,
    ) -> ModelResponse

    # Line 1333: Function selection response handler
    @staticmethod
    def handle_function_selection_response(
        model: ClaudeCodeModel,
        response: ClaudeJSONResponse,
        tools: list[ToolDefinition],
    ) -> ModelResponse

    # Line 1432: Structured output response handler
    @staticmethod
    def handle_structured_output_response(
        model: ClaudeCodeModel,
        response: ClaudeJSONResponse,
        output_file: str,
        tool_name: str,
        schema: dict[str, Any],
    ) -> ModelResponse

    # Line 1510: Unstructured output response handler
    @staticmethod
    def handle_unstructured_output_response(
        model: ClaudeCodeModel,
        response: ClaudeJSONResponse,
        output_file: str,
        schema: dict[str, Any],
    ) -> ModelResponse

    # Line 1952: Get model name from response
    @staticmethod
    def get_model_name(response: ClaudeJSONResponse) -> str

    # Line 1970: Create usage from response
    @staticmethod
    def create_usage(response: ClaudeJSONResponse) -> RequestUsage
```

---

### File: `model/json_extraction.py`

**Purpose:** JSON extraction strategies

**Methods to extract (9 total):**

```python
class JSONExtractor:
    """Handles JSON extraction from various sources."""

    # Line 1615: JSON schema validation
    @staticmethod
    def validate_json_schema(
        data: dict[str, Any], schema: dict[str, Any], source: str
    ) -> None

    # Line 1664: Try read directory structure
    @staticmethod
    def try_read_directory_structure(
        output_file: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None

    # Line 1704: Try read JSON file
    @staticmethod
    def try_read_json_file(output_file: str) -> dict[str, Any] | None

    # Line 1754: Read structured output file (main entry)
    @staticmethod
    def read_structured_output_file(
        model: ClaudeCodeModel, output_file: str, schema: dict[str, Any]
    ) -> dict[str, Any]

    # Line 1778: Extract from markdown
    @staticmethod
    def try_extract_from_markdown(text: str) -> dict[str, Any] | None

    # Line 1799: Extract JSON object
    @staticmethod
    def try_extract_json_object(text: str) -> dict[str, Any] | None

    # Line 1823: Extract JSON array
    @staticmethod
    def try_extract_json_array(text: str) -> list[Any] | None

    # Line 1854: Single field autowrap
    @staticmethod
    def try_single_field_autowrap(
        text: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None

    # Line 1917: Robust JSON extraction (main fallback)
    @staticmethod
    def extract_json_robust(
        model: ClaudeCodeModel, text: str, schema: dict[str, Any]
    ) -> dict[str, Any]
```

---

### File: `model/function_calling.py`

**Purpose:** Function calling logic (two-phase protocol)

**Methods to extract (9 total):**

```python
class FunctionCallHandler:
    """Handles function calling protocol."""

    # Line 200: Check for tool results
    @staticmethod
    def check_has_tool_results(messages: list[ModelMessage]) -> bool

    # Line 546: Handle structured follow-up
    @staticmethod
    async def handle_structured_follow_up(
        model: ClaudeCodeModel,
        messages: list[ModelMessage],
        settings: ClaudeCodeSettings,
    ) -> ClaudeJSONResponse

    # Line 619: Handle unstructured follow-up
    @staticmethod
    async def handle_unstructured_follow_up(
        model: ClaudeCodeModel,
        messages: list[ModelMessage],
        settings: ClaudeCodeSettings,
    ) -> ClaudeJSONResponse

    # Line 735: Try collect arguments
    @staticmethod
    async def try_collect_arguments(
        model: ClaudeCodeModel,
        tool_name: str,
        function_schema: dict[str, Any],
        messages: list[ModelMessage],
        settings: ClaudeCodeSettings,
    ) -> tuple[dict[str, Any] | None, ClaudeJSONResponse]

    # Line 783: Setup argument collection
    @staticmethod
    def setup_argument_collection(
        model: ClaudeCodeModel, tool_name: str, function_schema: dict[str, Any]
    ) -> tuple[str, str]

    # Line 864: Log argument collection attempt
    @staticmethod
    def log_argument_collection_attempt(attempt: int, max_attempts: int, tool_name: str) -> None

    # Line 883: Log argument collection result
    @staticmethod
    def log_argument_collection_result(success: bool, tool_name: str) -> None

    # Line 900: Handle argument collection (main logic)
    @staticmethod
    async def handle_argument_collection(
        model: ClaudeCodeModel,
        tool_name: str,
        function_schema: dict[str, Any],
        messages: list[ModelMessage],
        settings: ClaudeCodeSettings,
    ) -> tuple[dict[str, Any], ClaudeJSONResponse]

    # Line 1051: Handle function selection followup
    @staticmethod
    async def handle_function_selection_followup(
        model: ClaudeCodeModel,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> ModelResponse
```

---

### File: `model/__init__.py`

```python
"""Pydantic AI model implementation for Claude Code."""

from .base import ClaudeCodeModel

__all__ = ["ClaudeCodeModel"]
```

---

## Part 3: Move Other Files

### `messages.py` → `messages/formatting.py`

**Just move the entire file**, then create:

**messages/__init__.py:**
```python
"""Message formatting for Claude Code."""

from .formatting import format_messages

__all__ = ["format_messages"]
```

---

### `streaming.py` → `streaming/parser.py`

**Move the entire file PLUS add:**
```python
# From utils.py line 1139:
def parse_stream_json_line(line: str) -> ClaudeStreamEvent | None
```

---

### `streamed_response.py` → `streaming/response.py`

**Just move the entire file**

**streaming/__init__.py:**
```python
"""Streaming support for Claude Code."""

from .parser import run_claude_streaming
from .response import StreamedResponse

__all__ = ["run_claude_streaming", "StreamedResponse"]
```

---

### `structure_converter.py` → `output/structure_converter.py`

**Move the entire file PLUS add:**
```python
# From utils.py line 70:
def strip_markdown_code_fence(text: str) -> str
```

---

### `temp_path_utils.py` → `output/temp_paths.py`

**Just move the entire file**

**output/__init__.py:**
```python
"""Output handling utilities."""

from .structure_converter import (
    write_structure_to_filesystem,
    read_structure_from_filesystem,
    strip_markdown_code_fence,
)
from .temp_paths import get_temp_output_path

__all__ = [
    "write_structure_to_filesystem",
    "read_structure_from_filesystem",
    "strip_markdown_code_fence",
    "get_temp_output_path",
]
```

---

## Summary: Files to Create

**New files (5 packages, 13 new files):**
1. `cli/__init__.py`
2. `cli/commands.py` - 14 functions from utils.py
3. `cli/error_handling.py` - 11 functions from utils.py
4. `cli/execution.py` - 7 functions from utils.py
5. `model/__init__.py`
6. `model/base.py` - 9 methods from ClaudeCodeModel
7. `model/prompts.py` - 10 methods → PromptBuilder class
8. `model/responses.py` - 7 methods → ResponseHandler class
9. `model/json_extraction.py` - 9 methods → JSONExtractor class
10. `model/function_calling.py` - 9 methods → FunctionCallHandler class
11. `messages/__init__.py`
12. `streaming/__init__.py`
13. `output/__init__.py`

**Files to move:**
1. `messages.py` → `messages/formatting.py`
2. `streaming.py` → `streaming/parser.py` (+ parse_stream_json_line)
3. `streamed_response.py` → `streaming/response.py`
4. `structure_converter.py` → `output/structure_converter.py` (+ strip_markdown_code_fence)
5. `temp_path_utils.py` → `output/temp_paths.py`

**Files to delete:**
1. `utils.py` (after extracting all functions)
2. `model.py` (after extracting all methods)

**Files unchanged:**
- `__init__.py` (update imports)
- `exceptions.py`
- `provider.py`
- `registration.py`
- `types.py`
- `response_utils.py` (can inline later)