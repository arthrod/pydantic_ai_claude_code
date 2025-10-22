# Refactoring Plan - pydantic-ai-claude-code

**Date Started:** 2025-10-22
**Objective:** Remove duplications, redundancies, and orphaned code without losing functionality

---

## Analysis Summary

- **1 completely orphaned file** (279 lines that can be deleted)
- **~300 lines of duplicated execution logic** that could be consolidated
- **Type conversion duplication** across files
- **Small utility modules** that could be consolidated
- **One massive file** (model.py - 2,039 lines) that could benefit from restructuring

---

## Phase 1: Quick Wins (Low Risk)

### 1.1 Delete `tools.py` (279 lines)
**Status:** Ready to delete
**Risk:** Very Low
**Evidence:**
- File is never imported anywhere in codebase
- Contains legacy tool calling code (EXECUTE format, legacy JSON format)
- Completely replaced by newer two-phase function calling in model.py

**Actions:**
1. Delete `src/pydantic_ai_claude_code/tools.py`
2. Run tests to verify no breakage

### 1.2 Verify `logging_utils.py` deletion
**Status:** Already deleted (in git status)
**Actions:**
1. Confirm deletion is complete
2. Verify no remaining imports

### 1.3 Consolidate type conversion (~30 lines savings)
**Current state:**
- `model.py:1852-1877` - `_convert_primitive_value()` static method
- `structure_converter.py:314-333, 428-447` - inline type conversion

**Actions:**
1. Extract unified `convert_primitive_value()` to `utils.py`
2. Update all call sites to use centralized version
3. Run tests

---

## Phase 2: Refactoring (Medium Risk)

### 2.1 Consolidate sync/async execution (~150 lines savings)
**Status:** ✅ COMPLETED

**Current state:**
- `_try_sync_execution_with_rate_limit_retry()` (lines 722-777)
- `_try_async_execution_with_rate_limit_retry()` (lines 909-965)
- `run_claude_sync()` (lines 780-858)
- `run_claude_async()` (lines 968-1046)

**Problem:** 80% of logic is duplicated - only differs by async/await keywords

**Approach:**
1. Extract shared error handling logic to `_handle_execution_errors()`
2. Extract shared retry logic to `_execute_with_retry()`
3. Keep thin sync/async wrappers that call shared logic
4. Run tests extensively

### 2.2 Restructure Project Directory (IDE-assisted)
**Problem:** Files are too large and lack logical organization

**Current structure:**
```
src/pydantic_ai_claude_code/
├── __init__.py
├── exceptions.py
├── messages.py
├── model.py (2,039 lines - TOO BIG)
├── provider.py
├── registration.py
├── response_utils.py
├── streaming.py
├── streamed_response.py
├── structure_converter.py
├── temp_path_utils.py
├── types.py
└── utils.py (1,257 lines - LARGE)
```

**New structure:**
```
src/pydantic_ai_claude_code/
├── __init__.py              # Main exports
├── exceptions.py            # Custom exceptions
├── provider.py              # ClaudeCodeProvider
├── registration.py          # Model registration
├── types.py                 # Type definitions
├── cli/                     # CLI command building and execution
│   ├── __init__.py         # Export key functions
│   ├── commands.py         # build_claude_command(), resolve_claude_cli_path()
│   ├── execution.py        # run_claude_sync/async(), retry logic
│   └── error_handling.py   # Error detection, classification, rate limits
├── model/                   # Model implementation
│   ├── __init__.py         # Export ClaudeCodeModel
│   ├── base.py             # ClaudeCodeModel class + request methods
│   ├── prompts.py          # System prompt building
│   ├── responses.py        # Response handling and conversion
│   ├── json_extraction.py  # JSON extraction strategies
│   └── function_calling.py # Function calling logic (2-phase)
├── messages/               # Message formatting
│   ├── __init__.py         # Export message formatting functions
│   └── formatting.py       # format_messages(), format_tool_results(), etc.
├── streaming/              # Streaming support
│   ├── __init__.py         # Export streaming classes
│   ├── parser.py           # Stream JSON parsing, event handling
│   └── response.py         # StreamedResponse implementation
└── output/                 # Output handling
    ├── __init__.py         # Export output utilities
    ├── structure_converter.py  # Directory structure <-> JSON
    └── temp_paths.py       # Temp file management

# Legacy (keep for now, inline later):
├── response_utils.py       # Small helpers (15 lines - can inline)
```

---

## Detailed Restructuring Instructions

### Step 1: Create new package directories

Create these empty packages:
```bash
mkdir -p src/pydantic_ai_claude_code/cli
mkdir -p src/pydantic_ai_claude_code/model
mkdir -p src/pydantic_ai_claude_code/messages
mkdir -p src/pydantic_ai_claude_code/streaming
mkdir -p src/pydantic_ai_claude_code/output

touch src/pydantic_ai_claude_code/cli/__init__.py
touch src/pydantic_ai_claude_code/model/__init__.py
touch src/pydantic_ai_claude_code/messages/__init__.py
touch src/pydantic_ai_claude_code/streaming/__init__.py
touch src/pydantic_ai_claude_code/output/__init__.py
```

### Step 2: Split `utils.py` (1,257 lines) → `cli/` package

**cli/commands.py** (lines 1-440 of utils.py):
```python
# Functions to extract:
- convert_primitive_value()          # Keep here or move to types.py
- strip_markdown_code_fence()        # Move to output/
- create_subprocess_async()
- resolve_claude_cli_path()
- build_claude_command()
- _add_tool_permission_flags()
- _add_model_flags()
- _add_settings_flags()
- _get_next_call_subdirectory()
- _copy_additional_files()
- _determine_working_directory()
- _log_prompt_info()
- _setup_working_directory_and_prompt()
```

**cli/execution.py** (lines 603-1136 of utils.py):
```python
# Functions to extract:
- _execute_sync_command()
- _execute_async_command()
- _try_sync_execution_with_rate_limit_retry()
- _try_async_execution_with_rate_limit_retry()
- run_claude_sync()
- run_claude_async()
```

**cli/error_handling.py** (lines 119-820 of utils.py):
```python
# Functions to extract:
- _format_cli_error_message()
- detect_rate_limit()
- calculate_wait_time()
- detect_cli_infrastructure_failure()
- detect_oauth_error()
- _check_rate_limit()
- _handle_command_failure()
- _parse_json_response()
- _validate_claude_response()
- _classify_execution_error()
- _process_successful_response()
```

**Keep in utils.py or move to appropriate places:**
```python
# Streaming utilities (move to streaming/parser.py):
- parse_stream_json_line()

# Debug utilities (keep in utils.py or create debug.py):
- _debug_counter
- _get_debug_dir()
- _save_prompt_debug()
- _save_response_debug()
- _save_raw_response_to_working_dir()
```

**cli/__init__.py**:
```python
"""CLI command building and execution."""

from .commands import (
    build_claude_command,
    resolve_claude_cli_path,
)
from .execution import (
    run_claude_sync,
    run_claude_async,
)
from .error_handling import (
    detect_oauth_error,
    detect_rate_limit,
    detect_cli_infrastructure_failure,
)

__all__ = [
    "build_claude_command",
    "resolve_claude_cli_path",
    "run_claude_sync",
    "run_claude_async",
    "detect_oauth_error",
    "detect_rate_limit",
    "detect_cli_infrastructure_failure",
]
```

### Step 3: Split `model.py` (2,039 lines) → `model/` package

**model/base.py** (lines 1-150, 1900-2039):
```python
# Extract:
- ClaudeCodeModel class definition
- request() method
- request_stream() method
- name() property
```

**model/prompts.py** (lines 151-550):
```python
# Extract all prompt building:
- _build_system_prompt()
- _build_structured_output_instructions()
- _build_function_calling_instructions()
- _format_tool_schemas()
- _build_gradual_content_instructions()
```

**model/responses.py** (lines 551-1000):
```python
# Extract response handling:
- _handle_response()
- _handle_streaming_response()
- _convert_to_model_response()
- _parse_tool_calls()
- _extract_structured_output()
```

**model/json_extraction.py** (lines 1001-1250):
```python
# Extract JSON extraction strategies:
- _try_extract_json_from_file()
- _try_extract_json_from_markdown()
- _try_extract_json_from_text()
- _extract_json_with_fallbacks()
- _wrap_single_field_if_needed()
```

**model/function_calling.py** (lines 1251-1850):
```python
# Extract function calling logic:
- _prepare_function_call_request()
- _handle_function_call_phase()
- _format_tool_results_for_followup()
- _parse_function_call_response()
```

**model/__init__.py**:
```python
"""Pydantic AI model implementation for Claude Code."""

from .base import ClaudeCodeModel

__all__ = ["ClaudeCodeModel"]
```

### Step 4: Move `messages.py` → `messages/formatting.py`

**messages/__init__.py**:
```python
"""Message formatting for Claude Code."""

from .formatting import (
    format_messages,
    format_user_message,
    format_tool_results,
)

__all__ = [
    "format_messages",
    "format_user_message",
    "format_tool_results",
]
```

### Step 5: Move streaming files → `streaming/` package

**streaming/parser.py**:
- Move `streaming.py` → `streaming/parser.py`
- Add `parse_stream_json_line()` from utils.py

**streaming/response.py**:
- Move `streamed_response.py` → `streaming/response.py`

**streaming/__init__.py**:
```python
"""Streaming support for Claude Code."""

from .parser import run_claude_streaming
from .response import StreamedResponse

__all__ = ["run_claude_streaming", "StreamedResponse"]
```

### Step 6: Move output handling → `output/` package

**output/__init__.py**:
```python
"""Output handling utilities."""

from .structure_converter import (
    write_structure_to_filesystem,
    read_structure_from_filesystem,
)
from .temp_paths import get_temp_output_path

__all__ = [
    "write_structure_to_filesystem",
    "read_structure_from_filesystem",
    "get_temp_output_path",
]
```

**output/structure_converter.py**:
- Move `structure_converter.py` here
- Add `strip_markdown_code_fence()` from utils.py

**output/temp_paths.py**:
- Move `temp_path_utils.py` here

### Step 7: Update main `__init__.py`

```python
"""Pydantic AI model provider for Claude Code CLI."""

from importlib.metadata import version

from .exceptions import ClaudeOAuthError
from .model import ClaudeCodeModel
from .provider import ClaudeCodeProvider
from .types import ClaudeCodeSettings

__version__ = version("pydantic-ai-claude-code")

__all__ = [
    "ClaudeCodeModel",
    "ClaudeCodeProvider",
    "ClaudeCodeSettings",
    "ClaudeOAuthError",
    "__version__",
]
```

---

## Testing Strategy

After restructuring:

1. **Import verification**:
   ```bash
   python -c "from pydantic_ai_claude_code import ClaudeCodeProvider, ClaudeCodeModel"
   ```

2. **Run full test suite**:
   ```bash
   uv run pytest -v
   ```

3. **Check for circular imports**:
   ```bash
   python -c "import pydantic_ai_claude_code; print('OK')"
   ```

---

## Benefits

1. **Better organization**: Logical grouping by functionality
2. **Easier navigation**: Files under 500 lines each
3. **Clear responsibilities**: Each module has one job
4. **Maintainability**: Easy to find and modify code
5. **Testing**: Can test modules independently
6. **Reduced cognitive load**: Smaller files are easier to understand

---

## Notes for IDE Refactoring

- Your IDE will handle import updates automatically
- Move functions/classes, not just files
- Keep related code together (e.g., all prompt building in one file)
- Test after each major move
- Commit frequently (one package at a time)

---

## Phase 3: Polish (Optional)

### 3.1 Inline `response_utils.py` (15 lines savings)
**Status:** Optional - low impact

### 3.2 Make retry constants configurable
**Current:** Hard-coded in utils.py
**Improvement:** Add to ClaudeCodeSettings

---

## Test Strategy

After each phase:
1. Run full test suite: `uv run pytest`
2. Run specific tests for affected modules
3. Manual smoke test with examples

---

## Rollback Plan

If any phase causes issues:
1. Git revert specific commits
2. Check protocol.md for exact state before changes
3. Resume from last successful checkpoint

---

## Success Metrics

- All tests pass
- No functionality lost
- Code is more maintainable
- Line count reduced by ~450 lines
- Better separation of concerns