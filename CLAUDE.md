# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **pydantic-ai-claude-code**: a Pydantic AI model provider that wraps the local Claude Code CLI, enabling local execution of Claude models with all Pydantic AI features (structured responses, tool calling, streaming, multi-turn conversations).

The package acts as a bridge between Pydantic AI's model interface and the `claude` CLI tool. Users import this package and can use `Agent('claude-code:sonnet')` to run Claude locally instead of making cloud API calls.

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv)
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_basic.py

# Run specific test
uv run pytest tests/test_basic.py::test_basic_query_sync

# Run tests with output
uv run pytest -v -s
```

### Building and Publishing
```bash
# Build the package
uv build

# Publish to PyPI (production)
# Option 1: Using token from file
uv publish --token $(cat .pypi.token)

# Option 2: Using environment variable (recommended)
export UV_PUBLISH_TOKEN=$(cat .pypi.token)
uv publish

# Publish to TestPyPI
# Option 1: Using token from file
uv publish --publish-url https://test.pypi.org/legacy/ --token $(cat .pypi-test.token)

# Option 2: Using environment variable (recommended)
export UV_PUBLISH_TOKEN=$(cat .pypi-test.token)
uv publish --publish-url https://test.pypi.org/legacy/
```

**Note**: PyPI tokens are stored in `.pypi.token` (production) and `.pypi-test.token` (test) files. These are gitignored for security. The recommended approach is to use the `--token` flag or `UV_PUBLISH_TOKEN` environment variable.

### Version Management

The project uses `bump-my-version` for automated version management. Version is maintained in a single place (`pyproject.toml`) and automatically propagated.

**How it works:**
- Version is stored in `pyproject.toml` as the single source of truth
- `__init__.py` dynamically reads version from package metadata via `importlib.metadata.version()`
- `bump-my-version` automatically updates:
  - `pyproject.toml` version field
  - `CHANGELOG.md` - adds new version section with date
  - `.bumpversion.toml` - tracks current version

**Release workflow:**
```bash
# 1. Make your changes and add them to the "Unreleased" section in CHANGELOG.md

# 2. Bump the version (this creates a git commit and tag automatically)
uv run bump-my-version bump patch  # 0.5.4 -> 0.5.5
uv run bump-my-version bump minor  # 0.5.4 -> 0.6.0
uv run bump-my-version bump major  # 0.5.4 -> 1.0.0

# 3. Push to GitHub (including tags)
git push && git push --tags

# 4. Build and publish
uv build
uv publish --token $(cat .pypi.token)
```

**Development workflow (with uncommitted changes):**
```bash
# Test what changes would be made (dry run)
uv run bump-my-version bump patch --dry-run --allow-dirty --verbose

# Commit your changes first, then bump
git add .
git commit -m "Your changes"
uv run bump-my-version bump patch
```

**Important notes:**
- By default, `bump-my-version` requires a clean git working directory
- Use `--allow-dirty` flag to bypass this (useful for testing)
- The tool automatically creates a commit with message: "Bump version: X.Y.Z → X.Y.Z+1"
- It also creates a git tag in format `vX.Y.Z`
- After bumping, rebuild the package (`uv build`) for `__version__` to reflect the new version

### Running Examples
```bash
# Basic usage example
uv run python examples/basic_example.py

# Structured output example
uv run python examples/structured_example.py

# Tools and streaming example
uv run python examples/tools_and_streaming.py

# Async example
uv run python examples/async_example.py

# Advanced configuration example
uv run python examples/advanced_example.py

# Long response handling example
uv run python examples/long_response_example.py
```

## Architecture

### Core Components

1. **ClaudeCodeProvider** (`provider.py`)
   - Manages configuration for Claude CLI execution
   - Handles working directory setup (including temporary workspaces)
   - Configures tool permissions, rate limit retry, and other CLI flags
   - Provides context manager for automatic temp directory cleanup

2. **ClaudeCodeModel** (`model.py`)
   - Implements Pydantic AI's `Model` interface
   - Handles two request types:
     - `request()`: Async non-streaming requests
     - `request_stream()`: Async streaming requests
   - Converts between Pydantic AI message format and Claude CLI prompts
   - Manages three response modes:
     - **Plain text**: Direct text responses
     - **Structured output**: Instructs Claude to write JSON to a temp file, reads and validates it
     - **Tool calling**: Injects tool schemas into system prompt, parses tool call JSON from responses

3. **Message Formatting** (`messages.py`)
   - Converts Pydantic AI `ModelMessage` objects into prompts for Claude CLI
   - Handles user messages, assistant messages, tool results, and retry messages

4. **Tool Handling** (`tools.py`)
   - Formats Pydantic AI `ToolDefinition` objects into system prompt instructions
   - Parses tool call responses in format: `{"type": "tool_calls", "calls": [{"tool_name": "...", "args": {...}}]}`
   - Generates unique tool call IDs for Pydantic AI compatibility

5. **Streaming** (`streaming.py`, `streamed_response.py`)
   - Runs Claude CLI with `--output-format stream-json`
   - Parses newline-delimited JSON events (text deltas, tool use, results)
   - Implements Pydantic AI's `StreamedResponse` interface for text and structured output streaming

6. **CLI Utilities** (`utils.py`)
   - Builds Claude CLI commands with appropriate flags
   - Manages prompt file creation in working directories
   - Runs commands sync/async via subprocess
   - Parses stream-json output events

7. **Registration** (`registration.py`)
   - Auto-registers the `claude-code` model provider with Pydantic AI on import
   - Enables string-based model usage: `Agent('claude-code:sonnet')`

### Key Design Patterns

**Prompt File Strategy**: All prompts are passed to Claude CLI via files rather than command-line arguments:
1. Each execution creates or uses a working directory (temp directory if none specified)
2. Prompts are written to `prompt.md` in the working directory
3. Claude CLI is invoked with the command: `Follow the instructions in prompt.md`
4. This approach ensures consistent handling of prompts regardless of length or special characters
5. System prompts (including JSON schemas for structured output and function tools) are also written to `prompt.md` to avoid argument list size limits (~128KB)
6. User-specified `append_system_prompt` settings are included in the prompt file to avoid duplication
7. Temp directories are created with prefix `claude_prompt_*` in `/tmp/`

**Output File Strategy**: Only structured outputs require file writing:
- **Unstructured output**: Captured directly from CLI stdout (no file needed - simpler and more reliable)
- **Structured output**: Written to `/tmp/claude_structured_output_<uuid>.json` to ensure valid JSON
- **Function call output**: Written to `/tmp/claude_function_call_<uuid>.json`
- System prompt instructs Claude to write JSON to the specified file for validation
- Response is read from file and validated before returning to Pydantic AI
- Temporary files are cleaned up after reading
- Fallback to CLI response text parsing if file read fails

**Structured Output Strategy**: When Pydantic AI requests structured output (via `output_tools`):
1. Inject detailed JSON schema + example into system prompt
2. Instruct Claude to write JSON to a specific temp file path using the Write tool
3. After response, read the temp file and validate against schema
4. Return as `ToolCallPart` with the output tool's name
5. Fallback to robust JSON extraction from response text if file not created

**Tool Calling Strategy**: When Pydantic AI provides function tools, uses a two-phase protocol:
1. **Phase 1 (No tool results yet)**: Inject function schemas into system prompt and request structured JSON output with function name and arguments written to temp file
2. **Phase 2 (After tool execution)**: Format tool results as "Context:" entries and let Claude compose natural language response
3. Tool call requests are omitted from conversation history to prevent Claude from repeating calls
4. Supports both new file-based JSON format and legacy `EXECUTE: function_name(args)` text format for backwards compatibility
5. Pydantic AI handles tool execution and result formatting between phases

**Streaming Strategy**: Uses Claude CLI's `--output-format stream-json`:
- Parses line-by-line JSON events (`text_delta`, `tool_use`, `result`)
- Yields text chunks for `stream_text()`
- Buffers structured output or tool calls for final validation

**Rate Limit Strategy**: Automatically handles Claude CLI usage limits (enabled by default):
- Detects rate limit errors from CLI output (pattern: "limit reached.*resets TIME")
- Parses reset time from error message (e.g., "3PM", "11AM")
- Calculates wait time until reset (with 1-minute buffer)
- Sleeps until reset time, then automatically retries
- Fallback: Waits 5 minutes if reset time cannot be parsed
- Configurable via `retry_on_rate_limit` setting (default: True)
- Works for both sync and async execution

**Timeout Strategy**: Prevents indefinite hangs when Claude CLI takes too long (enabled by default):
- Default timeout: 15 minutes (900 seconds)
- Configurable via `timeout_seconds` setting in `ClaudeCodeSettings`
- Async: Uses `asyncio.wait_for()` with automatic process cleanup on timeout
- Sync: Uses `subprocess.run(timeout=...)` parameter
- On timeout: Raises `RuntimeError` with elapsed time and actionable suggestions
- Error messages include: prompt length, working directory, elapsed time
- Suggests breaking tasks into smaller pieces for long-running operations

**Long Response Strategy**: Handles responses that exceed typical output length limits by building content gradually:

*For Unstructured Outputs:*
1. Claude creates initial file with Write tool: `/tmp/claude_unstructured_output_<uuid>.txt`
2. Builds response incrementally using bash append operations:
   - `echo "additional content" >> file.txt`
   - `cat << 'EOF' >> file.txt ... EOF` for multi-line content
3. Avoids hitting output token limits by generating content in chunks
4. System reads final file after completion
5. Enables responses of unlimited length

*For Structured Outputs:*
1. Claude creates directory structure mirroring JSON schema: `/tmp/claude_json_fields_<uuid>/`
2. For each field in the schema:
   - **Scalar fields**: Creates `field_name.txt` file, builds content with append (`>>`)
   - **Array fields**: Creates `field_name/` directory with numbered files (`0000.txt`, `0001.txt`, etc.)
3. Content built gradually - no need to generate everything at once
4. Creates `.complete` marker file when done
5. System automatically assembles valid JSON from directory structure via `_assemble_json_from_directory()`:
   - Reads all field files in lexicographic order
   - Performs type conversion (string, integer, number, boolean)
   - Constructs proper JSON object
   - Validates against schema
6. Claude never manually writes JSON syntax - eliminates syntax errors
7. Scales to arbitrarily large responses (hundreds of array items, kilobytes of text per field)

*Benefits:*
- **No output limits**: Responses can be any size
- **No JSON errors**: System handles JSON construction
- **Natural workflow**: Claude builds content piece-by-piece naturally
- **Reliable**: Robust file operations vs. fragile JSON generation

## Important Implementation Notes

- **Auto-registration**: The package registers itself on import, so users don't need to explicitly configure the provider
- **Version management**: `__version__` is dynamically read from package metadata - single source of truth in `pyproject.toml`, managed via `bump-my-version` tool
- **Temp workspace default**: By default, `ClaudeCodeProvider` uses `use_temp_workspace=True` to mimic cloud provider isolation
- **Permission handling**: Defaults to `dangerously_skip_permissions=True` for non-interactive use
- **Rate limit retry**: Defaults to `retry_on_rate_limit=True` to automatically wait and retry when hitting usage limits
- **Model names**: Supports short names (sonnet, opus, haiku) and full model IDs (claude-sonnet-4-5-20250929)
- **CLI execution**: All requests shell out to the `claude` CLI binary (must be installed and authenticated)
- **JSON extraction**: Uses multiple fallback strategies (file read, markdown block parsing, regex extraction, single-field wrapping) to robustly extract JSON from Claude's responses
- **Unstructured output**: Captured directly from stdout without temp files for simplicity and reliability

## Testing Strategy

Tests are in `tests/` directory:
- `test_basic.py`: Basic queries, model variants (sonnet/opus/haiku), provider settings, temp workspace
- `test_structured_output.py`: Structured responses with Pydantic models
- `test_unstructured_output.py`: Plain text responses and file-based output handling
- `test_tools.py`: Custom tool calling
- `test_messages.py`: Message formatting
- `test_utils.py`: CLI utilities
- `test_long_responses.py`: JSON assembly from directory structure, gradual file building

All tests use `pytest` and `pytest-asyncio`. Tests make real calls to the local Claude CLI.

## Dependencies

- **pydantic-ai**: >=1.0.15 (provides Agent, Model interfaces, tool definitions)
- **pydantic**: >=2.11.9 (for Pydantic models)
- **Claude Code CLI**: Must be installed separately (see claude.com/claude-code)

## Logging

The package uses Python's standard logging module with loggers namespaced under `pydantic_ai_claude_code`:

- `pydantic_ai_claude_code.provider` - Provider initialization and settings
- `pydantic_ai_claude_code.model` - Model requests and response conversion
- `pydantic_ai_claude_code.utils` - CLI command building and execution
- `pydantic_ai_claude_code.streaming` - Streaming event processing
- `pydantic_ai_claude_code.messages` - Message formatting
- `pydantic_ai_claude_code.tools` - Tool formatting and parsing
- `pydantic_ai_claude_code.registration` - Model provider registration

By default, no logging is output (NullHandler). To enable logging:

```python
import logging
logging.getLogger('pydantic_ai_claude_code').setLevel(logging.DEBUG)
```

Log levels used:
- **DEBUG**: Detailed information for diagnosing problems (command execution, file operations, parsing)
- **INFO**: Confirmation that things are working (requests starting, CLI execution, registration)
- **WARNING**: Indication of potential issues (fallback strategies, missing files)
- **ERROR**: Serious problems (CLI failures, validation errors, parsing failures)

## Common Debugging Tips

1. **Check CLI availability**: Run `claude --version` to ensure CLI is installed
2. **Test CLI directly**: Run `claude --print --output-format json "What is 2+2?"` to verify CLI works
3. **Enable debug logging**: Use `logging.getLogger('pydantic_ai_claude_code').setLevel(logging.DEBUG)` to see detailed execution
4. **Check prompt files**: Prompts are written to `/tmp/claude_prompt_*/prompt.md` - examine these to verify prompt formatting
5. **Check output files** (for structured output only):
   - Structured output: `/tmp/claude_structured_output_*.json`
   - Function call output: `/tmp/claude_function_call_*.json`
   - Note: Files are cleaned up after successful reads
   - Unstructured output: No file created, captured from CLI stdout directly
6. **Validate JSON manually**: If structured output fails, check the temp file exists and contains valid JSON
7. **Fallback behavior**: If output files aren't created or can't be read, the system falls back to using CLI's direct response text
8. **Rate limit handling**: If you see "Rate limit hit. Waiting N minutes..." messages in logs, the package is automatically handling rate limits - just wait for it to complete
9. **Timeout errors**: If you get timeout errors:
   - Check error message for elapsed time and prompt length
   - Default timeout is 15 minutes (900 seconds)
   - Increase timeout via `ClaudeCodeSettings`: `{"timeout_seconds": 1800}` for 30 minutes
   - Consider breaking large tasks into smaller chunks
   - Check logs for detailed timing information at ERROR level
