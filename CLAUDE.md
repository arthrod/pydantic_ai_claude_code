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
uv publish

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/
```

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
```

## Architecture

### Core Components

1. **ClaudeCodeProvider** (`provider.py`)
   - Manages configuration for Claude CLI execution
   - Handles working directory setup (including temporary workspaces)
   - Configures tool permissions, max turns, and other CLI flags
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
5. Temp directories are created with prefix `claude_prompt_*` in `/tmp/`

**Structured Output Strategy**: When Pydantic AI requests structured output (via `output_tools`):
1. Inject detailed JSON schema + example into system prompt
2. Instruct Claude to write JSON to a specific temp file path using the Write tool
3. After response, read the temp file and validate against schema
4. Return as `ToolCallPart` with the output tool's name
5. Fallback to robust JSON extraction from response text if file not created

**Tool Calling Strategy**: When Pydantic AI provides function tools:
1. Inject tool schemas and calling format into system prompt
2. Parse Claude's response for `{"type": "tool_calls", ...}` JSON structure
3. Convert to `ToolCallPart` objects with unique IDs
4. Pydantic AI handles tool execution and result formatting

**Streaming Strategy**: Uses Claude CLI's `--output-format stream-json`:
- Parses line-by-line JSON events (`text_delta`, `tool_use`, `result`)
- Yields text chunks for `stream_text()`
- Buffers structured output or tool calls for final validation

## Important Implementation Notes

- **Auto-registration**: The package registers itself on import, so users don't need to explicitly configure the provider
- **Temp workspace default**: By default, `ClaudeCodeProvider` uses `use_temp_workspace=True` to mimic cloud provider isolation
- **Permission handling**: Defaults to `dangerously_skip_permissions=True` for non-interactive use
- **Model names**: Supports short names (sonnet, opus, haiku) and full model IDs (claude-sonnet-4-5-20250929)
- **CLI execution**: All requests shell out to the `claude` CLI binary (must be installed and authenticated)
- **JSON extraction**: Uses multiple fallback strategies (file read, markdown block parsing, regex extraction, single-field wrapping) to robustly extract JSON from Claude's responses

## Testing Strategy

Tests are in `tests/` directory:
- `test_basic.py`: Basic queries, model variants (sonnet/opus/haiku), provider settings, temp workspace
- `test_structured_output.py`: Structured responses with Pydantic models
- `test_tools.py`: Custom tool calling
- `test_messages.py`: Message formatting
- `test_utils.py`: CLI utilities

All tests use `pytest` and `pytest-asyncio`. Tests make real calls to the local Claude CLI.

## Dependencies

- **pydantic-ai**: >=1.0.15 (provides Agent, Model interfaces, tool definitions)
- **pydantic**: >=2.11.9 (for Pydantic models)
- **Claude Code CLI**: Must be installed separately (see claude.com/claude-code)

## Common Debugging Tips

1. **Check CLI availability**: Run `claude --version` to ensure CLI is installed
2. **Test CLI directly**: Run `claude --print --output-format json "What is 2+2?"` to verify CLI works
3. **Enable verbose mode**: Set `verbose=True` in `ClaudeCodeProvider` to see CLI output
4. **Check prompt files**: Prompts are written to `/tmp/claude_prompt_*/prompt.md` - examine these to verify prompt formatting
5. **Check structured output files**: Structured output creates files in `/tmp/claude_structured_output_*.json`
6. **Validate JSON manually**: If structured output fails, check the temp file exists and contains valid JSON
