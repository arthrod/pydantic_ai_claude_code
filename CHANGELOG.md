# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.7] - 2025-10-14

### Changed
- **Prompt instructions**: Simplified output generation prompts to remove explicit encouragement of appending content
  - Unstructured output: Changed from multi-step "build gradually" strategy to concise "Write your answer" instruction
  - Structured output: Removed "Build content gradually using append (>>)" section, changed to "Write content to files"
  - Both prompts now focus on writing content without prescribing incremental appending
  - Claude can still use append operations if needed, but it's no longer the recommended approach

## [0.5.6] - 2025-10-14

## [0.5.5] - 2025-10-14

### Changed
- **Version management**: `__version__` now dynamically reads from package metadata via `importlib.metadata`
  - Single source of truth in `pyproject.toml` - no need to maintain version in `__init__.py`
  - Automated version bumping via `bump-my-version` tool
  - Fallback to "0.0.0.dev" for development environments where package isn't installed

## [0.5.4] - 2025-10-14

### Fixed
- **Function selection clarity**: Completely redesigned function selection prompt to eliminate confusion
  - Changed from ambiguous "Available options" to clear numbered menu with explicit task description
  - Added prominent disclaimer: "This is NOT asking you to execute these functions - you are only SELECTING"
  - Provides step-by-step instructions and concrete examples with exact response format
  - Clear separation between selection task and execution prevents Claude from looking for built-in tools
  - Significantly improved test reliability: `test_agent_tool_with_context` now passes 80% of runs vs. 0% previously

- **Markdown formatting in function selection**: Enhanced regex parsing to handle markdown bold/italic formatting
  - Now correctly parses responses like `**CHOICE: function_name**` or `*CHOICE: function_name*`
  - Strips `**`, `*`, and `_` characters around function names in CHOICE responses
  - Makes parsing more robust against Claude's natural formatting preferences

### Changed
- **Code quality improvements**: Refactored complex functions to meet strict complexity limits
  - Reduced `run_claude_sync` from 57 statements to 23 lines by extracting 6 helper functions
  - Reduced `run_claude_async` from 68 statements to 27 lines by extracting 3 helper functions
  - Reduced `run_claude_with_jq_pipeline` from 46 statements to 25 lines by extracting 3 helper functions
  - All code now passes ruff PLR0915 (≤30 statements), PLR0912 (≤12 branches), PLR0913 (≤5 arguments)
  - All code passes mypy strict type checking and pylint quality checks
  - Improved maintainability with focused, single-responsibility helper methods

## [0.5.3] - 2025-10-13

### Added
- **Automatic rate limit retry**: Claude CLI rate limits are now automatically handled with smart retry logic
  - Detects rate limit errors from CLI output using pattern matching
  - Parses reset time (e.g., "3PM", "11AM") and calculates exact wait duration
  - Automatically waits until reset time + 1-minute buffer, then retries
  - Fallback to 5-minute wait if reset time cannot be parsed
  - Configurable via new `retry_on_rate_limit` parameter (default: `True`)
  - Works for both synchronous and asynchronous execution paths
  - Perfect for long-running batch operations

- **Code quality tools**: Added comprehensive linting and type checking
  - **ruff**: Fast linting and formatting (black-compatible)
  - **mypy**: Strict static type checking
  - **pylint**: Additional code quality checks
  - All tools configured in `pyproject.toml` with project-specific settings
  - All code passes strict quality checks

### Changed
- **Unstructured output handling**: Restored file-based approach for plain text responses
  - Instructions now tell Claude to write answer to temp file using Write tool
  - Simpler and more reliable than complex behavioral guidance
  - Consistent with structured output approach
  - Prevents meta-commentary and "task complete" messages
  - All unstructured output tests passing (6/6)

### Fixed
- Fixed warning stacklevel in provider registration to show correct calling location
- Improved error handling with proper exception chaining (`raise ... from e`)
- Enhanced code style consistency with automated formatting

## [0.5.2] - 2025-10-05

### Fixed
- **Argument list size limit**: Fixed system's argument list limit (~128KB) being exceeded when using large JSON schemas for structured output or function calling
  - System prompts (including JSON schemas) are now written to prompt.md file instead of being passed as CLI arguments
  - This allows unlimited schema sizes without hitting OS argument length limits
  - Applies to both structured output schemas and function tool definitions
  - User-specified `append_system_prompt` settings are also written to prompt.md to avoid duplication

## [0.5.1] - 2025-10-05

### Fixed
- **Tool calling reliability**: Reimplemented function tool calling to use file-based structured JSON output instead of text parsing, significantly improving reliability with complex parameter types
- **Type safety**: Added proper type annotations throughout codebase to pass strict mypy type checking
  - Added `__function_call_file` to `ClaudeCodeSettings` TypedDict
  - Used `dict[str, Any]` for argument dictionaries to support multiple value types
  - Replaced `type: ignore` comments with proper `cast()` calls
- **Message formatting**: Simplified tool result presentation by formatting as plain "Context:" entries instead of "Tool Result (function_name):" to reduce confusion
- **Test coverage**: Fixed test assertions to match new implementation patterns
  - Updated message formatting tests for new context format
  - Relaxed overly strict natural language assertions in agent tests
  - Fixed tool format prompt test expectations

### Changed
- Function calls now use two-phase protocol:
  - Phase 1: No tool results present → Use structured output to get function call as JSON
  - Phase 2: Tool results present → Use context to compose natural language answer
- Tool call requests are now omitted from conversation history to prevent Claude from repeating function calls
- Removed `--verbose` flag from Claude CLI commands as it was causing issues

## [0.5.0] - 2025-10-04

### Added
- Initial public release
- Full Pydantic AI compatibility with string-based model registration
- Structured responses using Pydantic models
- Custom Python tool calling
- True streaming via Claude CLI's stream-json mode
- Session persistence and conversation context
- Support for sonnet, haiku, and opus models
- Python 3.10+ compatibility
