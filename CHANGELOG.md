# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
