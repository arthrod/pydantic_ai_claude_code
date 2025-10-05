# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
