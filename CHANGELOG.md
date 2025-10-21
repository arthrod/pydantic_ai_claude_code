# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Working directory file overwrites**: Fixed issue where multiple CLI calls could overwrite files in the same working directory
  - Implemented numbered subdirectories (1/, 2/, 3/) for all working directories (both temp and user-specified)
  - Pre-determine working directory before creating tool result files or binary content files
  - Ensures Claude runs from the same directory where files are created
  - Prevents race conditions and file conflicts during multi-phase operations (tool selection → argument collection → final response)
  - Added `__temp_base_directory` internal setting to track base directory for numbered subdirectories
  - Added comprehensive tests: `test_temp_workspace_no_overwrite()` and `test_reused_settings_dict_no_overwrite()`

### Changed
- **Type safety improvements**: Added proper TypedDict field definition for `__temp_base_directory`
  - No `type: ignore` comments used
  - Full mypy compliance with proper type narrowing using `isinstance()` assertions

## [0.7.1] - 2025-10-19

### Fixed
- **Structured output with tools**: Fixed bug where agents with both `output_type` and `tools` would fail when the model chose not to call any tool (function selection = "none")
  - Previously made unstructured follow-up request, ignoring `output_type` setting
  - Now correctly makes structured follow-up request to get the typed output
  - Added new `_handle_structured_follow_up()` method to handle this case
  - Fixed `_build_system_prompt_parts()` to add output instructions after tool results

- **Control flow reliability**: Replaced fragile string-based control flow with settings-based approach
  - Eliminated `if "[Function selection: none" in content:` pattern
  - Now uses `__function_selection_result__` setting key ("none" or "selected")
  - More reliable and maintainable state management

### Changed
- **Code quality improvements**: Refactored code to meet linting standards
  - Extracted helper methods to reduce function complexity (PLR0915, PLR0912)
  - `_build_function_tools_prompt()`: Extracted `_xml_to_markdown()` and `_build_function_option_descriptions()`
  - `model.py request()`: Extracted `_handle_function_selection_followup()`
  - `messages.py`: Extracted `_process_user_prompt_part()` and `_process_tool_return_part()`
  - All functions now under 30 statements and 12 branches

- **Test improvements**: Made streaming behavior test more robust
  - Changed prompt from "Count from 1 to 5" to generating paragraphs about renewable energy
  - Removed fragile "read" keyword assertion that was testing Claude's wording style
  - Focused on core requirement: internal file references like "prompt.md" shouldn't leak

- **Binary content test updates**: Updated test assertions to match current @ syntax implementation
  - Changed from `[Image: filename]` format to `@filename` format
  - Aligns with commit 996ca2c implementation

## [0.7.0] - 2025-10-19

### Changed
- **Working directory improvements**: Consolidated all temporary files to use working directory instead of /tmp
  - Working directory is now pre-determined early in the request flow for consistent usage
  - All temp files (structured output, unstructured output, tool results, data structures) created in working directory
  - Added `__working_directory` internal setting to track the determined path across all phases
  - Prevents scattered temp files across filesystem - everything in one organized location
  - Better cleanup and debugging when all files are co-located with prompt.md

- **Tool result file handling**: Simplified tool result storage by writing directly to working directory
  - Changed from temporary files in /tmp to files in working directory alongside prompt.md
  - Removed complex additional_files merging logic - tool results now part of working directory
  - Tool result files named `tool_result_{counter}_{tool_name}.txt` for clarity
  - Streamlined message formatting by eliminating the tuple return value (no longer need separate file dict)
  - Improved traceability: all request artifacts (prompt, response, tool results) in same directory

- **Code organization**: Extracted common utilities into focused modules for better maintainability
  - Created `logging_utils.py` for standardized logging patterns (`log_section_separator`, `log_section_end`)
  - Created `response_utils.py` for response construction helpers (`create_tool_call_part`, `extract_model_parameters`, `get_working_directory`)
  - Created `temp_path_utils.py` for path generation (`generate_output_file_path`, `generate_temp_directory_path`)
  - Reduced code duplication across model.py, utils.py, and messages.py
  - Better separation of concerns with focused, single-purpose utility functions

- **Argument collection improvements**: Enhanced function calling with better context
  - Argument collection instructions now include function name and description
  - Added `__tool_name` and `__tool_description` to internal settings for retry attempts
  - Function context stored during initial setup and reused in retry prompts
  - Helps Claude understand the purpose of arguments being collected

- **Type safety improvements**: Added new TypedDict for model settings
  - New `ClaudeCodeModelSettings` extends Pydantic AI's `ModelSettings`
  - Includes Claude Code specific fields: `working_directory`, `additional_files`
  - Provides type hints for model_settings parameter in `agent.run_sync()`
  - Better IDE support and type checking when passing model settings

### Added
- **Web search support**: Added DuckDuckGo search capability via pydantic-ai-slim
  - Added `pydantic-ai-slim[duckduckgo]>=1.0.15` dependency
  - Enables use of Pydantic AI's `WebSearchTool` with Claude Code
  - New examples: `websearch_example.py` demonstrates 5 web search patterns
  - Supports search context sizing, domain filtering, and usage limits

- **Utility functions**: Added helper functions to improve code quality
  - `strip_markdown_code_fence()` in utils.py for cleaning code fence markers
  - `create_subprocess_async()` in utils.py for standardized async subprocess creation
  - `_format_cli_error_message()` in utils.py for consistent error formatting
  - `_determine_working_directory()` in utils.py for early directory path determination
  - `_log_prompt_info()` in utils.py for standardized prompt logging

## [0.6.0] - 2025-10-19

### Added
- **OAuth error handling**: New `ClaudeOAuthError` exception for graceful handling of OAuth token expiration in long-running sessions
  - Specific exception type (inherits from `RuntimeError`) allows targeted error handling
  - Includes `reauth_instruction` attribute with user-facing guidance (default: "Please run /login")
  - Automatically raised when Claude CLI reports OAuth token expired/revoked errors
  - Enables retry logic for batch processing jobs that exceed ~7 hour token lifetime
  - Exported in package `__all__` for easy import: `from pydantic_ai_claude_code import ClaudeOAuthError`
  - Fully documented in README.md with simple and batch processing examples

- **Tool results as file attachments**: Tool execution results now written to temporary files instead of embedded in prompts
  - Uses same mechanism as `additional_files` feature for consistency
  - Tool results written to `tool_result_{counter}_{tool_name}.txt` temp files
  - Prevents prompt bloat with large tool outputs
  - Files automatically merged into `additional_files` settings dict
  - Prompt includes reference: "Additional Information: The results from the {tool_name} tool are available in the file {filename}"

### Changed
- **Error detection priority system**: Implemented priority-based error handling to prevent false positives
  - Priority 1: OAuth errors (most specific - JSON parsing + keyword matching)
  - Priority 2: Rate limit errors (regex pattern matching)
  - Priority 3: Infrastructure failures (timeout, process errors)
  - Priority 4: Generic errors
  - Prevents multi-hour waits when OAuth has actually expired but rate limit pattern matches error message

- **Message formatting refactoring**: Reduced complexity in `messages.py` through helper function extraction
  - `format_messages_for_claude()` signature changed to return `tuple[str, dict[str, Path]]`
  - Extracted `_create_tool_result_file()` for temp file creation
  - Extracted `_process_request_parts()` for ModelRequest processing
  - Extracted `_process_response_parts()` for ModelResponse processing
  - Extracted `_count_request_parts()` and `_count_response_parts()` for conversation context
  - All functions now meet complexity thresholds (≤10 cyclomatic complexity)

### Fixed
- **Rate limit false positives**: OAuth errors now detected before rate limit pattern matching
  - Previously, error messages containing both OAuth keywords and rate limit text would trigger rate limit retry
  - Now correctly raises `ClaudeOAuthError` immediately for faster failure and clearer error messages

## [0.5.14] - 2025-10-18

### Fixed
- **Type safety improvements**: Resolved all mypy type errors for strict type checking
  - Added proper type casts in `structure_converter.py` for dictionary value access
  - Added missing stream event types: `ClaudeStreamMessageStartEvent` and `ClaudeStreamContentBlockDeltaEvent`
  - Fixed TypedDict compatibility issues in `utils.py` and `model.py`
  - All 11 source files now pass mypy strict type checking with zero errors

### Changed
- **Code quality improvements**: Refactored complex functions to meet project standards
  - Extracted `_format_field_tree_lines()` helper function to eliminate duplicate code
  - Reduced `_build_array_of_objects_example()` from 37 to ≤30 statements
  - Reduced `_build_object_example()` from 41 statements and 13 branches to within limits
  - All code passes ruff checks (≤30 statements, ≤12 branches, ≤5 arguments)
  - Improved code maintainability with shared formatting logic

### Added
- **Raw response saving**: Claude's raw JSON responses are now automatically saved to `response.json` in the working directory
  - Saved alongside `prompt.md` for complete request/response pairs
  - Always-on feature requiring no configuration
  - Works for both sync and async execution paths
  - Streaming requests only save `prompt.md` (events are incremental)
  - Provides full transparency for debugging and inspection

- **Numbered subdirectories for user-specified working directories**: Multiple CLI calls to the same working directory now create numbered subdirectories
  - First call: `working_dir/1/prompt.md`, `working_dir/1/response.json`
  - Second call: `working_dir/2/prompt.md`, `working_dir/2/response.json`
  - Prevents file overwrites during multi-phase operations (tool selection + argument collection)
  - Temp directories still work as before (no subdirectories for single-use temp dirs)

- **Additional files feature**: New `additional_files` setting allows copying files into working directory for Claude to read
  - Type: `dict[str, Path]` mapping destination filename to source file path
  - Supports subdirectories in destination: `{"docs/spec.md": Path("specs/feature.md")}`
  - Relative paths resolved from current working directory
  - Binary-safe file copying with `shutil.copy2()` (preserves permissions and timestamps)
  - Files copied before `prompt.md` is written so they can be referenced
  - Each call gets its own isolated copies in numbered subdirectories
  - Example:
    ```python
    agent.run_sync(
        "Analyze utils.py and compare with spec.md",
        model_settings={
            "additional_files": {
                "utils.py": Path("src/utils.py"),
                "docs/spec.md": Path("specs/feature.md"),
            }
        }
    )
    ```

## [0.5.13] - 2025-10-17

### Fixed
- **$ref resolution bug**: Fixed critical bug where JSON schema `$ref` references were not being resolved in `structure_converter.py`
  - Arrays of nested Pydantic models (e.g., `list[NestedModel]`) were incorrectly treated as arrays of strings
  - Added `_resolve_schema_ref()` calls in 4 locations to resolve references before checking `type` field
  - Fixes validation errors: `Input should be a valid dictionary or instance of NestedModel`
  - Added tests: `test_pydantic_generated_schema_with_ref_references` and `test_build_instructions_with_ref_references`

- **None/null value handling**: Implemented proper distinction between None, empty strings, and missing values
  - No file created = None/null value
  - Empty file = empty string ("")
  - Array gaps (e.g., 0000.txt, 0002.txt with 0001 missing) = None at that index position
  - Added helper functions: `_is_nullable()`, `_get_non_null_type()`, `_get_non_null_schema()`
  - Handles `anyOf`/`oneOf` schemas for nullable types (e.g., `str | None`)
  - Added 4 comprehensive tests for None handling in various scenarios

- **Streaming marker filtering**: Fixed streaming responses to properly filter out tool-use preambles
  - Changed marker instruction from conditional ("After completing any tool use...") to unconditional
  - Simple requests without tool use now properly output `<<<STREAM_START>>>` marker
  - Prevents "prompt.md" and tool-use commentary from appearing in streamed output
  - Adjusted `MIN_CONTENT_LENGTH` constant from 10 to 5 to allow concise responses
  - All 5 streaming behavior tests now pass reliably

## [0.5.12] - 2025-10-17

### Changed
- **Prompt restructuring**: Restructured all prompt construction functions with comprehensive markdown formatting for improved cognitive processing
  - `_build_function_tools_prompt`: Added headers, numbered steps, code blocks, and emphasis for function selection task
  - `build_structure_instructions`: Added markdown sections, examples, and clear hierarchy for file/folder organization
  - `_build_unstructured_output_instruction`: Added structured sections with requirements and emphasis
  - Improved scannability and comprehension with consistent markdown formatting

### Removed
- **Unused legacy code**: Deleted `format_tools_for_prompt` function from `tools.py`
  - Function implemented incorrect EXECUTE protocol that doesn't match actual control flow
  - Actual flow uses CHOICE format in `_build_function_tools_prompt`
  - Removed 80+ lines of unused code
  - Removed associated test functions

### Fixed
- **Message formatting consistency**: Updated role labels across codebase
  - Changed "Context:" to "Additional Information (from {function_name}):" for function return values
  - More descriptive labeling that clarifies purpose vs technical implementation
  - Updated tests to match new role labels

## [0.5.11] - 2025-10-16

## [0.5.10] - 2025-10-16

## [0.5.10] - 2025-10-15

### Changed
- **Removed completion marker requirement**: Simplified directory structure assembly by removing `.complete` marker files
  - CLI execution is synchronous, making completion markers redundant
  - Reduced complexity in `structure_converter.py` and `model.py`
  - Updated tests to verify markers are no longer created

- **Automatic validation error retry**: Argument collection now retries once on validation errors
  - Feeds validation errors back to Claude for self-correction
  - Creates new temp directory for retry attempt to avoid conflicts
  - Only returns error to user after all retries exhausted
  - Improves success rate for complex structured outputs

- **Code quality improvements**: Refactored to meet strict project standards
  - Reduced `extract_text_from_stream_event()` from 8 return statements to ≤6 by extracting 3 helper functions
  - Reduced `run_claude_sync()` from 39 statements to ≤30 by extracting execution helper
  - Reduced `run_claude_async()` from 41 statements to ≤30 by extracting execution helper
  - Reduced helper function arguments from 6 to 5 parameters
  - All code passes ruff checks (≤30 statements, ≤12 branches, ≤6 returns, ≤5 arguments)

- **Test suite optimization**: Removed redundant slow integration tests
  - Deleted `test_unstructured_output.py` (6 tests - redundant with test_basic.py)
  - Deleted `test_long_responses_integration.py` (5 slow integration tests - redundant with unit tests)
  - Removed 2 redundant model tests from test_basic.py
  - Test count reduced from 107 to 94 (~13% faster suite)
  - All unit tests still provide comprehensive coverage

## [0.5.9] - 2025-10-15

### Fixed
- **True incremental streaming**: Fixed streaming to deliver chunks in real-time as Claude generates them, not all at once after completion
  - Implemented background task pattern using `asyncio.create_task()` to consume CLI stream concurrently
  - Both Pydantic AI's internal code and user's `stream_text()` now read from a shared, growing buffer
  - Events are buffered as they arrive from CLI with polling-based iteration
  - Added `<<<STREAM_START>>>` marker to distinguish final response from tool-use messages
  - Text now appears incrementally with realistic delays (100-400ms between chunks)
  - Previously all text appeared at once after ~10 second delay despite CLI sending events incrementally

### Added
- **Comprehensive streaming tests**: New `test_streaming_behavior.py` with 5 tests verifying:
  - Incremental delivery with realistic timing gaps between chunks
  - Tool-use message filtering (only final response is streamed)
  - Complete response delivery with cumulative chunks
  - Usage tracking availability after completion
  - Background task concurrent consumption

### Changed
- **Code quality improvements**: Refactored to meet strict project standards
  - Reduced `_consume_stream_background()` from 52 statements to 30 by extracting `_process_marker_and_text()` helper
  - Reduced `_build_system_prompt_parts()` from 6 arguments to 4 by extracting tools from `model_request_parameters`
  - Added test constants for magic numbers (MIN_CHUNKS_FOR_STREAMING, MAX_TIME_TO_FIRST_CHUNK_MS, etc.)
  - All code passes ruff checks (≤30 statements, ≤12 branches, ≤5 arguments)

## [0.5.8] - 2025-10-14

## [0.5.7] - 2025-10-14

### Removed
- **Unsupported CLI flags**: Removed `--max-turns` and `--max-output-tokens` flags that were removed from Claude CLI
  - Removed from `ClaudeCodeSettings` TypedDict
  - Removed from `ClaudeCodeProvider.__init__()` and `get_settings()`
  - Removed from command building in `utils.py`
  - Removed from all examples, tests, and documentation
  - These settings were no longer supported by the Claude CLI

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
