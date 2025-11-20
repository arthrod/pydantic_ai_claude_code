# New Test Files for Refactored Modules

This directory contains comprehensive unit tests for the newly refactored modular structure introduced in the codebase restructuring.

## Test Files

### test_utils_modules.py
Tests for the `_utils` package including:
- **file_utils**: Directory management and file copying utilities
- **json_utils**: JSON extraction and markdown code fence stripping
- **type_utils**: Type conversion and validation

### test_core_modules.py
Tests for the `core` package including:
- **oauth_handler**: OAuth error detection from CLI output
- **retry_logic**: Rate limit detection, wait time calculation, and infrastructure failure detection
- **debug_saver**: Debug file saving and management
- **sandbox_runtime**: Sandbox runtime path resolution and command wrapping

### test_structured_modules.py
Tests for the `structured` package including:
- **file_handler**: Structured and unstructured output file handling
- **function_selector**: Two-phase function selection protocol
- **converter**: Re-exports from structure_converter module

### test_transport_module.py
Tests for the `transport` package including:
- **EnhancedCLITransport**: Enhanced CLI transport with retry logic
- **Settings conversion**: Converting settings to SDK options format

## Test Coverage

Total: **107 test methods** covering:
- Happy path scenarios
- Edge cases
- Error conditions
- Mocking of external dependencies
- Async operations
- File system operations
- Command execution
- JSON parsing
- OAuth detection
- Rate limiting
- Sandbox wrapping

## Running Tests

```bash
# Run all new tests
pytest tests/test_*modules.py tests/test_transport_module.py -v

# Run specific test file
pytest tests/test_utils_modules.py -v

# Run with coverage
pytest tests/test_*modules.py tests/test_transport_module.py --cov=src/pydantic_ai_claude_code --cov-report=html
```

## Test Structure

All tests follow pytest conventions:
- Test classes prefixed with `Test`
- Test methods prefixed with `test_`
- Clear, descriptive test names
- Proper use of fixtures (tmp_path, etc.)
- Mocking of external dependencies
- Assertions with meaningful messages