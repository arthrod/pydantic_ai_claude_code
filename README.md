# Pydantic AI Claude Code

Use your local Claude Code CLI as a Pydantic AI model provider.

This package provides a Pydantic AI-compatible model implementation that wraps the local Claude CLI, enabling you to use Claude locally with all Pydantic AI features including structured responses, tool calling, streaming, and multi-turn conversations.

## Features

- **Full Pydantic AI Compatibility**: Drop-in replacement for any Pydantic AI model
- **Structured Responses**: Get validated, typed responses using Pydantic models
- **Custom Tool Calling**: Use your own Python functions as tools
- **True Streaming**: Real-time response streaming via Claude CLI's stream-json mode
- **Local Execution**: All processing happens locally on your machine
- **Session Persistence**: Maintain conversation context across multiple requests
- **Configurable**: Fine-tune permissions, working directories, and tool access

## Installation

```bash
# Using uv (recommended)
uv add pydantic-ai-claude-code

# Using pip
pip install pydantic-ai-claude-code
```

**Prerequisites**: You must have [Claude Code CLI](https://claude.com/claude-code) installed and authenticated on your system.

## Quick Start

### Basic Usage (String Format - Simplest!)

```python
import pydantic_ai_claude_code  # Register the provider

from pydantic_ai import Agent

# Just use the string format - easiest way!
agent = Agent('claude-code:sonnet')

# Run a simple query
result = agent.run_sync("What is 2+2?")
print(result.output)  # Output: 4
```

### Structured Responses

```python
import pydantic_ai_claude_code

from pydantic import BaseModel
from pydantic_ai import Agent

class Analysis(BaseModel):
    complexity: int  # 1-10
    maintainability: str
    suggestions: list[str]

# String format works with all Pydantic AI features
agent = Agent('claude-code:sonnet', output_type=Analysis)

result = agent.run_sync("Analyze this code: def foo(): pass")
print(f"Complexity: {result.output.complexity}")
```

### Custom Tools

```python
import pydantic_ai_claude_code

from pydantic_ai import Agent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

agent = Agent(
    'claude-code:sonnet',
    tools=[get_weather],
)

result = agent.run_sync("What's the weather in Paris?")
print(result.output)
```

### Streaming

```python
import pydantic_ai_claude_code

from pydantic_ai import Agent

agent = Agent('claude-code:sonnet')

async with agent.run_stream('Write a haiku about code') as result:
    async for text in result.stream_text():
        print(text, end='', flush=True)
```

## Advanced Configuration

When you need fine-grained control, use the explicit model and provider:

```python
from pydantic_ai import Agent
from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider

# Custom provider with specific settings
provider = ClaudeCodeProvider(
    working_directory="/path/to/project",
    allowed_tools=["Read", "Edit", "Grep"],  # Restrict tool access
    permission_mode="acceptEdits",
    max_turns=10,
    use_temp_workspace=False,  # Use specific directory instead of /tmp
)

model = ClaudeCodeModel("sonnet", provider=provider)
agent = Agent(model)

result = agent.run_sync("Analyze the codebase structure")
```

### Temporary Workspace

```python
from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider

# Create isolated workspace that auto-cleans
with ClaudeCodeProvider(use_temp_workspace=True) as provider:
    model = ClaudeCodeModel("sonnet", provider=provider)
    agent = Agent(model)
    result = agent.run_sync("Create a test file")
# Workspace automatically cleaned up
```

### Logging

The package uses Python's standard logging module. To enable debug logging in your application:

```python
import logging

# Enable debug logging for pydantic-ai-claude-code
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('pydantic_ai_claude_code').setLevel(logging.DEBUG)

# Or just for specific components
logging.getLogger('pydantic_ai_claude_code.model').setLevel(logging.DEBUG)
logging.getLogger('pydantic_ai_claude_code.utils').setLevel(logging.INFO)
```

This will log:
- Model initialization and configuration
- CLI command execution and responses
- Message formatting and conversion
- Tool call parsing and execution
- Structured output handling
- Streaming events and completion

## Available Models

- `claude-code:sonnet` - Claude 3.5 Sonnet (default, recommended)
- `claude-code:opus` - Claude 3 Opus (most capable)
- `claude-code:haiku` - Claude 3.5 Haiku (fastest)

Or use full model names like `claude-code:claude-sonnet-4-5-20250929`

## Integration with Existing Projects

Replace your current LLM calls with Claude Code:

**Before:**
```python
agent = Agent('openai:gpt-4o')
# or
agent = Agent('anthropic:claude-3-5-sonnet-latest')
```

**After:**
```python
import pydantic_ai_claude_code  # Add this import

agent = Agent('claude-code:sonnet')  # Change this line
```

Everything else stays the same! All your tools, structured outputs, dependencies, and streaming code works identically.

## Key Differences from Cloud Providers

| Aspect | Cloud Providers | Claude Code Local |
|--------|----------------|-------------------|
| **Execution** | Remote API calls | Local on your machine |
| **Cost** | Per-token pricing | Uses Claude desktop subscription |
| **Data Privacy** | Data sent to cloud | Data stays local |
| **Speed** | Network latency | Local execution |
| **API Key** | Required | Not needed (uses local auth) |

## Examples

See the `examples/` directory for more demonstrations:

- `basic_example.py` - Simple queries and usage tracking
- `structured_example.py` - Structured output with Pydantic models
- `async_example.py` - Async/await usage patterns
- `advanced_example.py` - Custom provider configurations
- `tools_and_streaming.py` - Custom tools and streaming responses

## License

MIT License
