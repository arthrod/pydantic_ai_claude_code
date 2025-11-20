# SDK Update Tracking

## Current Version
- **SDK Version**: 0.1.8
- **Last Import**: 2024-01-20
- **Next Review**: 2024-04-20

## Update Process

### 1. Check for Updates
```bash
pip index versions claude-agent-sdk
```

### 2. Compare Changes
When updating, look for changes in:
- `types.py` - Message and option types
- `_errors.py` - Exception classes
- `_internal/message_parser.py` - Message parsing
- `_internal/transport/subprocess_cli.py` - CLI handling

### 3. Modifications to Track
Our modifications are marked with `# PYDANTIC_AI_MOD` comments:

- OAuth error detection (added to error handling)
- Rate limit retry logic (added to CLI transport)
- Sandbox runtime support (added to command building)

### 4. Testing After Update
Run full test suite:
```bash
uv run pytest tests/ -v
```

## Import Decisions

| File | Import % | Reason |
|------|----------|--------|
| types.py | 100% | Core type definitions needed |
| _errors.py | 100% | All error classes used |
| message_parser.py | 100% | Complete parsing logic |
| query.py | 90% | Minor patches for our use |
| subprocess_cli.py | 70% | Added OAuth detection |

## Changelog

### 2024-01-20 (v0.1.8)
- Initial import structure created
- Type stubs added for compatibility
- SDK dependency added to pyproject.toml
