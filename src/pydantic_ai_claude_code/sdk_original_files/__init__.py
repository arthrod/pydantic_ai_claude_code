"""
SDK Original Files Import Tracking

Imported from claude-agent-sdk conceptual design
Last updated: 2025-01-20
Next review: 2025-04-20

This directory contains SDK-compatible type definitions and components
that enable direct integration with the Claude Agent SDK while maintaining
pydantic_ai compatibility.

Import decisions:
- types.py: Core type definitions for SDK compatibility
- _errors.py: Error classes for consistent error handling

Usage:
These files provide the foundation for SDK integration without requiring
the actual SDK as a dependency. They define interfaces that can be
implemented by our adapter layer.

To update:
1. Check claude-agent-sdk releases for new features
2. Update type definitions as needed
3. Mark modifications with # PYDANTIC_AI_MOD
4. Run integration tests
"""

SDK_VERSION = "conceptual-0.1"
LAST_IMPORT_DATE = "2025-01-20"
