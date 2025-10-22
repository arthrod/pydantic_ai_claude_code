# Refactoring Plan - pydantic-ai-claude-code

**Date Started:** 2025-10-22
**Objective:** Remove duplications, redundancies, and orphaned code without losing functionality

---

## Analysis Summary

- **1 completely orphaned file** (279 lines that can be deleted)
- **~300 lines of duplicated execution logic** that could be consolidated
- **Type conversion duplication** across files
- **Small utility modules** that could be consolidated
- **One massive file** (model.py - 2,039 lines) that could benefit from restructuring

---

## Phase 1: Quick Wins (Low Risk)

### 1.1 Delete `tools.py` (279 lines)
**Status:** Ready to delete
**Risk:** Very Low
**Evidence:**
- File is never imported anywhere in codebase
- Contains legacy tool calling code (EXECUTE format, legacy JSON format)
- Completely replaced by newer two-phase function calling in model.py

**Actions:**
1. Delete `src/pydantic_ai_claude_code/tools.py`
2. Run tests to verify no breakage

### 1.2 Verify `logging_utils.py` deletion
**Status:** Already deleted (in git status)
**Actions:**
1. Confirm deletion is complete
2. Verify no remaining imports

### 1.3 Consolidate type conversion (~30 lines savings)
**Current state:**
- `model.py:1852-1877` - `_convert_primitive_value()` static method
- `structure_converter.py:314-333, 428-447` - inline type conversion

**Actions:**
1. Extract unified `convert_primitive_value()` to `utils.py`
2. Update all call sites to use centralized version
3. Run tests

---

## Phase 2: Refactoring (Medium Risk)

### 2.1 Consolidate sync/async execution (~150 lines savings)
**Current state:**
- `_try_sync_execution_with_rate_limit_retry()` (lines 722-777)
- `_try_async_execution_with_rate_limit_retry()` (lines 909-965)
- `run_claude_sync()` (lines 780-858)
- `run_claude_async()` (lines 968-1046)

**Problem:** 80% of logic is duplicated - only differs by async/await keywords

**Approach:**
1. Extract shared error handling logic to `_handle_execution_errors()`
2. Extract shared retry logic to `_execute_with_retry()`
3. Keep thin sync/async wrappers that call shared logic
4. Run tests extensively

### 2.2 Split `model.py` (2,039 lines → multiple focused files)
**Problem:** Too many responsibilities in one file

**New structure:**
```
src/pydantic_ai_claude_code/model/
  __init__.py           # Main ClaudeCodeModel class (facade)
  prompts.py            # System prompt building (~400 lines)
  responses.py          # Response handling & conversion (~450 lines)
  json_extraction.py    # JSON extraction strategies (~250 lines)
  function_calling.py   # Function calling logic (~600 lines)
```

**Approach:**
1. Create `model/` package directory
2. Extract prompt building methods to `prompts.py`
3. Extract response handling to `responses.py`
4. Extract JSON extraction to `json_extraction.py`
5. Extract function calling to `function_calling.py`
6. Update imports in `__init__.py`
7. Update all imports throughout codebase
8. Run full test suite

---

## Phase 3: Polish (Optional)

### 3.1 Inline `response_utils.py` (15 lines savings)
**Status:** Optional - low impact

### 3.2 Make retry constants configurable
**Current:** Hard-coded in utils.py
**Improvement:** Add to ClaudeCodeSettings

---

## Test Strategy

After each phase:
1. Run full test suite: `uv run pytest`
2. Run specific tests for affected modules
3. Manual smoke test with examples

---

## Rollback Plan

If any phase causes issues:
1. Git revert specific commits
2. Check protocol.md for exact state before changes
3. Resume from last successful checkpoint

---

## Success Metrics

- All tests pass
- No functionality lost
- Code is more maintainable
- Line count reduced by ~450 lines
- Better separation of concerns