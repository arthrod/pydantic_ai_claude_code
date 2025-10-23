# Bug Report: Empty Content Fields in Structured Output (v0.8.0)

## Summary

In `pydantic-ai-claude-code` version 0.8.0, when asking Claude to read source content from a file and populate structured output fields, Claude creates empty content fields instead of extracting and using the content from the source material.

**Severity**: 🔴 Critical
**Affects**: pydantic-ai-claude-code v0.8.0
**Working in**: pydantic-ai-claude-code v0.7.4
**Regression**: Yes (bug introduced in 0.8.0)

---

## Bug Description

When using structured outputs (Pydantic models) with file attachments, Claude misinterprets the instructions and creates empty `content` fields even when:

1. The source content file exists and has relevant information
2. The prompt explicitly instructs Claude to read and use the source content
3. The source content matches the requested field names

Claude's response states: *"Content files are all empty as specified in the user request"* — but the request never asks for empty fields.

---

## Impact

### Critical Business Impact
- **Grant application system**: Generated completely empty applications (0% usable)
- **All REQUIRED fields empty**: Including fields with matching content in source material
- **Silent failure**: No errors raised, but output is unusable
- **Data loss**: Hours of work creating source content rendered useless

### User Experience
- Applications scored 0/100 points by evaluation system
- Wasted API costs (~$0.22 per broken application generation)
- Time loss debugging mysterious empty outputs

---

## Reproduction

### Minimal Reproduction Case

See `bug/reproduce_bug_german.py` for the exact reproduction script.

**Steps:**

1. Install pydantic-ai-claude-code v0.8.0
2. Create a Pydantic model with fields (e.g., `name`, `content`)
3. Create a source content file with relevant information
4. Use `prepare_file_attachments()` to attach the source file
5. Prompt Claude to read the file and populate the fields
6. Result: All `content` fields are empty strings

**Expected behavior:**
Fields populated with content extracted from source file.

**Actual behavior (v0.8.0):**
All fields created with empty strings.

### Test Results

| Version | Required Fields Populated | Optional Fields Populated | Status |
|---------|---------------------------|---------------------------|--------|
| v0.8.0  | 0/8 (0%)                  | 0/4 (0%)                  | ❌ **BROKEN** |
| v0.7.4  | 8/8 (100%)                | 2/4 (50%)*                | ✅ **WORKS** |

*Optional fields empty only when source content doesn't cover those topics (expected)

---

## Root Cause Analysis

### What Changed in v0.8.0

The structured output format template includes instructions about handling empty strings:

```markdown
**Empty strings:**
```
description.txt contains "" (empty file for empty string)
```
```

In v0.8.0, Claude appears to misinterpret these instructions as applying to the OUTPUT rather than just being formatting guidelines for how to represent empty strings in the file structure.

### Claude's Response (v0.8.0)

From the Claude CLI response:

```
"content.txt - Empty file ready for content (per instructions, all content files are empty)"
```

Claude explicitly states it created empty files "per instructions" — but the instructions clearly say:
- "Read and understand all source content"
- "For each form field, write appropriate content using relevant parts of the source material"
- "Ensure all REQUIRED fields are filled"

### Why v0.7.4 Works

In v0.7.4, Claude correctly:
1. Reads the source_content.md file
2. Extracts relevant information for each field
3. Populates content fields with actual text
4. Only leaves fields empty when source doesn't contain relevant information

---

## Evidence

### File Locations

- **Original bug evidence**: `/tmp/claude_prompt_auw6be_k/`
  - `user_request.md`: Contains the prompt (instructs to populate fields)
  - `source_content.md`: Contains full project content (all information present)
  - `response.json`: Claude's response saying "all content files are empty"
  - `claude_data_structure_d59a9a36/fields/*/content.txt`: All empty

- **Reproduction scripts**: `./bug/`
  - `reproduce_bug.py`: Simple 3-field test
  - `reproduce_bug_realistic.py`: 12-field test with form requirements
  - `reproduce_bug_german.py`: **Exact original scenario** (German field names)

- **Test data**: `./bug/temp/`
  - `source_content.md`: English test content
  - `german_source.md`: German test content matching original

---

## Workaround

**Immediate fix:** Downgrade to v0.7.4

```bash
pip uninstall pydantic-ai-claude-code
pip install pydantic-ai-claude-code==0.7.4
```

**Verification:**
```bash
python -c "import pydantic_ai_claude_code; print(pydantic_ai_claude_code.__version__)"
# Should print: 0.7.4
```

---

## Test Instructions

### Run Reproduction Test

```bash
# Install v0.8.0 to see the bug
pip install pydantic-ai-claude-code==0.8.0
python bug/reproduce_bug_german.py

# Expected output:
# ❌ CRITICAL BUG: 8 REQUIRED fields are EMPTY!
#    (Total empty: 12/12)

# Install v0.7.4 to verify fix
pip uninstall -y pydantic-ai-claude-code
pip install pydantic-ai-claude-code==0.7.4
python bug/reproduce_bug_german.py

# Expected output:
# ✅ SUCCESS: All 10 required fields have content
# ⚠️  MINOR ISSUE: 2 optional fields are empty
```

---

## Technical Details

### Environment

- Python: 3.12
- pydantic-ai: 1.3.0
- pydantic: 2.12.3
- Model: `claude-code:sonnet` (Claude Sonnet 4.5 via CLI)

### Affected Code Path

1. `src.llm.application.generate_structured_application()` calls pydantic-ai Agent
2. Agent creates structured output using file structure template
3. Template includes `user_request.md` with actual prompt
4. Claude CLI should read `user_request.md` and `source_content.md`
5. **Bug**: In v0.8.0, Claude creates empty `content.txt` files
6. **Expected**: Claude should populate `content.txt` with extracted information

### Prompt Structure

The prompt has two levels:

1. **Outer template** (from pydantic-ai-claude-code):
   - Explains file structure format
   - Has instructions about empty strings
   - Says "Read the file `user_request.md`"

2. **Inner prompt** (user_request.md):
   - System prompt: "You are an expert grant application writer..."
   - User prompt: "Read @source_content.md and populate fields"
   - Clear instructions to fill fields with content

**Bug hypothesis**: v0.8.0 template changes caused Claude to confuse the outer template's empty string handling instructions with instructions for the actual output.

---

## Recommendation

### For pydantic-ai-claude-code Maintainers

1. **Revert** the template changes from v0.8.0 that introduced this regression
2. **Add tests** for structured outputs with file references
3. **Test case**: Ensure populated fields when source content exists
4. **Document** the expected behavior for empty vs populated fields

### For Users

- **Do not upgrade** to v0.8.0 until this is fixed
- **Pin version** to v0.7.4 in requirements: `pydantic-ai-claude-code==0.7.4`
- **Test thoroughly** after any version upgrades

---

## Contact

**Reporter**: Auto-generated bug report from grant application system debugging
**Date**: 2025-10-23
**System**: GrantApplications/LLM-powered application generator

**Repository**: https://github.com/anthropics/pydantic-ai-claude-code (if applicable)

---

## Appendix: Example Output Comparison

### v0.8.0 (BROKEN)

```json
{
  "fields": [
    {
      "name": "Beschreibe dein Projekt kurz",
      "description": "REQUIRED field...",
      "content": "",  // ❌ EMPTY despite source having content
      "word_limit": 100
    },
    ...all fields empty...
  ]
}
```

### v0.7.4 (WORKING)

```json
{
  "fields": [
    {
      "name": "Beschreibe dein Projekt kurz",
      "description": "REQUIRED field...",
      "content": "LLMSecTest ist ein Open-Source-Tool...",  // ✅ Populated
      "word_limit": 100
    },
    ...all required fields populated...
  ]
}
```
