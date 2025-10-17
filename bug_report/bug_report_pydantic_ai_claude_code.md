# Bug Report: `$ref` References Not Resolved in JSON Schemas

## Summary

The `pydantic-ai-claude-code` package (v0.5.12) fails to handle `$ref` references in JSON schemas, causing it to misinterpret nested object arrays as primitive string arrays. This results in incorrect filesystem instructions being generated, leading to validation failures when the model tries to parse the response.

## Impact

**Critical**: Any Pydantic model containing `list[NestedModel]` fields will fail with validation errors. This is a blocking issue for all non-trivial structured outputs.

## Environment

- Package version: `pydantic-ai-claude-code==0.5.12`
- Python version: 3.12
- pydantic version: 2.12
- pydantic-ai version: (latest)

## Steps to Reproduce

```python
from pydantic import BaseModel, Field
from pydantic_ai_claude_code.structure_converter import build_structure_instructions

class NestedModel(BaseModel):
    priority: int = Field(description='Priority level')
    action: str = Field(description='Action to take')

class ParentModel(BaseModel):
    summary: str = Field(description='Summary text')
    items: list[NestedModel] = Field(description='List of nested items')

# Generate schema
schema = ParentModel.model_json_schema()
print("Schema:", schema)

# Generate instructions
instructions = build_structure_instructions(schema, "/tmp/test")
print("\nInstructions:", instructions)
```

## Expected Behavior

The schema contains a `$ref` reference:

```json
{
  "properties": {
    "items": {
      "items": {
        "$ref": "#/$defs/NestedModel"
      },
      "type": "array"
    }
  },
  "$defs": {
    "NestedModel": {
      "properties": {
        "priority": {"type": "integer"},
        "action": {"type": "string"}
      },
      "type": "object"
    }
  }
}
```

**Expected instructions should specify subdirectories for objects**:
```
- items: Collection of items. Create directory 'items/',
  then numbered subdirectories (0000/, 0001/, ...) each containing: priority, action
```

**Expected filesystem structure**:
```
items/
├── 0000/
│   ├── priority.txt
│   └── action.txt
└── 0001/
    ├── priority.txt
    └── action.txt
```

## Actual Behavior

**Actual instructions incorrectly specify .txt files for primitives**:
```
- items: Collection of Text value. Create directory 'items/',
  then numbered files (0000.txt, 0001.txt, ...)
```

**Actual filesystem structure created by Claude**:
```
items/
├── 0000.txt
└── 0001.txt
```

**Result**: Validation error when reading:
```
pydantic.ValidationError: 1 validation error for ParentModel
items.0
  Input should be a valid dictionary or instance of NestedModel [type=model_type, input_value='some string content', input_type=str]
```

## Root Cause

The issue is in multiple locations in `structure_converter.py` where `$ref` references are not resolved before checking the `type` field:

### 1. **Line 76-77** (`_write_array_field`):
```python
items_schema = field_schema.get("items", {})
item_type = items_schema.get("type", "string")  # ❌ Returns "string" default when $ref present
```

### 2. **Line 286-287** (`_read_array_field`):
```python
items_schema = field_schema.get("items", {})
item_type = items_schema.get("type", "string")  # ❌ Same issue
```

### 3. **Line 379** (`_build_field_descriptions`):
```python
items_schema = field_schema.get("items", {})
item_type = items_schema.get("type", "string")  # ❌ Same issue
```

### 4. **Line 506** (`_build_example_structure`):
```python
items_schema = field_schema.get("items", {})
item_type = items_schema.get("type", "string")  # ❌ Same issue
```

When `items_schema` contains `{"$ref": "#/$defs/NestedModel"}`, the call to `.get("type", "string")` returns the default value `"string"` instead of resolving the reference to find the actual `"object"` type.

## Suggested Fix

Add a helper function to resolve `$ref` references:

```python
def _resolve_ref(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref references in schema.

    Args:
        schema: Schema that may contain $ref
        root_schema: Root schema containing $defs

    Returns:
        Resolved schema
    """
    if "$ref" not in schema:
        return schema

    ref_path = schema["$ref"]

    # Handle internal references like "#/$defs/ModelName"
    if ref_path.startswith("#/"):
        parts = ref_path[2:].split("/")
        resolved = root_schema
        for part in parts:
            resolved = resolved[part]
        return resolved

    # External references not supported yet
    return schema
```

Then update all affected locations to resolve references first. For example, in `_read_array_field`:

```python
def _read_array_field(
    field_name: str,
    field_schema: dict[str, Any],
    base_path: Path,
    root_schema: dict[str, Any],  # Add root_schema parameter
) -> list[Any]:
    """Read array field from directory with numbered files/subdirs."""
    array_dir = base_path / field_name

    # ... existing validation code ...

    items_schema = field_schema.get("items", {})
    items_schema = _resolve_ref(items_schema, root_schema)  # ✅ Resolve reference
    item_type = items_schema.get("type", "string")

    if item_type == "object":
        return _read_array_of_objects(array_dir, items_schema, root_schema)
    else:
        return _read_array_of_primitives(array_dir, item_type)
```

**Note**: This requires threading `root_schema` through all function calls, or alternatively, the main functions could resolve all `$ref` references in the schema upfront before processing.

## Real-World Failure Example

This bug caused a complete failure in a production grant application improvement system:

```python
class ImprovementRecommendation(BaseModel):
    priority: int
    criteria_addressed: list[str]
    current_weakness: str
    specific_action: str
    suggested_text: str | None = None

class ImprovementPlan(BaseModel):
    executive_summary: str
    recommendations: list[ImprovementRecommendation]  # ❌ Fails here
    preserve_unchanged: list[str]
```

The system ran for 1.5 hours processing 20 research improvements, only to fail at the final step when trying to parse the holistic improvement plan because `recommendations` was interpreted as `list[str]` instead of `list[ImprovementRecommendation]`.

## Workaround

Currently, there is **no workaround** without modifying the package source code or flattening all nested models into primitives (which defeats the purpose of structured outputs).

## Additional Context

- The v0.5.12 update successfully fixed the optional field handling issue (checking `required` before attempting to read missing paths)
- The prompt restructuring with markdown headers and code blocks is working well
- However, this `$ref` resolution issue is a fundamental blocker for complex schemas

## Related Code

The package appears to have been designed with the assumption that all schemas would have explicit `type` fields at every level, but Pydantic's JSON schema generation uses `$ref` references for nested models to avoid duplication.

This is standard JSON Schema behavior per the spec: https://json-schema.org/understanding-json-schema/structuring.html#ref

Thank you for the excellent work on this package! This fix would make it production-ready for complex use cases.
