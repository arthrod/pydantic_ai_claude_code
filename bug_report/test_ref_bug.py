#!/usr/bin/env python3
"""
Minimal reproducible test case for $ref resolution bug in pydantic-ai-claude-code.

Run this to demonstrate the issue:
    python test_ref_bug.py

Expected: Instructions should specify subdirectories for nested objects
Actual: Instructions incorrectly specify .txt files (treating objects as strings)
"""

from pydantic import BaseModel, Field
from pydantic_ai_claude_code.structure_converter import build_structure_instructions
import json


class NestedModel(BaseModel):
    """A nested model with multiple fields."""

    priority: int = Field(description="Priority level (1-10)")
    action: str = Field(description="Action to take")
    details: str | None = Field(default=None, description="Optional details")


class ParentModel(BaseModel):
    """Parent model containing a list of nested models."""

    summary: str = Field(description="Summary text")
    items: list[NestedModel] = Field(description="List of nested items")
    tags: list[str] = Field(description="Simple string list for comparison")


def main():
    print("=" * 80)
    print("Testing $ref Resolution Bug")
    print("=" * 80)

    # Generate schema
    schema = ParentModel.model_json_schema()

    print("\n1. Generated JSON Schema:")
    print("-" * 80)
    print(json.dumps(schema, indent=2))

    print("\n2. Key Schema Details:")
    print("-" * 80)
    items_field = schema["properties"]["items"]
    print(f"items field: {json.dumps(items_field, indent=2)}")
    print(f"\nNotice: items.items contains $ref: '{items_field['items'].get('$ref', 'N/A')}'")
    print(f"NestedModel definition: {json.dumps(schema['$defs']['NestedModel'], indent=2)}")

    # Generate instructions
    print("\n3. Generated Instructions:")
    print("-" * 80)
    instructions = build_structure_instructions(schema, "/tmp/test")
    print(instructions)

    print("\n4. Analysis:")
    print("-" * 80)

    # Check if instructions correctly handle nested objects
    if "numbered subdirectories" in instructions and "priority, action, details" in instructions:
        print("✅ PASS: Instructions correctly specify subdirectories for nested objects")
    elif "numbered files (0000.txt, 0001.txt" in instructions and "items: Collection of Text value" in instructions:
        print("❌ FAIL: Instructions incorrectly treat nested objects as strings")
        print("\nBUG CONFIRMED:")
        print("- Expected: 'Collection of items' with 'numbered subdirectories'")
        print("- Actual: 'Collection of Text value' with 'numbered files (.txt)'")
        print("\nThis happens because $ref references are not resolved before checking 'type'")
    else:
        print("⚠️  UNKNOWN: Unexpected instruction format")

    print("\n5. Expected vs Actual Filesystem Structure:")
    print("-" * 80)
    print("EXPECTED (correct):")
    print("""
items/
├── 0000/
│   ├── priority.txt
│   ├── action.txt
│   └── details.txt
└── 0001/
    ├── priority.txt
    ├── action.txt
    └── details.txt
    """)

    print("\nACTUAL (buggy):")
    print("""
items/
├── 0000.txt  ← Wrong! Treats object as string
└── 0001.txt  ← Wrong! Treats object as string
    """)

    print("\n6. Comparison with Simple String List (tags field):")
    print("-" * 80)
    print("The 'tags' field (list[str]) correctly generates:")
    print("- 'Collection of Text value'")
    print("- 'numbered files (0000.txt, 0001.txt, ...)'")
    print("\nThis is correct for primitives but incorrect when applied to nested objects!")

    print("\n" + "=" * 80)
    print("End of Test")
    print("=" * 80)


if __name__ == "__main__":
    main()
