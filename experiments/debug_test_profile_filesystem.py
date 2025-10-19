#!/usr/bin/env python3
"""Debug script to inspect filesystem during argument collection."""

import json
import uuid
from pathlib import Path

from pydantic import BaseModel

from pydantic_ai_claude_code.structure_converter import build_structure_instructions


class Address(BaseModel):
    age: int
    city: str

# Simulate Phase 2 argument collection
def test_filesystem_creation() -> Path:
    # Create schema for profile parameter
    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["age", "city"]
    }

    # Create temp directory that won't be auto-cleaned
    temp_dir = Path(f"/tmp/debug_profile_{uuid.uuid4().hex[:8]}")
    temp_dir.mkdir(exist_ok=True)

    print(f"Created temp directory: {temp_dir}")

    # Build instructions
    instructions = build_structure_instructions(
        schema=schema,
        temp_dir=str(temp_dir),
    )

    print("\n" + "="*80)
    print("INSTRUCTIONS FOR CLAUDE:")
    print("="*80)
    print(instructions)
    print("="*80)

    # Simulate what Claude should create based on the request:
    # "profile age=30, city=London"
    print("\nNow manually creating what Claude SHOULD create...")

    profile_dir = temp_dir / "profile"
    profile_dir.mkdir(exist_ok=True)

    age_file = profile_dir / "age.txt"
    age_file.write_text("30")

    city_file = profile_dir / "city.txt"
    city_file.write_text("London")

    print("\nCreated structure:")
    for item in sorted(temp_dir.rglob("*")):
        rel_path = item.relative_to(temp_dir)
        if item.is_file():
            content = item.read_text()
            print(f"  FILE: {rel_path} -> {content!r}")
        else:
            print(f"  DIR:  {rel_path}/")

    print(f"\nTemp directory NOT cleaned up: {temp_dir}")
    print(f"Inspect with: ls -laR {temp_dir}")

    return temp_dir

if __name__ == "__main__":
    temp_dir = test_filesystem_creation()

    # Now test reading it back
    print("\n" + "="*80)
    print("TESTING READ BACK:")
    print("="*80)

    from pydantic_ai_claude_code.structure_converter import (
        read_structure_from_filesystem,
    )

    result = read_structure_from_filesystem(
        schema={
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["age", "city"]
        },
        base_path=temp_dir,
    )

    print(f"Read result: {json.dumps(result, indent=2)}")
