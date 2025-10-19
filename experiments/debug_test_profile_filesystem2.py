#!/usr/bin/env python3
"""Debug script to inspect filesystem during argument collection - full function schema."""

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pydantic_ai_claude_code.structure_converter import (
    build_structure_instructions,
    read_structure_from_filesystem,
)


class Address(BaseModel):
    age: int
    city: str

# Simulate Phase 2 argument collection with FULL function schema
def test_filesystem_creation() -> tuple[Path, dict[str, Any]]:
    # This is the FULL schema for create_user function (all parameters)
    schema = {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "email": {"type": "string"},
            "profile": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "city": {"type": "string"}
                },
                "required": ["age", "city"]
            }
        },
        "required": ["username", "email", "profile"]
    }

    # Create temp directory that won't be auto-cleaned
    temp_dir = Path(f"/tmp/debug_profile_full_{uuid.uuid4().hex[:8]}")
    temp_dir.mkdir(exist_ok=True)

    print(f"Created temp directory: {temp_dir}")

    # Build instructions (this is what Claude receives)
    instructions = build_structure_instructions(
        schema=schema,
        temp_dir=str(temp_dir),
    )

    print("\n" + "="*80)
    print("INSTRUCTIONS FOR CLAUDE (Phase 2 - Argument Collection):")
    print("="*80)
    print(instructions)
    print("="*80)

    # Simulate what Claude SHOULD create based on the request:
    # "Create a user with username john_doe, email john@example.com, profile age=30, city=London"
    print("\nNow manually creating what Claude SHOULD create...")

    # Create username file
    username_file = temp_dir / "username.txt"
    username_file.write_text("john_doe")

    # Create email file
    email_file = temp_dir / "email.txt"
    email_file.write_text("john@example.com")

    # Create profile directory with nested files
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
            content = item.read_text()[:100]
            print(f"  FILE: {rel_path} -> {content!r}")
        else:
            print(f"  DIR:  {rel_path}/")

    print(f"\nTemp directory NOT cleaned up: {temp_dir}")
    print(f"Inspect with: ls -laR {temp_dir}")

    return temp_dir, schema

if __name__ == "__main__":
    temp_dir, schema = test_filesystem_creation()

    # Now test reading it back
    print("\n" + "="*80)
    print("TESTING READ BACK:")
    print("="*80)

    result = read_structure_from_filesystem(
        schema=schema,
        base_path=temp_dir,
    )

    print(f"Read result: {json.dumps(result, indent=2)}")
    print("\nThis matches the function signature:")
    print("  create_user(")
    print(f"    username={result.get('username')!r},")
    print(f"    email={result.get('email')!r},")
    print(f"    profile={result.get('profile')!r}")
    print("  )")
