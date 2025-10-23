#!/usr/bin/env python3
"""
Standalone reproduction of pydantic-ai-claude-code v0.8.0 bug.

This script has NO external dependencies beyond:
- pydantic-ai-claude-code
- pydantic-ai
- pydantic

Run with v0.8.0 to see the bug, v0.7.4 to see it work.
"""

import pathlib
import tempfile
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider


class ApplicationField(BaseModel):
    """A single application field."""
    name: str = Field(description="Field name")
    description: str = Field(description="Field description")
    content: str = Field(description="Field content from source material")
    word_limit: int | None = Field(description="Word limit (None if unlimited)")


class StructuredApplication(BaseModel):
    """Application with multiple fields."""
    fields: list[ApplicationField] = Field(description="All application fields")


# Create test data inline
SOURCE_CONTENT = """# PROJECT: LLMSecTest

## Beschreibe dein Projekt kurz

LLMSecTest ist ein Open-Source-Tool, das Entwicklern hilft, Sicherheitslücken in LLM-Anwendungen zu identifizieren. Es basiert auf OWASP Top 10 für LLMs 2025 und bietet umfassende Tests für alle 10 Schwachstellenkategorien.

## Welche gesellschaftliche Herausforderung willst du mit dem Projekt angehen?

Die LLM-Sicherheitskrise bedroht Millionen. Entwickler fehlt es an zugänglicher Test-Infrastruktur. 10.000+ Organisationen bauen LLM-Apps, aber es gibt kein umfassendes Open-Source-Security-Framework.

## Wie willst du dein Projekt technisch umsetzen?

Pytest-style Design für Entwicklervertrautheit. Kernkomponenten: Test Engine mit 10 Python-Modulen, Multi-LLM-Adapter, CI/CD-Integration, Reporting-System.

## Hast du schon an der Idee gearbeitet?

NEUES PROJEKT - Alle Entwicklung während der Förderung. Kein bestehender Code. 100% geplant.

## Welche ähnlichen Ansätze gibt es schon?

Garak: 6/10 OWASP-Kategorien. LLMGuard/Vigil: 2/10 Kategorien. LLMSecTest: 10/10 OWASP-Kategorien + pytest-Integration.

## Wer ist die Zielgruppe?

Primäre Nutzer: Application Developers (60%), Security Engineers (25%), AI/ML Researchers (10%).

## Skizziere kurz die wichtigsten Meilensteine

M1 (Wochen 1-6): Pytest-Plugin, LLM-Adapter, 3 OWASP-Module. M2 (Wochen 7-12): 7 OWASP-Module, 100+ Tests.

## An welchen Software-Projekten hast du gearbeitet?

GitHub: wehnsdaefflae. SpotTheBot (Prototype Fund Round 14). DoppelCheck (Media Lab Bayern). AMPEL (Universitätsklinikum Leipzig).

## Wie viele Stunden willst du arbeiten?

Einzelanträger: 950 Stunden über 6 Monate.

## Erfahrung, Hintergrund, Motivation

Dr. Mark Wernsdorfer, 10+ Jahre Python-Entwicklung, Erfahrung mit LLMs und Sicherheit.
"""


def test_bug():
    """Test that reproduces the v0.8.0 bug."""

    # Write source content to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, prefix='claude_source_') as f:
        f.write(SOURCE_CONTENT)
        source_file = pathlib.Path(f.name)

    try:
        # System prompt
        system_prompt = """
You are an expert grant application writer. Your task is to organize source material
into a structured application that follows specific form requirements.

You will receive:
1. Source content (project description, qualifications, organizational info)
2. Form requirements (field names, descriptions, word limits)

Your task:
- Distribute the source content into the required form fields
- Each field should contain relevant, well-written content
- Respect word limits (stay under, not over)
- Fill all required fields
- Use appropriate content for each field based on its name and description
- Write professionally and concisely
"""

        # User prompt - using @filename syntax for file reference
        user_prompt = """
Create a structured grant application by organizing the source content into the required form fields.

**Source Content:** @source_content.md

**Form Fields:**
- **Beschreibe dein Projekt kurz** (REQUIRED, max 100 words) - Text only
- **Welche gesellschaftliche Herausforderung willst du mit dem Projekt angehen?** (REQUIRED, max 175 words) - Text only
- **Wie willst du dein Projekt technisch umsetzen?** (REQUIRED, max 175 words) - Text only
- **Hast du schon an der Idee gearbeitet?** (REQUIRED, max 100 words) - Text only
- **Link zum bestehenden Projekt** (optional) - Link if applicable
- **Welche ähnlichen Ansätze gibt es schon?** (REQUIRED, max 100 words) - Text only
- **Wer ist die Zielgruppe?** (REQUIRED, max 100 words) - Text only
- **Skizziere kurz die wichtigsten Meilensteine** (REQUIRED, max 100 words) - Text only
- **An welchen Software-Projekten hast du gearbeitet?** (optional, max 100 words) - Max 3 projects
- **Wie viele Stunden willst du arbeiten?** (REQUIRED) - Hours during funding period
- **Erfahrung, Hintergrund, Motivation** (optional, max 100 words) - Background info
- **Verlängerung** (optional, max 175 words) - Extension request if applicable

**Instructions:**
1. Read and understand all source content
2. For each form field, write appropriate content using relevant parts of the source material
3. Ensure all REQUIRED fields are filled
4. Respect word limits strictly (stay under, preferably 95% of limit)
5. Write professionally, concisely, and persuasively
6. Do not invent information - only use what's in the source content
7. For each field, provide a brief description of what it contains

Return a structured application with all fields populated.
"""

        # Create provider without settings
        provider = ClaudeCodeProvider()

        # Create ClaudeCodeModel instance
        model = ClaudeCodeModel(model_name="sonnet", provider=provider)

        # Create agent
        agent = Agent(
            model=model,
            output_type=StructuredApplication,
            system_prompt=system_prompt,
        )

        # Prepare model settings with file attachment (using dict, not ClaudeCodeSettings)
        # This is the key part: pass the source file so @source_content.md works
        # Using dict (as our project does) triggers the bug in v0.8.0
        model_settings = {
            "additional_files": {
                "source_content.md": source_file
            }
        }

        # Run
        print("=" * 80)
        print("Testing pydantic-ai-claude-code...")
        try:
            import pydantic_ai_claude_code
            print(f"Version: {pydantic_ai_claude_code.__version__}")
        except:
            print("Version: unknown")
        print("=" * 80)

        result = agent.run_sync(user_prompt, model_settings=model_settings)
        application = result.output

        # Check results
        print(f"\nGenerated {len(application.fields)} fields\n")

        empty_count = 0
        required_empty = 0

        for i, field in enumerate(application.fields, 1):
            is_empty = len(field.content) == 0
            is_required = "REQUIRED" in field.description

            if is_empty:
                empty_count += 1
                if is_required:
                    required_empty += 1

            status = "❌ EMPTY" if is_empty else "✅ Content"
            req = "[REQ]" if is_required else "[OPT]"

            print(f"[{i}] {req} {field.name}")
            print(f"    {status}: {len(field.content)} chars")
            if not is_empty and len(field.content) < 200:
                print(f"    Preview: {field.content[:80]}...")
            print()

        # Summary
        print("=" * 80)
        if required_empty > 0:
            print(f"❌ BUG CONFIRMED: {required_empty} REQUIRED fields are EMPTY!")
            print(f"   Total empty: {empty_count}/{len(application.fields)}")
            print("\nThis is the v0.8.0 bug.")
            return 1
        elif empty_count > 0:
            print(f"⚠️  {empty_count} optional fields empty (expected if no content in source)")
            print(f"   All {len(application.fields) - empty_count} required fields populated")
            print("\nWorking correctly (v0.7.4 behavior)")
            return 0
        else:
            print(f"✅ SUCCESS: All {len(application.fields)} fields populated")
            print("\nWorking correctly (v0.7.4 behavior)")
            return 0

    finally:
        # Cleanup
        source_file.unlink(missing_ok=True)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("REPRODUCTION TEST FOR PYDANTIC-AI-CLAUDE-CODE BUG")
    print("=" * 80)
    print("\nThis test demonstrates the v0.8.0 regression where Claude creates")
    print("empty content fields instead of populating them from source material.")
    print("\nExpected behavior:")
    print("  - v0.8.0: ❌ All fields empty (BUG)")
    print("  - v0.7.4: ✅ Required fields populated (WORKING)")
    print("\n" + "=" * 80 + "\n")

    exit_code = test_bug()
    sys.exit(exit_code)
