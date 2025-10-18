"""Example demonstrating the additional_files feature.

This shows how to provide additional files to Claude that it can read
and reference during execution.
"""

from pathlib import Path

from pydantic_ai import Agent


def main():
    """Demonstrate additional_files feature."""

    # Create some example source files
    readme_content = """# Project README

This is a sample project demonstrating file analysis capabilities.

## Features
- File reading
- Content analysis
- Multi-file processing
"""

    code_content = """def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two integers.\"\"\"
    return a + b


def calculate_product(a: int, b: int) -> int:
    \"\"\"Calculate the product of two integers.\"\"\"
    return a * b
"""

    # Write files to temporary location
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "sources"
        source_dir.mkdir()

        readme_file = source_dir / "README.md"
        readme_file.write_text(readme_content)

        code_file = source_dir / "utils.py"
        code_file.write_text(code_content)

        # Create agent and run with additional files
        agent = Agent("claude-code:sonnet")

        print("=" * 80)
        print("Example 1: Analyzing code file")
        print("=" * 80)

        result = agent.run_sync(
            "Read utils.py and list all the function names you find.",
            model_settings={
                "additional_files": {
                    "utils.py": code_file,
                }
            },
        )
        print(f"Response: {result.output}\n")

        print("=" * 80)
        print("Example 2: Analyzing multiple files")
        print("=" * 80)

        result = agent.run_sync(
            "Read both README.md and utils.py. "
            "Tell me if the README accurately describes what the code does.",
            model_settings={
                "additional_files": {
                    "README.md": readme_file,
                    "utils.py": code_file,
                }
            },
        )
        print(f"Response: {result.output}\n")

        print("=" * 80)
        print("Example 3: Files in subdirectories")
        print("=" * 80)

        result = agent.run_sync(
            "List all files you can see and their locations.",
            model_settings={
                "additional_files": {
                    "docs/README.md": readme_file,
                    "src/utils.py": code_file,
                }
            },
        )
        print(f"Response: {result.output}\n")


if __name__ == "__main__":
    main()
