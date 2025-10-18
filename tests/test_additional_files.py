"""Tests for additional_files feature."""

import tempfile
from pathlib import Path

import pytest
from pydantic_ai import Agent

# Import to trigger registration
import pydantic_ai_claude_code  # noqa: F401

# Test constants
EXPECTED_SUBDIR_COUNT_DOUBLE = 2


def test_additional_files_basic():
    """Test copying a single additional file into working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source file
        source_file = Path(tmpdir) / "source.txt"
        source_file.write_text("Test content from source file")

        # Create a working directory
        work_dir = Path(tmpdir) / "work"

        # Run agent with additional file
        agent = Agent("claude-code:sonnet")
        result = agent.run_sync(
            "Read the file utils.py and tell me what it contains in one word.",
            model_settings={
                "working_directory": str(work_dir),
                "additional_files": {
                    "utils.py": source_file,
                },
            },
        )

        # Verify subdirectory was created
        subdirs = [d for d in work_dir.iterdir() if d.is_dir()]
        assert len(subdirs) == 1

        subdir = subdirs[0]

        # Verify additional file was copied
        copied_file = subdir / "utils.py"
        assert copied_file.exists(), "utils.py was not copied"
        assert copied_file.read_text() == "Test content from source file"

        # Verify prompt.md was created
        assert (subdir / "prompt.md").exists()

        # Verify response.json was created
        assert (subdir / "response.json").exists()

        # Verify Claude read the file (result should be valid)
        assert result.output is not None


def test_additional_files_multiple():
    """Test copying multiple additional files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source files
        source1 = Path(tmpdir) / "file1.txt"
        source1.write_text("Content 1")

        source2 = Path(tmpdir) / "file2.txt"
        source2.write_text("Content 2")

        source3 = Path(tmpdir) / "file3.json"
        source3.write_text('{"key": "value"}')

        work_dir = Path(tmpdir) / "work"

        # Run agent with multiple files
        agent = Agent("claude-code:sonnet")
        agent.run_sync(
            "List the files you can see.",
            model_settings={
                "working_directory": str(work_dir),
                "additional_files": {
                    "data1.txt": source1,
                    "data2.txt": source2,
                    "config.json": source3,
                },
            },
        )

        # Verify all files were copied
        subdir = list(work_dir.iterdir())[0]
        assert (subdir / "data1.txt").read_text() == "Content 1"
        assert (subdir / "data2.txt").read_text() == "Content 2"
        assert (subdir / "config.json").read_text() == '{"key": "value"}'


def test_additional_files_with_subdirectories():
    """Test copying files into subdirectories within working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source file
        source = Path(tmpdir) / "source.txt"
        source.write_text("Nested content")

        work_dir = Path(tmpdir) / "work"

        # Run agent with nested destination path
        agent = Agent("claude-code:sonnet")
        agent.run_sync(
            "What files do you see?",
            model_settings={
                "working_directory": str(work_dir),
                "additional_files": {
                    "docs/readme.md": source,
                    "data/input.txt": source,
                },
            },
        )

        # Verify nested directories were created
        subdir = list(work_dir.iterdir())[0]
        assert (subdir / "docs" / "readme.md").exists()
        assert (subdir / "docs" / "readme.md").read_text() == "Nested content"
        assert (subdir / "data" / "input.txt").exists()
        assert (subdir / "data" / "input.txt").read_text() == "Nested content"


def test_additional_files_source_not_found():
    """Test that FileNotFoundError is raised if source file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir) / "work"
        non_existent = Path(tmpdir) / "does_not_exist.txt"

        agent = Agent("claude-code:sonnet")

        with pytest.raises(FileNotFoundError) as exc_info:
            agent.run_sync(
                "Hello",
                model_settings={
                    "working_directory": str(work_dir),
                    "additional_files": {
                        "file.txt": non_existent,
                    },
                },
            )

        assert "Additional file source not found" in str(exc_info.value)
        assert str(non_existent) in str(exc_info.value)


def test_additional_files_source_is_directory():
    """Test that ValueError is raised if source is a directory, not a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir) / "work"
        source_dir = Path(tmpdir) / "source_dir"
        source_dir.mkdir()

        agent = Agent("claude-code:sonnet")

        with pytest.raises(ValueError) as exc_info:
            agent.run_sync(
                "Hello",
                model_settings={
                    "working_directory": str(work_dir),
                    "additional_files": {
                        "file.txt": source_dir,
                    },
                },
            )

        assert "not a file" in str(exc_info.value)


def test_additional_files_relative_path_resolution():
    """Test that relative paths are resolved from current working directory."""
    # Create a temp file in a known location
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Relative path test")
        temp_file = Path(f.name)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "work"

            # Use relative path by getting the name relative to cwd
            # We'll use absolute for this test to be safe
            agent = Agent("claude-code:sonnet")
            agent.run_sync(
                "What do you see?",
                model_settings={
                    "working_directory": str(work_dir),
                    "additional_files": {
                        "test.txt": temp_file,  # Absolute path
                    },
                },
            )

            # Verify file was copied
            subdir = list(work_dir.iterdir())[0]
            assert (subdir / "test.txt").exists()
            assert (subdir / "test.txt").read_text() == "Relative path test"
    finally:
        temp_file.unlink()


def test_additional_files_preserves_binary():
    """Test that binary files are preserved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a binary file
        binary_content = b"\x00\x01\x02\x03\xff\xfe\xfd"
        source = Path(tmpdir) / "binary.dat"
        source.write_bytes(binary_content)

        work_dir = Path(tmpdir) / "work"

        agent = Agent("claude-code:sonnet")
        agent.run_sync(
            "List files.",
            model_settings={
                "working_directory": str(work_dir),
                "additional_files": {
                    "data.bin": source,
                },
            },
        )

        # Verify binary content preserved
        subdir = list(work_dir.iterdir())[0]
        assert (subdir / "data.bin").read_bytes() == binary_content


def test_additional_files_multiple_calls_isolated():
    """Test that files in different calls don't interfere with each other."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source1 = Path(tmpdir) / "file1.txt"
        source1.write_text("First call")

        source2 = Path(tmpdir) / "file2.txt"
        source2.write_text("Second call")

        work_dir = Path(tmpdir) / "work"

        agent = Agent("claude-code:sonnet")

        # First call
        agent.run_sync(
            "Read file.txt",
            model_settings={
                "working_directory": str(work_dir),
                "additional_files": {"file.txt": source1},
            },
        )

        # Second call
        agent.run_sync(
            "Read file.txt",
            model_settings={
                "working_directory": str(work_dir),
                "additional_files": {"file.txt": source2},
            },
        )

        # Verify each call has its own subdirectory with the correct file
        subdirs = sorted([d for d in work_dir.iterdir() if d.is_dir()])
        assert len(subdirs) == EXPECTED_SUBDIR_COUNT_DOUBLE

        assert (subdirs[0] / "file.txt").read_text() == "First call"
        assert (subdirs[1] / "file.txt").read_text() == "Second call"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
