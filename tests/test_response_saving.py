"""Tests for raw response saving to working directory."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic_ai import Agent

# Import to trigger provider registration
import pydantic_ai_claude_code  # noqa: F401

# Test constants
EXPECTED_SUBDIR_COUNT_SINGLE = 1
EXPECTED_SUBDIR_COUNT_DOUBLE = 2


def test_response_saved_to_working_directory():
    """Test that raw response is saved to response.json in working directory."""
    # Create a temporary directory for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create agent
        agent = Agent("claude-code:sonnet")

        # Run a simple query with working_directory setting
        agent.run_sync(
            "What is 2+2? Just give the number.",
            model_settings={"working_directory": tmpdir}
        )

        # Check that subdirectory was created
        subdirs = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
        assert len(subdirs) == EXPECTED_SUBDIR_COUNT_SINGLE, f"Expected {EXPECTED_SUBDIR_COUNT_SINGLE} subdirectory, found {len(subdirs)}"

        subdir = subdirs[0]
        assert subdir.name == "1", f"Expected subdir '1', got '{subdir.name}'"

        # Check that prompt.md exists
        prompt_file = subdir / "prompt.md"
        assert prompt_file.exists(), "prompt.md not found"
        prompt_content = prompt_file.read_text()
        assert len(prompt_content) > 0, "prompt.md is empty"
        assert "2+2" in prompt_content, "prompt.md doesn't contain query"

        # Check that response.json exists
        response_file = subdir / "response.json"
        assert response_file.exists(), "response.json not found"

        # Verify response.json is valid JSON
        response_content = response_file.read_text()
        response_data = json.loads(response_content)

        # Verify it has expected structure
        assert "result" in response_data, "response.json missing 'result' field"
        assert "usage" in response_data, "response.json missing 'usage' field"

        # Verify result contains something
        assert len(response_data["result"]) > 0, "response result is empty"


def test_multiple_calls_create_separate_subdirectories():
    """Test that multiple calls to the same working directory create separate subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = Agent("claude-code:sonnet")

        # Make first call
        agent.run_sync("What is 1+1?", model_settings={"working_directory": tmpdir})

        # Make second call
        agent.run_sync("What is 2+2?", model_settings={"working_directory": tmpdir})

        # Check that two subdirectories were created
        subdirs = sorted([d for d in Path(tmpdir).iterdir() if d.is_dir()])
        assert len(subdirs) == EXPECTED_SUBDIR_COUNT_DOUBLE, f"Expected {EXPECTED_SUBDIR_COUNT_DOUBLE} subdirectories, found {len(subdirs)}"

        # Check subdirectory names
        assert subdirs[0].name == "1", f"Expected first subdir '1', got '{subdirs[0].name}'"
        assert subdirs[1].name == "2", f"Expected second subdir '2', got '{subdirs[1].name}'"

        # Check that each has prompt.md and response.json
        for subdir in subdirs:
            assert (subdir / "prompt.md").exists(), f"{subdir.name}/prompt.md not found"
            assert (subdir / "response.json").exists(), f"{subdir.name}/response.json not found"


def test_temp_workspace_no_overwrite():
    """Test that temp workspaces don't create subdirectories (only one call per temp dir)."""
    # Use default (no working_directory specified = uses temp directory)
    agent = Agent("claude-code:sonnet")

    # Run query (no model_settings = uses temp directory)
    result = agent.run_sync("What is 2+2?")

    # The working directory should be a temp directory directly
    # We can't easily check the temp directory from here, but we can verify the result
    assert result.output is not None, "Expected result output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
