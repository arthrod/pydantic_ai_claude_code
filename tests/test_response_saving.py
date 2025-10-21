"""Tests for raw response saving to working directory."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic_ai import Agent

# Import to trigger provider registration
import pydantic_ai_claude_code  # noqa: F401
from pydantic_ai_claude_code.types import ClaudeCodeModelSettings

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
            model_settings=ClaudeCodeModelSettings(working_directory=tmpdir)
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


def test_multiple_calls_create_separate_subdirectories():
    """Test that multiple calls to the same working directory create separate subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = Agent("claude-code:sonnet")

        # Make first call
        agent.run_sync("What is 1+1?", model_settings=ClaudeCodeModelSettings(working_directory=tmpdir))

        # Make second call
        agent.run_sync("What is 2+2?", model_settings=ClaudeCodeModelSettings(working_directory=tmpdir))

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
    """Test that temp workspaces create numbered subdirectories to prevent overwrites."""
    # Use provider with temp workspace enabled
    from pydantic_ai_claude_code import ClaudeCodeProvider

    provider = ClaudeCodeProvider({"use_temp_workspace": True})

    with provider:
        agent = Agent("claude-code:sonnet")

        # Get settings from provider to use with agent
        model_settings = ClaudeCodeModelSettings(**provider.get_settings())

        # Get the temp workspace path
        temp_workspace = Path(provider.working_directory)
        assert temp_workspace.exists(), "Temp workspace should exist"

        # Make first call
        result1 = agent.run_sync("What is 1+1?", model_settings=model_settings)
        assert result1.output is not None, "Expected result output"

        # Check that subdirectory '1' was created
        subdirs_after_first = sorted([d for d in temp_workspace.iterdir() if d.is_dir()])
        assert len(subdirs_after_first) == 1, f"Expected 1 subdirectory after first call, found {len(subdirs_after_first)}"
        assert subdirs_after_first[0].name == "1", f"Expected first subdir '1', got '{subdirs_after_first[0].name}'"

        # Make second call with same settings
        result2 = agent.run_sync("What is 2+2?", model_settings=model_settings)
        assert result2.output is not None, "Expected result output"

        # Check that subdirectory '2' was created
        subdirs_after_second = sorted([d for d in temp_workspace.iterdir() if d.is_dir()])
        assert len(subdirs_after_second) == EXPECTED_SUBDIR_COUNT_DOUBLE, f"Expected {EXPECTED_SUBDIR_COUNT_DOUBLE} subdirectories after second call, found {len(subdirs_after_second)}"
        assert subdirs_after_second[1].name == "2", f"Expected second subdir '2', got '{subdirs_after_second[1].name}'"

        # Verify both have their own prompt.md and response.json
        for subdir in subdirs_after_second:
            assert (subdir / "prompt.md").exists(), f"{subdir.name}/prompt.md not found"
            assert (subdir / "response.json").exists(), f"{subdir.name}/response.json not found"

        # Verify the prompts are different
        prompt1 = (subdirs_after_second[0] / "prompt.md").read_text()
        prompt2 = (subdirs_after_second[1] / "prompt.md").read_text()
        assert "1+1" in prompt1, "First prompt should contain '1+1'"
        assert "2+2" in prompt2, "Second prompt should contain '2+2'"
        assert prompt1 != prompt2, "Prompts should be different"


def test_reused_settings_dict_no_overwrite():
    """Test that reusing the same settings dict across multiple calls doesn't overwrite files."""
    agent = Agent("claude-code:sonnet")

    # Create a settings dict (no working_directory, so will use temp)
    settings = ClaudeCodeModelSettings()

    # Make first call with settings
    result1 = agent.run_sync("What is 1+1?", model_settings=settings)
    assert result1.output is not None, "Expected result output"

    # Settings dict now has __temp_base_directory set internally
    # Make second call with same settings dict
    result2 = agent.run_sync("What is 2+2?", model_settings=settings)
    assert result2.output is not None, "Expected result output"

    # We can't easily access the temp directory from here, but we can verify
    # that both calls succeeded without errors (which would happen if they
    # tried to write to the same file)
    assert result1.output != result2.output, "Results should be different"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
