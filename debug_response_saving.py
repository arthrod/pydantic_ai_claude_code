#!/usr/bin/env python3
"""Debug script to investigate empty response.result issue."""

import json
import logging
import tempfile
from pathlib import Path

from pydantic_ai import Agent

from pydantic_ai_claude_code.types import ClaudeCodeModelSettings

# Enable full debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """Run diagnostic test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Using temp directory: {tmpdir}")

        # Create agent
        agent = Agent("claude-code:sonnet")

        # Run a simple query with working_directory setting
        logger.info("Running agent query...")
        result = agent.run_sync(
            "What is 2+2? Just give the number.",
            model_settings=ClaudeCodeModelSettings(working_directory=tmpdir)
        )

        logger.info(f"Agent result.output: {result.output!r}")

        # Check subdirectories
        subdirs = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
        logger.info(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")

        if subdirs:
            subdir = subdirs[0]
            logger.info(f"Using subdir: {subdir}")

            # Check prompt.md
            prompt_file = subdir / "prompt.md"
            if prompt_file.exists():
                logger.info(f"prompt.md size: {prompt_file.stat().st_size} bytes")
                logger.info(f"prompt.md first 200 chars:\n{prompt_file.read_text()[:200]}")
            else:
                logger.error("prompt.md NOT FOUND")

            # Check response.json
            response_file = subdir / "response.json"
            if response_file.exists():
                logger.info(f"response.json size: {response_file.stat().st_size} bytes")
                response_content = response_file.read_text()
                logger.info(f"response.json raw content (first 500 chars):\n{response_content[:500]}")

                # Parse and examine
                response_data = json.loads(response_content)
                logger.info(f"response_data keys: {list(response_data.keys())}")
                logger.info(f"response_data['result'] type: {type(response_data.get('result'))}")
                logger.info(f"response_data['result'] value: {response_data.get('result')!r}")
                logger.info(f"response_data['result'] length: {len(response_data.get('result', ''))}")

                # Check usage
                if 'usage' in response_data:
                    logger.info(f"response_data['usage']: {response_data['usage']}")

                # Print full response_data for inspection
                logger.info(f"Full response_data:\n{json.dumps(response_data, indent=2)}")
            else:
                logger.error("response.json NOT FOUND")
        else:
            logger.error("No subdirectories found!")

        # Copy files to permanent location for inspection
        debug_dir = Path("/tmp/debug_response_saving")
        debug_dir.mkdir(exist_ok=True)

        if subdirs:
            import shutil
            dest_dir = debug_dir / "test_output"
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(subdirs[0], dest_dir)
            logger.info(f"\n✓ Files copied to: {dest_dir}")
        else:
            logger.info(f"\n✗ No files to copy")

if __name__ == "__main__":
    main()
