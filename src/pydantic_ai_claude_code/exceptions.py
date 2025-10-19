"""Exception classes for Claude Code model."""


class ClaudeOAuthError(RuntimeError):
    """Raised when Claude Code CLI OAuth token is expired or revoked.

    This exception is raised when the Claude CLI returns an authentication error,
    typically indicating that the OAuth token has expired and the user needs to
    run /login to re-authenticate.

    Attributes:
        message: The error message from Claude CLI
        reauth_instruction: Instructions for re-authentication (e.g., "Please run /login")

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_claude_code import ClaudeOAuthError

        agent = Agent("claude-code:sonnet")

        try:
            result = agent.run_sync("What is 2+2?")
        except ClaudeOAuthError as e:
            print(f"Authentication expired: {e}")
            print(f"Action required: {e.reauth_instruction}")
            # Prompt user to run /login, then retry
        except RuntimeError as e:
            # Handle other CLI errors
            print(f"CLI error: {e}")
        ```
    """

    def __init__(self, message: str, reauth_instruction: str = "Please run /login"):
        """Initialize ClaudeOAuthError.

        Args:
            message: The error message from Claude CLI
            reauth_instruction: Instructions for re-authentication (default: "Please run /login")
        """
        super().__init__(message)
        self.reauth_instruction = reauth_instruction
