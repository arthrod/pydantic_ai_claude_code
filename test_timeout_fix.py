"""Test that timeout_seconds configuration works properly."""

from pydantic_ai_claude_code.provider import ClaudeCodeProvider


def test_timeout_in_provider_init():
    """Test that timeout_seconds is stored in provider."""
    provider = ClaudeCodeProvider(settings={"timeout_seconds": 1800})
    assert provider.timeout_seconds == 1800
    print("✓ Provider stores timeout_seconds: 1800")


def test_timeout_in_get_settings():
    """Test that timeout_seconds appears in get_settings()."""
    provider = ClaudeCodeProvider(settings={"timeout_seconds": 3600})
    settings = provider.get_settings()
    assert settings.get("timeout_seconds") == 3600
    print("✓ get_settings() returns timeout_seconds: 3600")


def test_timeout_default_value():
    """Test that timeout_seconds defaults to 900."""
    provider = ClaudeCodeProvider()
    assert provider.timeout_seconds == 900
    settings = provider.get_settings()
    assert settings.get("timeout_seconds") == 900
    print("✓ Default timeout_seconds: 900")


def test_timeout_with_overrides():
    """Test that timeout_seconds can be overridden in get_settings()."""
    provider = ClaudeCodeProvider(settings={"timeout_seconds": 600})
    settings = provider.get_settings(timeout_seconds=1200)
    assert settings.get("timeout_seconds") == 1200
    print("✓ get_settings() override works: 1200")


if __name__ == "__main__":
    test_timeout_in_provider_init()
    test_timeout_in_get_settings()
    test_timeout_default_value()
    test_timeout_with_overrides()
    print("\n✅ All timeout configuration tests passed!")
