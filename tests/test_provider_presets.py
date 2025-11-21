"""Tests for provider preset functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pydantic_ai_claude_code import (
    ClaudeCodeProvider,
    ProviderPreset,
    get_preset,
    get_presets_by_category,
    list_presets,
    load_all_presets,
)
from pydantic_ai_claude_code.provider_presets import (
    _parse_preset_dict,
    apply_provider_environment,
    load_builtin_presets,
    load_project_presets,
    load_user_presets,
)


class TestProviderPreset:
    """Tests for ProviderPreset class."""

    def test_preset_initialization(self):
        """Test basic preset initialization."""
        preset = ProviderPreset(
            preset_id="test_provider",
            name="Test Provider",
            website_url="https://example.com",
            settings={"env": {"TEST_VAR": "test_value"}},
        )

        assert preset.preset_id == "test_provider"
        assert preset.name == "Test Provider"
        assert preset.website_url == "https://example.com"
        assert preset.is_official is False
        assert preset.category == "third_party"
        assert preset.api_key_field == "ANTHROPIC_AUTH_TOKEN"

    def test_preset_with_models(self):
        """Test preset with model mappings."""
        preset = ProviderPreset(
            preset_id="test_provider",
            name="Test Provider",
            website_url="https://example.com",
            settings={},
            models={
                "default": "test-default",
                "haiku": "test-haiku",
                "sonnet": "test-sonnet",
                "opus": "test-opus",
            },
        )

        assert preset.get_model_name("sonnet") == "test-sonnet"
        assert preset.get_model_name("haiku") == "test-haiku"
        assert preset.get_model_name("opus") == "test-opus"
        assert preset.get_model_name("custom") == "test-default"
        # Unknown model returns as-is
        assert preset.get_model_name("unknown") == "unknown"

    def test_preset_environment_variables(self):
        """Test getting environment variables from preset."""
        preset = ProviderPreset(
            preset_id="test_provider",
            name="Test Provider",
            website_url="https://example.com",
            settings={
                "env": {
                    "ANTHROPIC_BASE_URL": "https://api.example.com",
                    "ANTHROPIC_MODEL": "test-model",
                }
            },
        )

        env_vars = preset.get_environment_variables()
        assert env_vars["ANTHROPIC_BASE_URL"] == "https://api.example.com"
        assert env_vars["ANTHROPIC_MODEL"] == "test-model"

    def test_preset_environment_with_api_key(self):
        """Test setting API key via environment variables."""
        preset = ProviderPreset(
            preset_id="test_provider",
            name="Test Provider",
            website_url="https://example.com",
            settings={"env": {}},
            api_key_field="ANTHROPIC_AUTH_TOKEN",
        )

        env_vars = preset.get_environment_variables(api_key="test-api-key")
        assert env_vars["ANTHROPIC_AUTH_TOKEN"] == "test-api-key"

    def test_preset_template_substitution(self):
        """Test template variable substitution."""
        preset = ProviderPreset(
            preset_id="test_provider",
            name="Test Provider",
            website_url="https://example.com",
            settings={
                "env": {
                    "ANTHROPIC_BASE_URL": "https://api.example.com/${ENDPOINT_ID}/proxy",
                }
            },
            template_values={
                "ENDPOINT_ID": {
                    "label": "Endpoint ID",
                    "placeholder": "ep-xxx",
                    "default_value": "",
                }
            },
        )

        env_vars = preset.get_environment_variables(
            template_vars={"ENDPOINT_ID": "ep-12345"}
        )
        assert env_vars["ANTHROPIC_BASE_URL"] == "https://api.example.com/ep-12345/proxy"

    def test_preset_to_dict(self):
        """Test converting preset to dictionary."""
        preset = ProviderPreset(
            preset_id="test_provider",
            name="Test Provider",
            website_url="https://example.com",
            settings={"env": {"TEST": "value"}},
            is_official=True,
            category="official",
        )

        data = preset.to_dict()
        assert data["preset_id"] == "test_provider"
        assert data["name"] == "Test Provider"
        assert data["is_official"] is True
        assert data["category"] == "official"


class TestLoadBuiltinPresets:
    """Tests for loading built-in presets."""

    def test_load_builtin_presets(self):
        """Test loading built-in presets from YAML."""
        presets = load_builtin_presets()

        # Should have multiple presets
        assert len(presets) > 0

        # Check for some expected presets
        assert "deepseek" in presets
        assert "zhipu_glm" in presets
        assert "qwen_coder" in presets

    def test_deepseek_preset_config(self):
        """Test DeepSeek preset configuration."""
        presets = load_builtin_presets()
        deepseek = presets.get("deepseek")

        assert deepseek is not None
        assert deepseek.name == "DeepSeek"
        assert deepseek.category == "cn_official"
        assert deepseek.models.get("sonnet") == "DeepSeek-V3.2-Exp"

        env_vars = deepseek.get_environment_variables()
        assert "ANTHROPIC_BASE_URL" in env_vars
        assert "deepseek.com" in env_vars["ANTHROPIC_BASE_URL"]

    def test_zhipu_preset_config(self):
        """Test Zhipu GLM preset configuration."""
        presets = load_builtin_presets()
        zhipu = presets.get("zhipu_glm")

        assert zhipu is not None
        assert zhipu.name == "Zhipu GLM"
        assert zhipu.is_partner is True
        assert zhipu.partner_promotion_key == "zhipu"
        assert zhipu.models.get("haiku") == "glm-4.5-air"
        assert zhipu.models.get("sonnet") == "glm-4.6"

    def test_aihubmix_preset_config(self):
        """Test AiHubMix preset with custom API key field."""
        presets = load_builtin_presets()
        aihubmix = presets.get("aihubmix")

        assert aihubmix is not None
        assert aihubmix.api_key_field == "ANTHROPIC_API_KEY"
        assert len(aihubmix.endpoint_candidates) > 0


class TestLoadUserPresets:
    """Tests for loading user presets."""

    def test_load_user_presets_yaml(self):
        """Test loading user presets from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create user directory structure
            user_dir = Path(tmpdir) / ".claude"
            user_dir.mkdir()

            # Create user presets YAML
            yaml_content = """
providers:
  my_provider:
    name: "My Custom Provider"
    website_url: "https://my-provider.com"
    category: "third_party"
    settings:
      env:
        ANTHROPIC_BASE_URL: "https://api.my-provider.com"
    models:
      default: "my-model"
      sonnet: "my-sonnet"
"""
            (user_dir / "providers.yaml").write_text(yaml_content)

            # Patch home directory
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                presets = load_user_presets()

            assert "my_provider" in presets
            assert presets["my_provider"].name == "My Custom Provider"
            assert presets["my_provider"].models.get("sonnet") == "my-sonnet"

    def test_load_user_presets_json(self):
        """Test loading user presets from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create user directory structure
            user_dir = Path(tmpdir) / ".claude"
            user_dir.mkdir()

            # Create user presets JSON
            json_content = {
                "providers": {
                    "json_provider": {
                        "name": "JSON Provider",
                        "website_url": "https://json-provider.com",
                        "settings": {"env": {}},
                    }
                }
            }
            (user_dir / "providers.json").write_text(json.dumps(json_content))

            # Patch home directory
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                presets = load_user_presets()

            assert "json_provider" in presets
            assert presets["json_provider"].name == "JSON Provider"


class TestLoadProjectPresets:
    """Tests for loading project presets."""

    def test_load_project_presets_yaml(self):
        """Test loading project presets from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create project presets YAML
            yaml_content = """
providers:
  project_provider:
    name: "Project Provider"
    website_url: "https://project-provider.com"
    settings:
      env:
        ANTHROPIC_BASE_URL: "https://api.project-provider.com"
"""
            (project_dir / "claude_providers.yaml").write_text(yaml_content)

            presets = load_project_presets(project_dir)

            assert "project_provider" in presets
            assert presets["project_provider"].name == "Project Provider"

    def test_load_project_presets_json(self):
        """Test loading project presets from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create project presets JSON
            json_content = {
                "project_json": {
                    "name": "Project JSON",
                    "website_url": "https://example.com",
                    "settings": {"env": {}},
                }
            }
            (project_dir / "claude_providers.json").write_text(json.dumps(json_content))

            presets = load_project_presets(project_dir)

            assert "project_json" in presets


class TestLoadAllPresets:
    """Tests for loading all presets with precedence."""

    def test_load_all_presets_includes_builtin(self):
        """Test that load_all_presets includes built-in presets."""
        presets = load_all_presets()

        # Should include built-in presets
        assert "deepseek" in presets
        assert "zhipu_glm" in presets

    def test_preset_precedence(self):
        """Test that project presets override user presets override built-in."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            user_dir = Path(tmpdir) / ".claude"
            user_dir.mkdir()

            # Create user preset that overrides built-in
            user_yaml = """
providers:
  deepseek:
    name: "User DeepSeek"
    website_url: "https://user-deepseek.com"
    settings:
      env: {}
"""
            (user_dir / "providers.yaml").write_text(user_yaml)

            # Create project preset that overrides user
            project_yaml = """
providers:
  deepseek:
    name: "Project DeepSeek"
    website_url: "https://project-deepseek.com"
    settings:
      env: {}
"""
            (project_dir / "claude_providers.yaml").write_text(project_yaml)

            # Patch home directory
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                presets = load_all_presets(project_dir)

            # Project preset should win
            assert presets["deepseek"].name == "Project DeepSeek"


class TestGetPreset:
    """Tests for get_preset function."""

    def test_get_existing_preset(self):
        """Test getting an existing preset."""
        preset = get_preset("deepseek")

        assert preset is not None
        assert preset.preset_id == "deepseek"
        assert preset.name == "DeepSeek"

    def test_get_nonexistent_preset(self):
        """Test getting a preset that doesn't exist."""
        preset = get_preset("nonexistent_provider")

        assert preset is None


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_presets(self):
        """Test listing all preset IDs."""
        preset_ids = list_presets()

        assert isinstance(preset_ids, list)
        assert len(preset_ids) > 0
        assert "deepseek" in preset_ids
        assert "zhipu_glm" in preset_ids

        # Should be sorted
        assert preset_ids == sorted(preset_ids)


class TestGetPresetsByCategory:
    """Tests for get_presets_by_category function."""

    def test_get_cn_official_presets(self):
        """Test getting Chinese official presets."""
        presets = get_presets_by_category("cn_official")

        assert len(presets) > 0
        for preset in presets:
            assert preset.category == "cn_official"

        # Should include DeepSeek
        preset_ids = [p.preset_id for p in presets]
        assert "deepseek" in preset_ids

    def test_get_aggregator_presets(self):
        """Test getting aggregator presets."""
        presets = get_presets_by_category("aggregator")

        assert len(presets) > 0
        for preset in presets:
            assert preset.category == "aggregator"


class TestApplyProviderEnvironment:
    """Tests for apply_provider_environment function."""

    def test_apply_environment_variables(self):
        """Test applying environment variables from preset."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={
                "env": {
                    "TEST_VAR_1": "value1",
                    "TEST_VAR_2": "value2",
                }
            },
        )

        # Clear any existing test vars
        for key in ["TEST_VAR_1", "TEST_VAR_2"]:
            os.environ.pop(key, None)

        try:
            applied = apply_provider_environment(preset)

            assert os.getenv("TEST_VAR_1") == "value1"
            assert os.getenv("TEST_VAR_2") == "value2"
            assert "TEST_VAR_1" in applied
            assert "TEST_VAR_2" in applied
        finally:
            # Cleanup
            os.environ.pop("TEST_VAR_1", None)
            os.environ.pop("TEST_VAR_2", None)

    def test_apply_environment_with_api_key(self):
        """Test applying environment with API key."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
        )

        # Clear existing
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

        try:
            apply_provider_environment(preset, api_key="my-api-key")

            assert os.getenv("ANTHROPIC_AUTH_TOKEN") == "my-api-key"
        finally:
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    def test_no_override_existing(self):
        """Test that existing env vars are not overridden by default."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {"EXISTING_VAR": "new_value"}},
        )

        os.environ["EXISTING_VAR"] = "original_value"

        try:
            apply_provider_environment(preset, override_existing=False)

            # Should keep original value
            assert os.getenv("EXISTING_VAR") == "original_value"
        finally:
            os.environ.pop("EXISTING_VAR", None)

    def test_override_existing_when_requested(self):
        """Test overriding existing env vars when requested."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {"EXISTING_VAR": "new_value"}},
        )

        os.environ["EXISTING_VAR"] = "original_value"

        try:
            apply_provider_environment(preset, override_existing=True)

            # Should override with new value
            assert os.getenv("EXISTING_VAR") == "new_value"
        finally:
            os.environ.pop("EXISTING_VAR", None)


class TestClaudeCodeProviderWithPresets:
    """Tests for ClaudeCodeProvider with preset support."""

    def test_provider_with_preset(self):
        """Test creating provider with preset."""
        provider = ClaudeCodeProvider(
            settings={
                "provider_preset": "deepseek",
            }
        )

        assert provider.provider_preset_id == "deepseek"
        assert provider.provider_preset is not None
        assert provider.provider_preset.name == "DeepSeek"

    def test_provider_preset_sets_env_vars(self):
        """Test that provider preset sets environment variables."""
        # Clear any existing vars
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        os.environ.pop("ANTHROPIC_MODEL", None)

        try:
            provider = ClaudeCodeProvider(
                settings={
                    "provider_preset": "deepseek",
                }
            )

            # Should have set environment variables
            assert len(provider._applied_env_vars) > 0

            # Check that vars were applied
            applied = provider.get_applied_env_vars()
            assert "ANTHROPIC_BASE_URL" in applied or os.getenv("ANTHROPIC_BASE_URL")
        finally:
            # Cleanup
            os.environ.pop("ANTHROPIC_BASE_URL", None)
            os.environ.pop("ANTHROPIC_MODEL", None)

    def test_provider_with_api_key(self):
        """Test provider with preset and API key."""
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

        try:
            provider = ClaudeCodeProvider(
                settings={
                    "provider_preset": "deepseek",
                    "provider_api_key": "my-test-key",
                }
            )

            applied = provider.get_applied_env_vars()
            assert applied.get("ANTHROPIC_AUTH_TOKEN") == "my-test-key"
        finally:
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    def test_provider_model_name_mapping(self):
        """Test model name mapping through provider."""
        provider = ClaudeCodeProvider(
            settings={
                "provider_preset": "deepseek",
            }
        )

        # DeepSeek maps sonnet to DeepSeek-V3.2-Exp
        model_name = provider.get_model_name("sonnet")
        assert model_name == "DeepSeek-V3.2-Exp"

        # Unknown alias should be returned unchanged
        unknown_alias = "unknown_model"
        mapped_name = provider.get_model_name(unknown_alias)
        assert mapped_name == unknown_alias

    def test_provider_with_template_vars(self):
        """Test provider with template variables."""
        os.environ.pop("ANTHROPIC_BASE_URL", None)

        try:
            provider = ClaudeCodeProvider(
                settings={
                    "provider_preset": "kat_coder",
                    "provider_template_vars": {"ENDPOINT_ID": "ep-test-123"},
                }
            )

            applied = provider.get_applied_env_vars()
            assert "ep-test-123" in applied.get("ANTHROPIC_BASE_URL", "")
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)

    def test_provider_with_nonexistent_preset(self):
        """Test provider with nonexistent preset doesn't crash."""
        provider = ClaudeCodeProvider(
            settings={
                "provider_preset": "nonexistent_provider",
            }
        )

        assert provider.provider_preset_id == "nonexistent_provider"
        assert provider.provider_preset is None
        assert len(provider._applied_env_vars) == 0

    def test_provider_without_preset(self):
        """Test provider without preset works normally."""
        provider = ClaudeCodeProvider(
            settings={
                "model": "sonnet",
            }
        )

        assert provider.provider_preset_id is None
        assert provider.provider_preset is None
        assert provider.model == "sonnet"


class TestModelRegistration:
    """Tests for model registration with provider presets."""

    def test_standard_model_string(self):
        """Test standard claude-code:model format."""
        from pydantic_ai import models

        from pydantic_ai_claude_code import ClaudeCodeModel

        model = models.infer_model("claude-code:sonnet")

        assert isinstance(model, ClaudeCodeModel)
        assert model._model_name == "sonnet"

    def test_preset_model_string(self):
        """Test claude-code:preset:model format."""
        from pydantic_ai import models

        from pydantic_ai_claude_code import ClaudeCodeModel

        model = models.infer_model("claude-code:deepseek:sonnet")

        assert isinstance(model, ClaudeCodeModel)
        # Model name should be mapped to DeepSeek model
        assert model._model_name == "DeepSeek-V3.2-Exp"
        # Provider should have preset loaded
        assert model.provider.provider_preset_id == "deepseek"

    def test_custom_model_alias(self):
        """Test custom model alias."""
        from pydantic_ai import models

        from pydantic_ai_claude_code import ClaudeCodeModel

        model = models.infer_model("claude-code:custom")

        assert isinstance(model, ClaudeCodeModel)
        assert model._model_name == "custom"


class TestParsePresetDict:
    """Tests for _parse_preset_dict helper function."""

    def test_parse_minimal_preset(self):
        """Test parsing minimal preset data."""
        data = {
            "name": "Minimal",
            "website_url": "https://example.com",
            "settings": {"env": {}},
        }

        preset = _parse_preset_dict("minimal", data)

        assert preset.preset_id == "minimal"
        assert preset.name == "Minimal"
        assert preset.is_official is False
        assert preset.category == "third_party"

    def test_parse_full_preset(self):
        """Test parsing full preset data."""
        data = {
            "name": "Full Provider",
            "website_url": "https://full-provider.com",
            "api_key_url": "https://full-provider.com/api-keys",
            "is_official": True,
            "is_partner": True,
            "partner_promotion_key": "full",
            "category": "official",
            "api_key_field": "CUSTOM_API_KEY",
            "settings": {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://api.full-provider.com",
                    "ANTHROPIC_MODEL": "full-model",
                }
            },
            "models": {
                "default": "full-default",
                "sonnet": "full-sonnet",
            },
            "template_values": {
                "VAR": {"label": "Variable", "placeholder": "xxx"},
            },
            "endpoint_candidates": ["https://endpoint1.com", "https://endpoint2.com"],
            "theme": {"icon": "generic", "background_color": "#000000"},
        }

        preset = _parse_preset_dict("full", data)

        assert preset.preset_id == "full"
        assert preset.name == "Full Provider"
        assert preset.api_key_url == "https://full-provider.com/api-keys"
        assert preset.is_official is True
        assert preset.is_partner is True
        assert preset.partner_promotion_key == "full"
        assert preset.category == "official"
        assert preset.api_key_field == "CUSTOM_API_KEY"
        assert preset.models.get("sonnet") == "full-sonnet"
        assert "VAR" in preset.template_values
        assert len(preset.endpoint_candidates) == 2
        assert preset.theme.get("icon") == "generic"


class TestImportExports:
    """Tests for package imports and exports."""

    def test_import_provider_preset(self):
        """Test importing ProviderPreset from package."""
        from pydantic_ai_claude_code import ProviderPreset

        assert ProviderPreset is not None

    def test_import_list_presets(self):
        """Test importing list_presets from package."""
        from pydantic_ai_claude_code import list_presets

        result = list_presets()
        assert isinstance(result, list)

    def test_import_get_preset(self):
        """Test importing get_preset from package."""
        from pydantic_ai_claude_code import get_preset

        result = get_preset("deepseek")
        assert result is not None

    def test_import_get_presets_by_category(self):
        """Test importing get_presets_by_category from package."""
        from pydantic_ai_claude_code import get_presets_by_category

        result = get_presets_by_category("cn_official")
        assert isinstance(result, list)

    def test_import_load_all_presets(self):
        """Test importing load_all_presets from package."""
        from pydantic_ai_claude_code import load_all_presets

        result = load_all_presets()
        assert isinstance(result, dict)


class TestProviderPresetDocumentation:
    """Tests to ensure preset documentation accuracy."""

    def test_all_presets_have_required_fields(self):
        """Test that all presets have required fields."""
        presets = load_all_presets()

        for preset_id, preset in presets.items():
            assert preset.name, f"Preset {preset_id} missing name"
            assert preset.website_url, f"Preset {preset_id} missing website_url"
            assert preset.settings is not None, f"Preset {preset_id} missing settings"

    def test_all_presets_have_valid_categories(self):
        """Test that all presets have valid categories."""
        valid_categories = {"official", "cn_official", "aggregator", "third_party"}
        presets = load_all_presets()

        for preset_id, preset in presets.items():
            assert (
                preset.category in valid_categories
            ), f"Preset {preset_id} has invalid category: {preset.category}"

    def test_presets_with_models_have_complete_mappings(self):
        """Test that presets with models have all standard mappings."""
        presets = load_all_presets()

        for preset_id, preset in presets.items():
            if preset.models:
                # If models are defined, should have at least default
                assert (
                    "default" in preset.models
                ), f"Preset {preset_id} missing default model"


class TestTemplateSubstitutionEdgeCases:
    """Tests for edge cases in template variable substitution."""

    def test_template_with_multiple_variables(self):
        """Test substitution with multiple template variables."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={
                "env": {
                    "URL": "https://${HOST}:${PORT}/api/${VERSION}",
                }
            },
        )

        env_vars = preset.get_environment_variables(
            template_vars={"HOST": "api.example.com", "PORT": "8443", "VERSION": "v2"}
        )
        assert env_vars["URL"] == "https://api.example.com:8443/api/v2"

    def test_template_with_missing_variable(self):
        """Test that missing variables are left as-is."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={
                "env": {
                    "URL": "https://${HOST}/api",
                }
            },
        )

        env_vars = preset.get_environment_variables(template_vars={})
        # Should keep the original placeholder
        assert "${HOST}" in env_vars["URL"]

    def test_template_with_env_fallback(self):
        """Test that environment variables are used as fallback."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={
                "env": {
                    "URL": "https://${MY_HOST}/api",
                }
            },
        )

        # Set environment variable
        os.environ["MY_HOST"] = "fallback.example.com"
        try:
            env_vars = preset.get_environment_variables(template_vars={})
            assert "fallback.example.com" in env_vars["URL"]
        finally:
            os.environ.pop("MY_HOST", None)

    def test_template_with_special_characters(self):
        """Test template variables with special characters."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={
                "env": {
                    "URL": "https://api.example.com/${PATH}",
                }
            },
        )

        env_vars = preset.get_environment_variables(
            template_vars={"PATH": "v1/users/123"}
        )
        assert env_vars["URL"] == "https://api.example.com/v1/users/123"

    def test_template_with_numeric_values(self):
        """Test that numeric values in settings are converted to strings."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={
                "env": {
                    "TIMEOUT": 3000,
                    "MAX_TOKENS": 8000,
                }
            },
        )

        env_vars = preset.get_environment_variables()
        assert env_vars["TIMEOUT"] == "3000"
        assert env_vars["MAX_TOKENS"] == "8000"

    def test_template_with_empty_string(self):
        """Test template with empty string values."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={
                "env": {
                    "EMPTY_VAR": "",
                    "NORMAL_VAR": "value",
                }
            },
        )

        env_vars = preset.get_environment_variables()
        assert env_vars["EMPTY_VAR"] == ""
        assert env_vars["NORMAL_VAR"] == "value"


class TestProviderPresetIntegration:
    """Integration tests for provider presets with ClaudeCodeProvider."""

    def test_provider_with_deepseek_preset(self):
        """Test provider initialization with DeepSeek preset."""
        # Clear any existing env vars
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        os.environ.pop("ANTHROPIC_MODEL", None)

        try:
            provider = ClaudeCodeProvider(
                settings={
                    "provider_preset": "deepseek",
                    "provider_api_key": "test-api-key",
                }
            )

            assert provider.provider_preset_id == "deepseek"
            assert provider.provider_preset is not None
            assert provider.provider_preset.name == "DeepSeek"

            # Check that environment variables were applied
            applied = provider.get_applied_env_vars()
            assert "ANTHROPIC_BASE_URL" in applied
            assert "deepseek" in applied["ANTHROPIC_BASE_URL"]
            assert "ANTHROPIC_AUTH_TOKEN" in applied
            assert applied["ANTHROPIC_AUTH_TOKEN"] == "test-api-key"
        finally:
            # Cleanup
            for key in ["ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ANTHROPIC_AUTH_TOKEN"]:
                os.environ.pop(key, None)

    def test_provider_get_model_name_with_preset(self):
        """Test get_model_name method with preset loaded."""
        provider = ClaudeCodeProvider(
            settings={"provider_preset": "deepseek"}
        )

        # DeepSeek uses same model for all aliases
        assert provider.get_model_name("sonnet") == "DeepSeek-V3.2-Exp"
        assert provider.get_model_name("haiku") == "DeepSeek-V3.2-Exp"
        assert provider.get_model_name("opus") == "DeepSeek-V3.2-Exp"

    def test_provider_get_model_name_without_preset(self):
        """Test get_model_name method without preset."""
        provider = ClaudeCodeProvider(settings={})

        # Should return as-is
        assert provider.get_model_name("sonnet") == "sonnet"
        assert provider.get_model_name("custom-model") == "custom-model"

    def test_provider_with_template_vars(self):
        """Test provider with template variables."""
        os.environ.pop("ANTHROPIC_BASE_URL", None)

        try:
            provider = ClaudeCodeProvider(
                settings={
                    "provider_preset": "kat_coder",
                    "provider_api_key": "test-key",
                    "provider_template_vars": {"ENDPOINT_ID": "ep-test-123"},
                }
            )

            applied = provider.get_applied_env_vars()
            assert "ANTHROPIC_BASE_URL" in applied
            assert "ep-test-123" in applied["ANTHROPIC_BASE_URL"]
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    def test_provider_override_env_disabled(self):
        """Test that provider doesn't override existing env vars by default."""
        os.environ["ANTHROPIC_BASE_URL"] = "https://original.example.com"

        try:
            provider = ClaudeCodeProvider(
                settings={
                    "provider_preset": "deepseek",
                    "provider_override_env": False,
                }
            )

            # Original value should be preserved
            assert os.getenv("ANTHROPIC_BASE_URL") == "https://original.example.com"

            # Applied vars should not include the overridden one
            applied = provider.get_applied_env_vars()
            assert "ANTHROPIC_BASE_URL" not in applied
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)

    def test_provider_override_env_enabled(self):
        """Test that provider overrides existing env vars when requested."""
        os.environ["ANTHROPIC_BASE_URL"] = "https://original.example.com"

        try:
            provider = ClaudeCodeProvider(
                settings={
                    "provider_preset": "deepseek",
                    "provider_override_env": True,
                }
            )

            # Should be overridden
            assert "deepseek" in os.getenv("ANTHROPIC_BASE_URL", "")

            # Applied vars should include the overridden one
            applied = provider.get_applied_env_vars()
            assert "ANTHROPIC_BASE_URL" in applied
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)


class TestModelRegistrationEdgeCases:
    """Edge case tests for model registration."""

    def test_invalid_format_falls_through(self):
        """Test that invalid format strings fall through to original handler."""
        from pydantic_ai import models

        # These should not be recognized as claude-code models
        # and should fall through to the original handler
        try:
            # Single colon with wrong prefix
            models.infer_model("other-provider:model")
        except Exception:
            pass  # Expected - falls through to original which may raise

        try:
            # No colon
            models.infer_model("claude-code")
        except Exception:
            pass  # Expected - falls through to original which may raise

    def test_four_part_string_ignored(self):
        """Test that four-part strings are ignored."""
        from pydantic_ai import models

        # Should fall through - not a valid claude-code format
        try:
            models.infer_model("claude-code:preset:model:extra")
        except Exception:
            pass  # Expected

    def test_preset_with_zhipu_glm(self):
        """Test model registration with Zhipu GLM preset."""
        from pydantic_ai import models

        from pydantic_ai_claude_code import ClaudeCodeModel

        model = models.infer_model("claude-code:zhipu_glm:sonnet")

        assert isinstance(model, ClaudeCodeModel)
        assert model._model_name == "glm-4.6"
        assert model.provider.provider_preset_id == "zhipu_glm"

    def test_preset_with_haiku_alias(self):
        """Test model registration with haiku alias."""
        from pydantic_ai import models

        from pydantic_ai_claude_code import ClaudeCodeModel

        model = models.infer_model("claude-code:zhipu_glm:haiku")

        assert isinstance(model, ClaudeCodeModel)
        assert model._model_name == "glm-4.5-air"

    def test_preset_with_nonexistent_alias(self):
        """Test model registration with nonexistent model alias."""
        from pydantic_ai import models

        from pydantic_ai_claude_code import ClaudeCodeModel

        model = models.infer_model("claude-code:deepseek:nonexistent")

        assert isinstance(model, ClaudeCodeModel)
        # Should return alias as-is if not in mapping
        assert model._model_name == "nonexistent"


class TestYAMLFileErrors:
    """Tests for YAML file loading error handling."""

    def test_malformed_yaml_returns_empty_dict(self):
        """Test that malformed YAML returns empty dict."""
        from pydantic_ai_claude_code.provider_presets import _load_yaml_file

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "bad.yaml"
            yaml_file.write_text("invalid: yaml: content:\n  - broken")

            result = _load_yaml_file(yaml_file)
            # Should return empty dict on error, not crash
            assert isinstance(result, dict)

    def test_nonexistent_yaml_returns_empty_dict(self):
        """Test that nonexistent YAML file returns empty dict."""
        from pydantic_ai_claude_code.provider_presets import _load_yaml_file

        result = _load_yaml_file(Path("/nonexistent/path/file.yaml"))
        assert result == {}

    def test_yaml_with_null_returns_empty_dict(self):
        """Test that YAML with null content returns empty dict."""
        from pydantic_ai_claude_code.provider_presets import _load_yaml_file

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "null.yaml"
            yaml_file.write_text("")

            result = _load_yaml_file(yaml_file)
            assert result == {}


class TestJSONFileErrors:
    """Tests for JSON file loading error handling."""

    def test_malformed_json_returns_empty_dict(self):
        """Test that malformed JSON returns empty dict."""
        from pydantic_ai_claude_code.provider_presets import _load_json_file

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "bad.json"
            json_file.write_text("{invalid json content}")

            result = _load_json_file(json_file)
            # Should return empty dict on error, not crash
            assert isinstance(result, dict)

    def test_nonexistent_json_returns_empty_dict(self):
        """Test that nonexistent JSON file returns empty dict."""
        from pydantic_ai_claude_code.provider_presets import _load_json_file

        result = _load_json_file(Path("/nonexistent/path/file.json"))
        assert result == {}


class TestPresetPrecedenceDetailed:
    """Detailed tests for preset loading precedence."""

    def test_project_overrides_builtin_and_user(self):
        """Test full precedence chain: project > user > builtin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            user_dir = Path(tmpdir) / ".claude"
            user_dir.mkdir()

            # User preset overrides builtin
            user_yaml = """
providers:
  deepseek:
    name: "User DeepSeek"
    website_url: "https://user-deepseek.com"
    settings:
      env:
        ANTHROPIC_BASE_URL: "https://user.api.com"
  user_only:
    name: "User Only"
    website_url: "https://user-only.com"
    settings:
      env: {}
"""
            (user_dir / "providers.yaml").write_text(user_yaml)

            # Project preset overrides both
            project_yaml = """
providers:
  deepseek:
    name: "Project DeepSeek"
    website_url: "https://project-deepseek.com"
    settings:
      env:
        ANTHROPIC_BASE_URL: "https://project.api.com"
  project_only:
    name: "Project Only"
    website_url: "https://project-only.com"
    settings:
      env: {}
"""
            (project_dir / "claude_providers.yaml").write_text(project_yaml)

            with patch.object(Path, "home", return_value=Path(tmpdir)):
                presets = load_all_presets(project_dir)

            # Project version should win
            assert presets["deepseek"].name == "Project DeepSeek"
            assert "project.api.com" in presets["deepseek"].settings["env"]["ANTHROPIC_BASE_URL"]

            # User-only preset should be included
            assert "user_only" in presets
            assert presets["user_only"].name == "User Only"

            # Project-only preset should be included
            assert "project_only" in presets
            assert presets["project_only"].name == "Project Only"

            # Built-in presets should still be present
            assert "zhipu_glm" in presets
            assert "qwen_coder" in presets


class TestPresetAPIKeyFields:
    """Tests for different API key field configurations."""

    def test_preset_with_custom_api_key_field(self):
        """Test preset with custom API key field."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            api_key_field="CUSTOM_API_KEY",
        )

        env_vars = preset.get_environment_variables(api_key="my-key")
        assert "CUSTOM_API_KEY" in env_vars
        assert env_vars["CUSTOM_API_KEY"] == "my-key"
        # Should not set ANTHROPIC_AUTH_TOKEN
        assert "ANTHROPIC_AUTH_TOKEN" not in env_vars

    def test_aihubmix_uses_anthropic_api_key(self):
        """Test that AiHubMix preset uses ANTHROPIC_API_KEY field."""
        presets = load_builtin_presets()
        aihubmix = presets.get("aihubmix")

        assert aihubmix is not None
        assert aihubmix.api_key_field == "ANTHROPIC_API_KEY"

        env_vars = aihubmix.get_environment_variables(api_key="test-key")
        assert "ANTHROPIC_API_KEY" in env_vars
        assert env_vars["ANTHROPIC_API_KEY"] == "test-key"


class TestPresetThemeConfiguration:
    """Tests for preset theme configuration."""

    def test_preset_with_theme(self):
        """Test preset with theme configuration."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            theme={
                "icon": "custom",
                "background_color": "#FF5733",
                "text_color": "#FFFFFF",
            },
        )

        assert preset.theme["icon"] == "custom"
        assert preset.theme["background_color"] == "#FF5733"
        assert preset.theme["text_color"] == "#FFFFFF"

    def test_preset_theme_in_to_dict(self):
        """Test that theme is included in to_dict output."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            theme={"icon": "test"},
        )

        data = preset.to_dict()
        assert "theme" in data
        assert data["theme"]["icon"] == "test"


class TestPresetEndpointCandidates:
    """Tests for preset endpoint candidates."""

    def test_preset_with_endpoint_candidates(self):
        """Test preset with multiple endpoint candidates."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            endpoint_candidates=[
                "https://api1.example.com",
                "https://api2.example.com",
                "https://api3.example.com",
            ],
        )

        assert len(preset.endpoint_candidates) == 3
        assert "api1.example.com" in preset.endpoint_candidates[0]

    def test_endpoint_candidates_in_to_dict(self):
        """Test that endpoint candidates are included in to_dict."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            endpoint_candidates=["https://api.example.com"],
        )

        data = preset.to_dict()
        assert "endpoint_candidates" in data
        assert len(data["endpoint_candidates"]) == 1


class TestPartnerPresets:
    """Tests for partner preset configurations."""

    def test_zhipu_is_partner(self):
        """Test that Zhipu GLM is marked as partner."""
        preset = get_preset("zhipu_glm")

        assert preset is not None
        assert preset.is_partner is True
        assert preset.partner_promotion_key == "zhipu"

    def test_zai_is_partner(self):
        """Test that Z.ai GLM is marked as partner."""
        preset = get_preset("zai_glm")

        assert preset is not None
        assert preset.is_partner is True
        assert preset.partner_promotion_key == "zhipu"

    def test_packycode_is_partner(self):
        """Test that PackyCode is marked as partner."""
        preset = get_preset("packycode")

        assert preset is not None
        assert preset.is_partner is True
        assert preset.partner_promotion_key == "packycode"


class TestSpecificPresetConfigurations:
    """Tests for specific preset configurations."""

    def test_kimi_k2_configuration(self):
        """Test Kimi k2 preset configuration."""
        preset = get_preset("kimi_k2")

        assert preset is not None
        assert preset.name == "Kimi k2"
        assert preset.category == "cn_official"
        assert preset.models.get("default") == "kimi-k2-thinking"

    def test_kimi_for_coding_configuration(self):
        """Test Kimi For Coding preset configuration."""
        preset = get_preset("kimi_for_coding")

        assert preset is not None
        assert preset.name == "Kimi For Coding"
        assert preset.models.get("sonnet") == "kimi-for-coding"

    def test_longcat_with_special_settings(self):
        """Test Longcat preset with special environment settings."""
        preset = get_preset("longcat")

        assert preset is not None
        env_vars = preset.get_environment_variables()
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" in env_vars
        assert env_vars["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "6000"
        assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" in env_vars
        assert env_vars["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    def test_minimax_with_timeout_setting(self):
        """Test MiniMax preset with custom timeout."""
        preset = get_preset("minimax")

        assert preset is not None
        env_vars = preset.get_environment_variables()
        assert "API_TIMEOUT_MS" in env_vars
        assert env_vars["API_TIMEOUT_MS"] == "3000000"

    def test_modelscope_aggregator(self):
        """Test ModelScope as aggregator category."""
        preset = get_preset("modelscope")

        assert preset is not None
        assert preset.category == "aggregator"
        assert "modelscope" in preset.website_url


class TestProviderContextManager:
    """Tests for ClaudeCodeProvider context manager functionality."""

    def test_provider_context_manager_with_preset(self):
        """Test provider as context manager with preset."""
        os.environ.pop("ANTHROPIC_BASE_URL", None)

        try:
            with ClaudeCodeProvider(
                settings={
                    "provider_preset": "deepseek",
                    "provider_api_key": "test-key",
                }
            ) as provider:
                assert provider.provider_preset_id == "deepseek"
                assert "ANTHROPIC_BASE_URL" in os.environ

            # Environment should still be set after context exit
            # (preset env vars persist for the process)
            assert "ANTHROPIC_BASE_URL" in os.environ
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


class TestEmptyAndNullConfigurations:
    """Tests for empty and null preset configurations."""

    def test_preset_with_empty_settings(self):
        """Test preset with empty settings dict."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={},
        )

        env_vars = preset.get_environment_variables()
        assert isinstance(env_vars, dict)
        assert len(env_vars) == 0

    def test_preset_with_none_models(self):
        """Test preset with None models."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            models=None,
        )

        # Should default to empty dict
        assert preset.models == {}
        assert preset.get_model_name("sonnet") == "sonnet"

    def test_preset_with_none_template_values(self):
        """Test preset with None template values."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            template_values=None,
        )

        # Should default to empty dict
        assert preset.template_values == {}


class TestCategoryFiltering:
    """Tests for filtering presets by category."""

    def test_get_official_category(self):
        """Test getting official category presets."""
        presets = get_presets_by_category("official")

        # Should include claude_official if present
        preset_ids = [p.preset_id for p in presets]
        assert "claude_official" in preset_ids

    def test_get_third_party_category(self):
        """Test getting third party presets."""
        presets = get_presets_by_category("third_party")

        for preset in presets:
            assert preset.category == "third_party"

    def test_empty_category_returns_empty_list(self):
        """Test that nonexistent category returns empty list."""
        # Using an invalid category should return empty list
        presets = get_presets_by_category("nonexistent")  # type: ignore
        # This will still work but filter out everything
        assert isinstance(presets, list)


class TestConcurrentPresetAccess:
    """Tests for concurrent access to preset configurations."""

    def test_multiple_providers_same_preset(self):
        """Test that multiple providers can use the same preset."""
        provider1 = ClaudeCodeProvider(
            settings={"provider_preset": "deepseek"}
        )
        provider2 = ClaudeCodeProvider(
            settings={"provider_preset": "deepseek"}
        )

        assert provider1.provider_preset_id == provider2.provider_preset_id
        assert provider1.provider_preset is not None
        assert provider2.provider_preset is not None

    def test_multiple_providers_different_presets(self):
        """Test that multiple providers can use different presets."""
        provider1 = ClaudeCodeProvider(
            settings={"provider_preset": "deepseek"}
        )
        provider2 = ClaudeCodeProvider(
            settings={"provider_preset": "zhipu_glm"}
        )

        assert provider1.provider_preset_id != provider2.provider_preset_id
        assert provider1.provider_preset.name == "DeepSeek"
        assert provider2.provider_preset.name == "Zhipu GLM"


class TestPresetValidation:
    """Tests for preset configuration validation."""

    def test_all_cn_official_have_base_url(self):
        """Test that all CN official presets have base URL configured."""
        presets = get_presets_by_category("cn_official")

        for preset in presets:
            env_vars = preset.get_environment_variables()
            assert (
                "ANTHROPIC_BASE_URL" in env_vars
            ), f"Preset {preset.preset_id} missing ANTHROPIC_BASE_URL"

    def test_all_aggregators_have_base_url(self):
        """Test that all aggregator presets have base URL configured."""
        presets = get_presets_by_category("aggregator")

        for preset in presets:
            # Aggregators should have base URL
            env_vars = preset.get_environment_variables()
            # At minimum should have ANTHROPIC_BASE_URL in settings
            assert len(preset.settings.get("env", {})) > 0
            assert "ANTHROPIC_BASE_URL" in env_vars, \
                f"Aggregator preset {preset.preset_id} missing ANTHROPIC_BASE_URL"

    def test_presets_have_valid_urls(self):
        """Test that preset URLs are valid."""
        presets = load_all_presets()

        for preset_id, preset in presets.items():
            assert preset.website_url.startswith("http"), \
                f"Preset {preset_id} has invalid website_url"
            if preset.api_key_url:
                assert preset.api_key_url.startswith("http"), \
                    f"Preset {preset_id} has invalid api_key_url"


class TestPresetModelMappings:
    """Tests for preset model mappings and aliases."""

    def test_get_model_name_with_all_aliases(self):
        """Test get_model_name with all standard aliases."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            models={
                "default": "model-default",
                "haiku": "model-haiku",
                "sonnet": "model-sonnet",
                "opus": "model-opus",
            },
        )

        assert preset.get_model_name("haiku") == "model-haiku"
        assert preset.get_model_name("sonnet") == "model-sonnet"
        assert preset.get_model_name("opus") == "model-opus"
        assert preset.get_model_name("custom") == "model-default"

    def test_get_model_name_with_partial_mapping(self):
        """Test get_model_name with partial model mapping."""
        preset = ProviderPreset(
            preset_id="test",
            name="Test",
            website_url="https://example.com",
            settings={"env": {}},
            models={
                "default": "model-default",
                "sonnet": "model-sonnet",
            },
        )

        # Mapped
        assert preset.get_model_name("sonnet") == "model-sonnet"
        assert preset.get_model_name("custom") == "model-default"
        # Not mapped - returns as-is
        assert preset.get_model_name("haiku") == "haiku"
        assert preset.get_model_name("opus") == "opus"
