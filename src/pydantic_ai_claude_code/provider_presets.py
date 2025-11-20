"""Provider preset management for different AI providers.

This module loads and manages provider presets from YAML configuration files,
supporting both built-in presets and user/project-level overrides.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Type definitions
ProviderCategory = Literal["official", "cn_official", "aggregator", "third_party"]


class ProviderPreset:
    """Represents a provider preset configuration."""

    def __init__(
        self,
        preset_id: str,
        name: str,
        website_url: str,
        settings: dict[str, Any],
        *,
        api_key_url: str | None = None,
        is_official: bool = False,
        is_partner: bool = False,
        partner_promotion_key: str | None = None,
        category: ProviderCategory = "third_party",
        api_key_field: str = "ANTHROPIC_AUTH_TOKEN",
        models: dict[str, str] | None = None,
        template_values: dict[str, dict[str, str]] | None = None,
        endpoint_candidates: list[str] | None = None,
        theme: dict[str, str] | None = None,
    ):
        """Initialize a provider preset.

        Args:
            preset_id: Unique identifier for the preset (e.g., "deepseek")
            name: Display name for the provider
            website_url: URL to the provider's website
            settings: Settings configuration including env vars
            api_key_url: URL to get API key (if different from website)
            is_official: Whether this is an official Anthropic preset
            is_partner: Whether this is a partner provider
            partner_promotion_key: i18n key for partner promotion
            category: Provider category
            api_key_field: Name of the API key environment variable
            models: Model name mappings (default, haiku, sonnet, opus)
            template_values: Template variable configurations
            endpoint_candidates: Alternative endpoint URLs
            theme: Visual theme configuration
        """
        self.preset_id = preset_id
        self.name = name
        self.website_url = website_url
        self.api_key_url = api_key_url
        self.settings = settings
        self.is_official = is_official
        self.is_partner = is_partner
        self.partner_promotion_key = partner_promotion_key
        self.category = category
        self.api_key_field = api_key_field
        self.models = models or {}
        self.template_values = template_values or {}
        self.endpoint_candidates = endpoint_candidates or []
        self.theme = theme or {}

    def get_model_name(self, model_alias: str) -> str:
        """Get the actual model name for a given alias.

        Args:
            model_alias: Model alias (e.g., "sonnet", "haiku", "opus", or "custom")

        Returns:
            Actual model name to use, or the alias itself if no mapping exists
        """
        if model_alias == "custom":
            return self.models.get("default", model_alias)
        return self.models.get(model_alias, model_alias)

    def get_environment_variables(
        self,
        api_key: str | None = None,
        template_vars: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Get environment variables for this provider.

        Args:
            api_key: API key to set (uses appropriate field name)
            template_vars: Template variable values to substitute

        Returns:
            Dictionary of environment variables to set
        """
        env_vars = {}
        template_vars = template_vars or {}

        # Get env vars from settings
        if "env" in self.settings:
            for key, value in self.settings["env"].items():
                if isinstance(value, str):
                    # Substitute template variables
                    env_vars[key] = self._substitute_templates(value, template_vars)
                else:
                    env_vars[key] = str(value)

        # Set API key if provided
        if api_key:
            env_vars[self.api_key_field] = api_key

        return env_vars

    def _substitute_templates(
        self, value: str, template_vars: dict[str, str]
    ) -> str:
        """Substitute template variables in a string.

        Args:
            value: String with ${VAR} placeholders
            template_vars: Variable name to value mapping

        Returns:
            String with variables substituted
        """
        # Pattern matches ${VAR_NAME}
        pattern = r"\$\{([^}]+)\}"

        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in template_vars:
                return template_vars[var_name]
            # Check environment as fallback
            env_value = os.getenv(var_name)
            if env_value:
                return env_value
            # Return original if not found
            logger.warning(
                "Template variable %s not found for provider %s",
                var_name,
                self.preset_id,
            )
            return match.group(0)

        return re.sub(pattern, replace, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert preset to dictionary representation."""
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "website_url": self.website_url,
            "api_key_url": self.api_key_url,
            "settings": self.settings,
            "is_official": self.is_official,
            "is_partner": self.is_partner,
            "partner_promotion_key": self.partner_promotion_key,
            "category": self.category,
            "api_key_field": self.api_key_field,
            "models": self.models,
            "template_values": self.template_values,
            "endpoint_candidates": self.endpoint_candidates,
            "theme": self.theme,
        }


def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents.

    Args:
        file_path: Path to the YAML file

    Returns:
        Parsed YAML content as a dictionary
    """
    try:
        import yaml
    except ImportError:
        logger.error(
            "PyYAML is required for provider presets. Install with: pip install pyyaml"
        )
        raise ImportError(
            "PyYAML is required for provider presets. "
            "Install with: pip install pyyaml"
        )

    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load YAML file %s: %s", file_path, e)
        return {}


def _load_json_file(file_path: Path) -> dict[str, Any]:
    """Load a JSON file and return its contents.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON content as a dictionary
    """
    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load JSON file %s: %s", file_path, e)
        return {}


def _parse_preset_dict(preset_id: str, data: dict[str, Any]) -> ProviderPreset:
    """Parse a preset dictionary into a ProviderPreset object.

    Args:
        preset_id: Unique identifier for the preset
        data: Raw preset data from YAML/JSON

    Returns:
        Parsed ProviderPreset object
    """
    return ProviderPreset(
        preset_id=preset_id,
        name=data.get("name", preset_id),
        website_url=data.get("website_url", ""),
        api_key_url=data.get("api_key_url"),
        settings=data.get("settings", {}),
        is_official=data.get("is_official", False),
        is_partner=data.get("is_partner", False),
        partner_promotion_key=data.get("partner_promotion_key"),
        category=data.get("category", "third_party"),
        api_key_field=data.get("api_key_field", "ANTHROPIC_AUTH_TOKEN"),
        models=data.get("models", {}),
        template_values=data.get("template_values", {}),
        endpoint_candidates=data.get("endpoint_candidates", []),
        theme=data.get("theme", {}),
    )


def load_builtin_presets() -> dict[str, ProviderPreset]:
    """Load built-in provider presets from the package's providers.yaml.

    Returns:
        Dictionary mapping preset IDs to ProviderPreset objects
    """
    presets: dict[str, ProviderPreset] = {}

    # Load from package's providers.yaml
    package_dir = Path(__file__).parent
    builtin_file = package_dir / "providers.yaml"

    if builtin_file.exists():
        data = _load_yaml_file(builtin_file)
        providers_data = data.get("providers", {})

        for preset_id, preset_data in providers_data.items():
            presets[preset_id] = _parse_preset_dict(preset_id, preset_data)
            logger.debug("Loaded built-in preset: %s", preset_id)

    return presets


def load_user_presets() -> dict[str, ProviderPreset]:
    """Load user-level provider presets from ~/.claude/providers.yaml or providers.json.

    Returns:
        Dictionary mapping preset IDs to ProviderPreset objects
    """
    presets: dict[str, ProviderPreset] = {}
    user_dir = Path.home() / ".claude"

    # Try YAML first, then JSON
    yaml_file = user_dir / "providers.yaml"
    json_file = user_dir / "providers.json"

    if yaml_file.exists():
        data = _load_yaml_file(yaml_file)
        providers_data = data.get("providers", data)  # Support flat format too

        for preset_id, preset_data in providers_data.items():
            if isinstance(preset_data, dict):
                presets[preset_id] = _parse_preset_dict(preset_id, preset_data)
                logger.debug("Loaded user preset: %s", preset_id)
    elif json_file.exists():
        data = _load_json_file(json_file)
        providers_data = data.get("providers", data)

        for preset_id, preset_data in providers_data.items():
            if isinstance(preset_data, dict):
                presets[preset_id] = _parse_preset_dict(preset_id, preset_data)
                logger.debug("Loaded user preset: %s", preset_id)

    return presets


def load_project_presets(project_dir: Path | None = None) -> dict[str, ProviderPreset]:
    """Load project-level provider presets from the project directory.

    Args:
        project_dir: Project directory (defaults to current working directory)

    Returns:
        Dictionary mapping preset IDs to ProviderPreset objects
    """
    presets: dict[str, ProviderPreset] = {}
    project_dir = project_dir or Path.cwd()

    # Try YAML first, then JSON
    yaml_file = project_dir / "claude_providers.yaml"
    json_file = project_dir / "claude_providers.json"

    if yaml_file.exists():
        data = _load_yaml_file(yaml_file)
        providers_data = data.get("providers", data)

        for preset_id, preset_data in providers_data.items():
            if isinstance(preset_data, dict):
                presets[preset_id] = _parse_preset_dict(preset_id, preset_data)
                logger.debug("Loaded project preset: %s", preset_id)
    elif json_file.exists():
        data = _load_json_file(json_file)
        providers_data = data.get("providers", data)

        for preset_id, preset_data in providers_data.items():
            if isinstance(preset_data, dict):
                presets[preset_id] = _parse_preset_dict(preset_id, preset_data)
                logger.debug("Loaded project preset: %s", preset_id)

    return presets


def load_all_presets(
    project_dir: Path | None = None,
) -> dict[str, ProviderPreset]:
    """Load all provider presets with proper precedence.

    Priority (highest to lowest):
    1. Project presets (./claude_providers.yaml or .json)
    2. User presets (~/.claude/providers.yaml or .json)
    3. Built-in presets

    Args:
        project_dir: Project directory for project-level presets

    Returns:
        Dictionary mapping preset IDs to ProviderPreset objects
    """
    presets: dict[str, ProviderPreset] = {}

    # Load in reverse priority order (so higher priority overrides)
    builtin = load_builtin_presets()
    presets.update(builtin)
    logger.debug("Loaded %d built-in presets", len(builtin))

    user = load_user_presets()
    presets.update(user)
    logger.debug("Loaded %d user presets", len(user))

    project = load_project_presets(project_dir)
    presets.update(project)
    logger.debug("Loaded %d project presets", len(project))

    return presets


def get_preset(
    preset_id: str, project_dir: Path | None = None
) -> ProviderPreset | None:
    """Get a specific provider preset by ID.

    Args:
        preset_id: The preset identifier (e.g., "deepseek")
        project_dir: Project directory for project-level presets

    Returns:
        ProviderPreset if found, None otherwise
    """
    presets = load_all_presets(project_dir)
    return presets.get(preset_id)


def list_presets(project_dir: Path | None = None) -> list[str]:
    """List all available preset IDs.

    Args:
        project_dir: Project directory for project-level presets

    Returns:
        List of preset IDs
    """
    presets = load_all_presets(project_dir)
    return sorted(presets.keys())


def get_presets_by_category(
    category: ProviderCategory, project_dir: Path | None = None
) -> list[ProviderPreset]:
    """Get all presets in a specific category.

    Args:
        category: Category to filter by
        project_dir: Project directory for project-level presets

    Returns:
        List of presets in the category
    """
    presets = load_all_presets(project_dir)
    return [p for p in presets.values() if p.category == category]


def apply_provider_environment(
    preset: ProviderPreset,
    api_key: str | None = None,
    template_vars: dict[str, str] | None = None,
    override_existing: bool = False,
) -> dict[str, str]:
    """Apply provider environment variables.

    Args:
        preset: Provider preset to apply
        api_key: API key to set
        template_vars: Template variable values
        override_existing: Whether to override existing env vars

    Returns:
        Dictionary of environment variables that were set
    """
    env_vars = preset.get_environment_variables(api_key, template_vars)
    applied: dict[str, str] = {}

    for key, value in env_vars.items():
        if override_existing or not os.getenv(key):
            os.environ[key] = value
            applied[key] = value
            logger.debug("Set environment variable: %s", key)
        else:
            logger.debug(
                "Skipping existing environment variable: %s (use override_existing=True to override)",
                key,
            )

    return applied
