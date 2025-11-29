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
        """
        Create a ProviderPreset representing a provider's metadata and runtime configuration.
        
        Parameters:
            preset_id (str): Unique identifier for the preset (e.g., "deepseek").
            name (str): Human-readable display name for the provider.
            website_url (str): Provider website URL.
            settings (dict[str, Any]): Configuration dictionary (commonly includes an "env" mapping of environment variable names to values or templates).
            api_key_url (str | None): URL where users can obtain an API key, if different from the website.
            is_official (bool): Whether this preset is an official built-in provider.
            is_partner (bool): Whether this preset represents a partner provider.
            partner_promotion_key (str | None): i18n key used to display partner promotion text.
            category (ProviderCategory): Logical category for the provider (defaults to "third_party").
            api_key_field (str): Environment variable name used for the provider API key (default "ANTHROPIC_AUTH_TOKEN").
            models (dict[str, str] | None): Optional mapping of model aliases (e.g., "default", "haiku") to actual model names.
            template_values (dict[str, dict[str, str]] | None): Optional per-context template variable mappings used for substituting placeholders in settings.
            endpoint_candidates (list[str] | None): Optional list of alternative endpoint URLs to try.
            theme (dict[str, str] | None): Optional visual/theme metadata for UI presentation.
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
        """
        Return the concrete model name for a given alias.
        
        Parameters:
            model_alias (str): Model alias (e.g., "sonnet", "haiku", "opus", or "custom"). If "custom", the preset's "default" model mapping is used when present.
        
        Returns:
            str: The mapped model name for the alias, or the alias itself if no mapping exists.
        """
        if model_alias == "custom":
            return self.models.get("default", model_alias)
        return self.models.get(model_alias, model_alias)

    def get_environment_variables(
        self,
        api_key: str | None = None,
        template_vars: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Builds the environment variables required by this provider preset.
        
        Parameters:
            api_key (str | None): If provided, sets the preset's configured API key field to this value.
            template_vars (dict[str, str] | None): Values used to substitute `${VAR}` placeholders in string environment values; missing placeholders fall back to existing OS environment variables.
        
        Returns:
            dict[str, str]: Mapping of environment variable names to their stringified values.
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
        """
        Replace `${VAR}` placeholders in `value` with provided template values or environment variables.
        
        Placeholders of the form `${NAME}` are resolved using `template_vars[NAME]` when present; if not found there, the process environment is consulted. If a placeholder cannot be resolved, it is left unchanged and a warning is logged referencing this preset's ID.
        
        Parameters:
            value (str): String containing `${VAR}` placeholders.
            template_vars (dict[str, str]): Mapping of placeholder names to replacement values.
        
        Returns:
            str: The input string with resolved substitutions; unresolved placeholders remain as `${NAME}`.
        """
        # Pattern matches ${VAR_NAME}
        pattern = r"\$\{([^}]+)\}"

        def replace(match: re.Match) -> str:
            """
            Return the substitution for a regex match representing a `${VAR}` placeholder.
            
            Parameters:
                match (re.Match): A regex match where `group(1)` is the placeholder variable name to resolve.
            
            Returns:
                str: The replacement value from `template_vars` if present; otherwise the environment variable with that name if set; if neither is found, the original matched placeholder.
            """
            var_name = match.group(1)
            if var_name in template_vars:
                return template_vars[var_name]
            # Check environment as fallback (empty string is valid)
            env_value = os.getenv(var_name)
            if env_value is not None:
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
        """
        Serialize the ProviderPreset into a plain dictionary.
        
        Returns:
            dict: Mapping with keys `preset_id`, `name`, `website_url`, `api_key_url`, `settings`, `is_official`, `is_partner`, `partner_promotion_key`, `category`, `api_key_field`, `models`, `template_values`, `endpoint_candidates`, and `theme` representing the preset's data.
        """
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
    """
    Load and parse a YAML file into a dictionary.

    Parses the YAML at file_path and returns the resulting mapping. If the file does not exist or parsing fails, an empty dictionary is returned.

    Returns:
        Parsed YAML content as a dict; an empty dict if the file is missing or parsing fails.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        logger.exception(
            "PyYAML is required for provider presets. Install with: pip install pyyaml"
        )
        raise ImportError(
            "PyYAML is required for provider presets. "
            "Install with: pip install pyyaml"
        ) from exc

    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.exception("Failed to load YAML file %s: %s", file_path, e)
        return {}


def _load_json_file(file_path: Path) -> dict[str, Any]:
    """
    Load and parse a JSON file into a dictionary.

    Parameters:
        file_path (Path): Path to the JSON file to read.

    Returns:
        dict[str, Any]: Parsed JSON content as a dictionary. Returns an empty dictionary if the file does not exist or if reading/parsing fails.
    """
    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.exception("Failed to load JSON file %s: %s", file_path, e)
        return {}


def _parse_preset_dict(preset_id: str, data: dict[str, Any]) -> ProviderPreset:
    """
    Convert a raw preset mapping into a ProviderPreset instance.
    
    Parameters:
        preset_id: Identifier to assign to the resulting preset; used as the default name when `data["name"]` is absent.
        data: Raw preset mapping (typically parsed from YAML/JSON). Recognized keys and their defaults:
            - name: display name (defaults to `preset_id`)
            - website_url: provider website (defaults to "")
            - api_key_url: URL describing how to obtain an API key
            - settings: provider settings including `env` (defaults to {})
            - is_official: whether the preset is official (defaults to False)
            - is_partner: whether the preset is a partner (defaults to False)
            - partner_promotion_key: optional partner promotion identifier
            - category: provider category (defaults to "third_party")
            - api_key_field: environment variable name for the API key (defaults to "ANTHROPIC_AUTH_TOKEN")
            - models: model alias-to-name mapping (defaults to {})
            - template_values: per-model template values (defaults to {})
            - endpoint_candidates: list of endpoint URLs (defaults to [])
            - theme: optional theme metadata (defaults to {})
    
    Returns:
        A ProviderPreset populated from the provided mapping.
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


def _load_presets_from_dir(
    dir_path: Path,
    yaml_name: str,
    json_name: str,
    log_prefix: str,
) -> dict[str, ProviderPreset]:
    """
    Load provider presets from YAML or JSON files in a directory.

    Tries YAML first, then JSON. Supports both 'providers' key and flat format.

    Parameters:
        dir_path: Directory to load presets from.
        yaml_name: Name of YAML file to try (e.g., 'providers.yaml').
        json_name: Name of JSON file to try (e.g., 'providers.json').
        log_prefix: Prefix for log messages (e.g., 'user', 'project').

    Returns:
        Mapping of preset IDs to ProviderPreset instances.
    """
    presets: dict[str, ProviderPreset] = {}

    yaml_file = dir_path / yaml_name
    json_file = dir_path / json_name

    if yaml_file.exists():
        data = _load_yaml_file(yaml_file)
        providers_data = data.get("providers", data)  # Support flat format too

        for preset_id, preset_data in providers_data.items():
            if isinstance(preset_data, dict):
                presets[preset_id] = _parse_preset_dict(preset_id, preset_data)
                logger.debug("Loaded %s preset: %s", log_prefix, preset_id)
    elif json_file.exists():
        data = _load_json_file(json_file)
        providers_data = data.get("providers", data)

        for preset_id, preset_data in providers_data.items():
            if isinstance(preset_data, dict):
                presets[preset_id] = _parse_preset_dict(preset_id, preset_data)
                logger.debug("Loaded %s preset: %s", log_prefix, preset_id)

    return presets


def load_builtin_presets() -> dict[str, ProviderPreset]:
    """
    Load built-in provider presets from the package's providers.yaml.

    If the providers.yaml file is missing or cannot be parsed, returns an empty mapping.

    Returns:
        A mapping of preset IDs to ProviderPreset instances; empty if no built-in presets are available.
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
    """
    Load user-level provider presets from the user's ~/.claude directory.

    Checks for providers.yaml first, then providers.json. Each file may contain either a top-level
    "providers" mapping or a flat mapping of preset IDs to preset dictionaries; valid preset entries
    are converted to ProviderPreset instances.

    Returns:
        dict[str, ProviderPreset]: Mapping of preset IDs to loaded ProviderPreset objects.
    """
    user_dir = Path.home() / ".claude"
    return _load_presets_from_dir(
        user_dir, "providers.yaml", "providers.json", "user"
    )


def load_project_presets(project_dir: Path | None = None) -> dict[str, ProviderPreset]:
    """Load project-level provider presets from the project directory.

    Args:
        project_dir: Project directory (defaults to current working directory)

    Returns:
        Dictionary mapping preset IDs to ProviderPreset objects
    """
    project_dir = project_dir or Path.cwd()
    return _load_presets_from_dir(
        project_dir, "claude_providers.yaml", "claude_providers.json", "project"
    )


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
    """
    Retrieve the provider preset with the given identifier.
    
    Parameters:
        preset_id (str): The preset identifier (e.g., "deepseek").
        project_dir (Path | None): Optional project directory to include project-level presets (defaults to current working directory).
    
    Returns:
        ProviderPreset | None: The ProviderPreset for the given ID, or `None` if no matching preset exists.
    """
    presets = load_all_presets(project_dir)
    return presets.get(preset_id)


def list_presets(project_dir: Path | None = None) -> list[str]:
    """
    List all available provider preset IDs from built-in, user, and project scopes.
    
    Parameters:
        project_dir (Path | None): Path to the project directory to include project-level presets; if None, the current working directory is used.
    
    Returns:
        list[str]: A sorted list of preset ID strings.
    """
    presets = load_all_presets(project_dir)
    return sorted(presets.keys())


def get_presets_by_category(
    category: ProviderCategory, project_dir: Path | None = None
) -> list[ProviderPreset]:
    """
    Retrieve all provider presets matching the specified category.
    
    Parameters:
        category (ProviderCategory): Category to filter presets by.
        project_dir (Path | None): Project directory to include project-level presets; if None, uses the default lookup (current working directory).
    
    Returns:
        list[ProviderPreset]: List of ProviderPreset instances whose `category` equals the given category.
    """
    presets = load_all_presets(project_dir)
    return [p for p in presets.values() if p.category == category]


def compute_provider_environment(
    preset: ProviderPreset,
    api_key: str | None = None,
    template_vars: dict[str, str] | None = None,
    override_existing: bool = False,
) -> dict[str, str]:
    """
    Compute environment variables for a provider preset without modifying global state.

    This function determines which environment variables should be set based on the preset
    configuration and the override_existing flag, but does NOT modify os.environ.

    Parameters:
        preset (ProviderPreset): The provider preset whose environment settings will be computed.
        api_key (str | None): If provided, include the preset's API key field with this value.
        template_vars (dict[str, str] | None): Values used to substitute placeholders in environment values.
        override_existing (bool): If True, include variables even if they exist in os.environ; otherwise skip them.

    Returns:
        dict[str, str]: Mapping of environment variable names to values that would be applied.
    """
    env_vars = preset.get_environment_variables(api_key, template_vars)
    computed: dict[str, str] = {}

    for key, value in env_vars.items():
        if override_existing or key not in os.environ:
            computed[key] = value
            logger.debug("Computed environment variable: %s", key)
        else:
            logger.debug(
                "Skipping existing environment variable: %s (use override_existing=True to override)",
                key,
            )

    return computed


def apply_provider_environment(
    preset: ProviderPreset,
    api_key: str | None = None,
    template_vars: dict[str, str] | None = None,
    override_existing: bool = False,
) -> dict[str, str]:
    """
    Apply environment variables defined by a provider preset to the current process environment.

    WARNING: This function modifies global os.environ. Consider using compute_provider_environment()
    instead and passing the environment variables directly to subprocess calls.

    Parameters:
        preset (ProviderPreset): The provider preset whose environment settings will be applied.
        api_key (str | None): If provided, set the preset's API key field to this value.
        template_vars (dict[str, str] | None): Values used to substitute placeholders in environment values.
        override_existing (bool): If True, overwrite existing environment variables; otherwise preserve them.

    Returns:
        dict[str, str]: Mapping of environment variable names to values that were actually set.
    """
    env_vars = preset.get_environment_variables(api_key, template_vars)
    applied: dict[str, str] = {}

    for key, value in env_vars.items():
        if override_existing or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
            logger.debug("Set environment variable: %s", key)
        else:
            logger.debug(
                "Skipping existing environment variable: %s (use override_existing=True to override)",
                key,
            )

    return applied