import os
import argparse
import yaml
from collections.abc import Mapping
from config.manage_api_client import init_service, get_server_config, get_agent_models


# Add global configuration cache
_config_cache = None


def get_project_dir():
    """Get project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_config():
    """Load configuration file"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    default_config_path = get_project_dir() + "config.yaml"
    custom_config_path = get_project_dir() + "data/.config.yaml"

    # Load default configuration
    default_config = read_config(default_config_path)
    custom_config = read_config(custom_config_path)

    if custom_config.get("manager-api", {}).get("url"):
        config = get_config_from_api(custom_config)
    else:
        # Merge configurations
        config = merge_configs(default_config, custom_config)
    # Initialize directories
    ensure_directories(config)
    _config_cache = config
    return config


def get_config_from_api(config):
    """Get configuration from Java API"""
    # Initialize API client
    init_service(config)

    # Get server configuration
    config_data = get_server_config()
    if config_data is None:
        raise Exception("Failed to fetch server config from API")

    config_data["read_config_from_api"] = True
    config_data["manager-api"] = {
        "url": config["manager-api"].get("url", ""),
        "secret": config["manager-api"].get("secret", ""),
    }
    # Server configuration prioritizes local settings
    if config.get("server"):
        config_data["server"] = {
            "ip": config["server"].get("ip", ""),
            "port": config["server"].get("port", ""),
            "http_port": config["server"].get("http_port", ""),
            "vision_explain": config["server"].get("vision_explain", ""),
            "auth_key": config["server"].get("auth_key", ""),
        }
    return config_data


def get_private_config_from_api(config, device_id, client_id):
    """Get private configuration from Java API"""
    return get_agent_models(device_id, client_id, config["selected_module"])


def ensure_directories(config):
    """Ensure all configuration paths exist"""
    dirs_to_create = set()
    project_dir = get_project_dir()  # Get project root directory
    # Log file directory
    log_dir = config.get("log", {}).get("log_dir", "tmp")
    dirs_to_create.add(os.path.join(project_dir, log_dir))

    # ASR/TTS module output directories
    for module in ["ASR", "TTS"]:
        if config.get(module) is None:
            continue
        for provider in config.get(module, {}).values():
            output_dir = provider.get("output_dir", "")
            if output_dir:
                dirs_to_create.add(output_dir)

    # Create model directories based on selected_module
    selected_modules = config.get("selected_module", {})
    for module_type in ["ASR", "LLM", "TTS"]:
        selected_provider = selected_modules.get(module_type)
        if not selected_provider:
            continue
        if config.get(module) is None:
            continue
        if config.get(selected_provider) is None:
            continue
        provider_config = config.get(module_type, {}).get(selected_provider, {})
        output_dir = provider_config.get("output_dir")
        if output_dir:
            full_model_dir = os.path.join(project_dir, output_dir)
            dirs_to_create.add(full_model_dir)

    # Create directories (preserve original data directory creation)
    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except PermissionError:
            print(f"Warning: Cannot create directory {dir_path}, please check write permissions")


def merge_configs(default_config, custom_config):
    """
    Recursively merge configurations, custom_config has higher priority

    Args:
        default_config: Default configuration
        custom_config: User custom configuration

    Returns:
        Merged configuration
    """
    if not isinstance(default_config, Mapping) or not isinstance(
        custom_config, Mapping
    ):
        return custom_config

    merged = dict(default_config)

    for key, value in custom_config.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
