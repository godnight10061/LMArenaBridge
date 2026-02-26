"""
Configuration management for LMArenaBridge.
Handles loading, saving, and managing configuration.
"""

import json
import os
import sys
from typing import Optional, Dict, Any
from collections import defaultdict

from . import constants


# Global state
_current_config_file: str = constants.CONFIG_FILE
_current_token_index: int = 0


def get_config_file() -> str:
    """Get the current config file path."""
    return _current_config_file


def set_config_file(path: str) -> None:
    """Set the config file path (useful for tests)."""
    global _current_config_file, _current_token_index
    if _current_config_file != path:
        _current_config_file = path
        _current_token_index = 0


def read_raw_config(path: str) -> Optional[dict]:
    """Read and parse config from disk, returning None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            return value
        print(f"Warning: config file at '{path}' did not contain a JSON object.", file=sys.stderr)
        return None
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse config from disk (invalid JSON): {e}", file=sys.stderr)
        return None
    except OSError as e:
        print(f"Warning: could not read config from disk: {e}", file=sys.stderr)
        return None


def get_config() -> dict:
    """
    Load configuration from file with defaults.
    Returns a dictionary with all configuration values.
    """
    global _current_token_index
    
    # Reset token index if config file changed
    if _current_token_index != 0:
        global_state = _get_global_state()
        if global_state.get("_last_config_file") != _current_config_file:
            _current_token_index = 0
    
    config = read_raw_config(_current_config_file) or {}

    # Ensure default keys exist
    _apply_config_defaults(config)
    
    return config


def apply_config_defaults(config: dict) -> None:
    """Apply default values to config dictionary in-place."""
    _apply_config_defaults(config)


def _apply_config_defaults(config: dict) -> None:
    """Apply default values to config dictionary."""
    config.setdefault("password", "admin")
    config.setdefault("auth_token", "")
    config.setdefault("auth_tokens", [])
    config.setdefault("cf_clearance", "")
    config.setdefault("api_keys", [])
    config.setdefault("usage_stats", {})
    config.setdefault("prune_invalid_tokens", False)
    config.setdefault("persist_arena_auth_cookie", False)
    config.setdefault("camoufox_proxy_window_mode", constants.DEFAULT_CAMOUFOX_PROXY_WINDOW_MODE)
    config.setdefault("camoufox_fetch_window_mode", constants.DEFAULT_CAMOUFOX_FETCH_WINDOW_MODE)
    config.setdefault("chrome_fetch_window_mode", constants.DEFAULT_CHROME_FETCH_WINDOW_MODE)
    
    # Normalize api_keys
    if isinstance(config.get("api_keys"), list):
        normalized_keys = []
        for key_entry in config["api_keys"]:
            if isinstance(key_entry, dict):
                if "key" not in key_entry:
                    continue
                if "name" not in key_entry:
                    key_entry["name"] = "Unnamed Key"
                if "created" not in key_entry:
                    key_entry["created"] = 1704236400  # Default timestamp
                if "rpm" not in key_entry:
                    key_entry["rpm"] = constants.DEFAULT_RATE_LIMIT_RPM
                normalized_keys.append(key_entry)
        config["api_keys"] = normalized_keys


def save_config(
    config: dict,
    *,
    preserve_auth_tokens: bool = True,
    preserve_api_keys: bool = True,
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        preserve_auth_tokens: If True, don't overwrite auth tokens from disk
        preserve_api_keys: If True, don't overwrite API keys from disk
    """
    try:
        if preserve_auth_tokens or preserve_api_keys:
            on_disk = read_raw_config(_current_config_file)

            if isinstance(on_disk, dict):
                def _preserve_list_from_disk(config_dict: dict, on_disk_config: dict, key: str) -> None:
                    if key in on_disk_config:
                        value_on_disk = on_disk_config.get(key)
                        if isinstance(value_on_disk, list):
                            config_dict[key] = list(value_on_disk)

                if preserve_auth_tokens:
                    _preserve_list_from_disk(config, on_disk, "auth_tokens")
                    if "auth_token" in on_disk:
                        config["auth_token"] = str(on_disk.get("auth_token") or "")
                if preserve_api_keys:
                    _preserve_list_from_disk(config, on_disk, "api_keys")

        # usage_stats will be set by the caller
        
        tmp_path = f"{_current_config_file}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        os.replace(tmp_path, _current_config_file)
    except (OSError, TypeError):
        raise


# Global state storage (for cross-module state)
_global_state: Dict[str, Any] = {}


def _get_global_state() -> Dict[str, Any]:
    """Get the global state dictionary."""
    return _global_state


def set_global_state(key: str, value: Any) -> None:
    """Set a global state value."""
    _global_state[key] = value


def get_global_state(key: str, default: Any = None) -> Any:
    """Get a global state value."""
    return _global_state.get(key, default)


# === Model management ===

def get_models() -> list:
    """Load models from file."""
    try:
        with open(constants.MODELS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_models(models: list) -> None:
    """Save models to file."""
    try:
        tmp_path = f"{constants.MODELS_FILE}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(models, f, indent=2)
        os.replace(tmp_path, constants.MODELS_FILE)
    except Exception as e:
        print(f"Error saving models: {e}")


# === Default config for startup ===

def get_default_config() -> dict:
    """Get default configuration values."""
    return {
        "password": "admin",
        "auth_token": "",
        "auth_tokens": [],
        "cf_clearance": "",
        "api_keys": [
            {
                "name": "Default Key",
                "key": "",  # Will be generated
                "rpm": constants.DEFAULT_RATE_LIMIT_RPM,
                "created": 0,
            }
        ],
        "usage_stats": {},
        "prune_invalid_tokens": False,
        "persist_arena_auth_cookie": False,
        "camoufox_proxy_window_mode": constants.DEFAULT_CAMOUFOX_PROXY_WINDOW_MODE,
        "camoufox_fetch_window_mode": constants.DEFAULT_CAMOUFOX_FETCH_WINDOW_MODE,
        "chrome_fetch_window_mode": constants.DEFAULT_CHROME_FETCH_WINDOW_MODE,
    }
