from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    _validate_config(config)
    return config


def _validate_config(config: dict):
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping (dict)")
    if "behaviors" not in config:
        raise ValueError("Config must contain 'behaviors' list")
    if not isinstance(config["behaviors"], list):
        raise ValueError("'behaviors' must be a list")
    for bcfg in config["behaviors"]:
        if "name" not in bcfg:
            raise ValueError(f"Each behavior must have a 'name' field, got: {bcfg}")
        if not isinstance(bcfg.get("enabled", True), bool):
            raise ValueError(f"Behavior '{bcfg['name']}': 'enabled' must be bool")
