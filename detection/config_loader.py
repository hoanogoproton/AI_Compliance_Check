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
    crop = config.get("crop")
    if crop is not None:
        if not isinstance(crop, list) or len(crop) != 4:
            raise ValueError("'crop' must be a list of 4 integers [x, y, w, h] or null")
        x, y, w, h = crop
        if not all(isinstance(v, int) for v in crop):
            raise ValueError("'crop' values must be integers")
        if x < 0 or y < 0:
            raise ValueError("'crop' x and y must be >= 0")
        if w <= 0 or h <= 0:
            raise ValueError("'crop' w and h must be > 0")
    if "behaviors" not in config:
        raise ValueError("Config must contain 'behaviors' list")
    if not isinstance(config["behaviors"], list):
        raise ValueError("'behaviors' must be a list")
    for bcfg in config["behaviors"]:
        if "name" not in bcfg:
            raise ValueError(f"Each behavior must have a 'name' field, got: {bcfg}")
        if not isinstance(bcfg.get("enabled", True), bool):
            raise ValueError(f"Behavior '{bcfg['name']}': 'enabled' must be bool")
    _validate_classifier(config.get("classifier"))


def _validate_classifier(classifier: dict | None):
    """Validate the optional `classifier.behaviors` section.

    Structure only is validated here (keys + types). Missing model files are
    tolerated so configs can be loaded before any model has been trained; the
    pipeline skips a behavior whose model files are absent.
    """
    if classifier is None:
        return
    if not isinstance(classifier, dict):
        raise ValueError("'classifier' must be a mapping")
    behaviors = classifier.get("behaviors")
    if behaviors is None:
        return
    if not isinstance(behaviors, dict):
        raise ValueError("'classifier.behaviors' must be a mapping of behavior_name -> config")
    for bname, bcfg in behaviors.items():
        if not isinstance(bcfg, dict):
            raise ValueError(f"classifier.behaviors.{bname} must be a mapping")
        for key in ("model_path", "scaler_path", "metadata_path"):
            val = bcfg.get(key)
            if val is not None and not isinstance(val, str):
                raise ValueError(f"classifier.behaviors.{bname}.{key} must be a string path")
        threshold = bcfg.get("threshold")
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError(f"classifier.behaviors.{bname}.threshold must be a number")
        if threshold is not None and not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"classifier.behaviors.{bname}.threshold must be in [0, 1]")
