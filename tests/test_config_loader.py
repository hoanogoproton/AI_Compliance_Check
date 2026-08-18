import yaml

from detection.config_loader import load_config


def test_load_config_basic(tmp_path):
    cfg = {
        "behaviors": [
            {"name": "hand_to_head", "enabled": True, "params": {}}
        ]
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    loaded = load_config(str(p))
    assert len(loaded["behaviors"]) == 1
    assert loaded["behaviors"][0]["name"] == "hand_to_head"


def test_load_config_missing_file():
    try:
        load_config("nonexistent.yaml")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_config_no_behaviors(tmp_path):
    cfg = {"model": {"path": "test.pt"}}
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    try:
        load_config(str(p))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_load_config_invalid_behaviors_type(tmp_path):
    cfg = {"behaviors": "not a list"}
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    try:
        load_config(str(p))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
