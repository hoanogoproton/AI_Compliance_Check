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


def test_load_config_classifier_section_valid(tmp_path):
    cfg = {
        "behaviors": [{"name": "hand_to_head", "enabled": True, "params": {}}],
        "classifier": {"behaviors": {"hand_to_head": {
            "model_path": "./models/hand_to_head/classifier.pkl",
            "scaler_path": "./models/hand_to_head/scaler.pkl",
            "metadata_path": "./models/hand_to_head/metadata.json",
            "threshold": 0.82,
        }}},
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    loaded = load_config(str(p))
    assert "classifier" in loaded
    assert loaded["classifier"]["behaviors"]["hand_to_head"]["threshold"] == 0.82


def test_load_config_classifier_threshold_out_of_range(tmp_path):
    cfg = {
        "behaviors": [{"name": "hand_to_head", "enabled": True, "params": {}}],
        "classifier": {"behaviors": {"hand_to_head": {"threshold": 1.5}}},
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    try:
        load_config(str(p))
        assert False, "Should have raised ValueError for out-of-range threshold"
    except ValueError:
        pass


def test_load_config_classifier_behaviors_not_mapping(tmp_path):
    cfg = {
        "behaviors": [{"name": "hand_to_head", "enabled": True, "params": {}}],
        "classifier": {"behaviors": ["not", "a", "mapping"]},
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    try:
        load_config(str(p))
        assert False, "Should have raised ValueError for non-mapping behaviors"
    except ValueError:
        pass
