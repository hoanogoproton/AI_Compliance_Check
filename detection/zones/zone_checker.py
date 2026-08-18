from pathlib import Path

import yaml

from detection.zones.zone_definition import Zone


def load_zones(config_data: dict) -> dict[str, Zone]:
    zones = {}
    raw = config_data.get("zones", {})
    if not raw:
        return zones
    for name, data in raw.items():
        zones[name] = Zone.from_dict(name, data)
    return zones


def save_zones(zones: dict[str, Zone], path: str):
    data = {name: zone.to_dict() for name, zone in zones.items()}
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            config = yaml.safe_load(f) or {}
        config["zones"] = data
    else:
        config = {"zones": data}
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
