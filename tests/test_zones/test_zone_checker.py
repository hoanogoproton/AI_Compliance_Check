import pytest
import yaml

from detection.zones.zone_definition import Zone
from detection.zones.zone_checker import load_zones


def test_zone_contains_point_inside():
    zone = Zone(name="test", label="Test", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    assert zone.contains_point(50, 50)


def test_zone_contains_point_outside():
    zone = Zone(name="test", label="Test", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    assert not zone.contains_point(200, 200)


def test_zone_contains_bbox_center():
    zone = Zone(name="test", label="Test", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    assert zone.contains_bbox_center((40, 40, 60, 60))
    assert not zone.contains_bbox_center((200, 200, 220, 220))


def test_zone_intersects_bbox():
    zone = Zone(name="test", label="Test", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    assert zone.intersects_bbox((40, 40, 60, 60))
    assert zone.intersects_bbox((90, 40, 110, 60))
    assert not zone.intersects_bbox((110, 40, 130, 60))
    assert not zone.intersects_bbox((200, 200, 220, 220))


def test_zone_polygon_property():
    zone = Zone(name="test", label="Test", points=[[0, 0], [100, 0], [100, 100], [0, 100]])
    poly = zone.polygon
    assert poly.shape == (4, 2)


def test_zone_to_dict_roundtrip():
    zone = Zone(name="test", label="Test Zone", points=[[0, 0], [100, 0], [100, 100]])
    d = zone.to_dict()
    assert d["label"] == "Test Zone"
    assert d["points"] == [[0, 0], [100, 0], [100, 100]]
    restored = Zone.from_dict("test", d)
    assert restored.name == "test"
    assert restored.points == zone.points


def test_load_zones_from_config():
    config = {
        "zones": {
            "area1": {"label": "Area 1", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            "area2": {"label": "Area 2", "points": [[20, 20], [30, 20], [30, 30]]},
        }
    }
    zones = load_zones(config)
    assert len(zones) == 2
    assert zones["area1"].name == "area1"
    assert zones["area2"].label == "Area 2"


def test_load_zones_empty():
    assert load_zones({}) == {}
