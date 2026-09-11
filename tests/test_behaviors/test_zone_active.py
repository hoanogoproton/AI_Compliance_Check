"""Test logic OR của ``_compute_zone_active`` khi nhiều behavior dùng chung zone.

Trong config thật, ``hand_snatch_object`` và ``hand_shake_object`` cùng xem
zone ``Hand_Shake_OBJ``: trạng thái đỏ của zone phải là OR giữa các behavior
(hand_shake dựa ``detected`` trong frame_data, hand_snatch dựa
``current_triggered_zones``), thay vì behavior đứng sau ghi đè behavior trước.
"""
import detection.pipeline as pipeline_mod
from detection.behaviors.hand_shake_object import HandShakeObjectBehavior
from detection.behaviors.hand_snatch_object import HandSnatchObjectBehavior
from detection.zones.zone_definition import Zone


def _make_zone():
    return Zone(name="SharedZone", label="Shared",
                points=[[200, 100], [300, 100], [300, 180], [200, 180]])


def _make_behaviors():
    zone = _make_zone()
    shake = HandShakeObjectBehavior({}, zones=[zone], fps=30)
    snatch = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[zone], fps=30)
    return [shake, snatch]


def _frame_data(detected_map):
    """frame_data giả đúng cấu trúc pipeline Phase 2 dựng cho một người."""
    return {
        1: {
            "bbox": (0, 0, 400, 300),
            "keypoints": None,
            "behaviors": {name: {"detected": det} for name, det in detected_map.items()},
        }
    }


def test_zone_active_or_of_shared_behaviors():
    behaviors = _make_behaviors()
    # hand_shake detected, hand_snatch chưa có detection → zone vẫn đỏ
    zone_active = pipeline_mod._compute_zone_active(
        behaviors, _frame_data({"hand_shake_object": True, "hand_snatch_object": False}), 0
    )
    assert zone_active["SharedZone"] == "active"


def test_zone_inactive_when_no_behavior_detected():
    behaviors = _make_behaviors()
    zone_active = pipeline_mod._compute_zone_active(
        behaviors, _frame_data({"hand_shake_object": False, "hand_snatch_object": False}), 0
    )
    assert zone_active["SharedZone"] == "inactive"


def test_zone_active_from_snatch_triggered_zones():
    behaviors = _make_behaviors()
    # hand_snatch bắn (current_triggered_zones), hand_shake không detected:
    # OR từ phía hand_snatch cũng phải làm zone đỏ
    behaviors[1]._frame_alert_zones.add("SharedZone")
    zone_active = pipeline_mod._compute_zone_active(
        behaviors, _frame_data({"hand_shake_object": False, "hand_snatch_object": False}), 0
    )
    assert zone_active["SharedZone"] == "active"


def test_zone_active_when_both_behaviors_active():
    behaviors = _make_behaviors()
    behaviors[1]._frame_alert_zones.add("SharedZone")
    zone_active = pipeline_mod._compute_zone_active(
        behaviors, _frame_data({"hand_shake_object": True, "hand_snatch_object": False}), 0
    )
    assert zone_active["SharedZone"] == "active"