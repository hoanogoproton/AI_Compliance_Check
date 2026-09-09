"""Tests cho khung "giờ trôi qua" và vòng lặp tải theo CSV của get_video.py.

Chỉ kiểm tra logic thuần (tính khung giờ, đặt tên tệp, vòng lặp CSV, state
file) — không mở trình duyệt: fetch_one_camera được thay bằng hàm giả.
"""

from datetime import datetime
from pathlib import Path

import get_video as gv


# -- previous_hour_window ------------------------------------------------------

def test_previous_hour_window_mid_hour():
    begin, end = gv.previous_hour_window(datetime(2026, 9, 9, 14, 20))
    assert begin == datetime(2026, 9, 9, 13, 0)
    assert end == datetime(2026, 9, 9, 14, 0)


def test_previous_hour_window_exactly_on_the_hour():
    begin, end = gv.previous_hour_window(datetime(2026, 9, 9, 14, 0, 30))
    assert begin == datetime(2026, 9, 9, 13, 0)
    assert end == datetime(2026, 9, 9, 14, 0)


def test_previous_hour_window_crosses_midnight():
    begin, end = gv.previous_hour_window(datetime(2026, 9, 10, 0, 5))
    assert begin == datetime(2026, 9, 9, 23, 0)
    assert end == datetime(2026, 9, 10, 0, 0)


# -- save_download -------------------------------------------------------------

class _FakeDownload:
    def __init__(self, suggested: str):
        self.suggested_filename = suggested
        self.saved_to: Path | None = None

    def save_as(self, path) -> None:
        self.saved_to = Path(path)


def test_save_download_prefers_suggested_name(tmp_path):
    download = _FakeDownload("CA927-FB-RAI7-No3-20260909-131530.mp4")
    target = gv.save_download(download, tmp_path, "CA927-FB-RAI7-No3", 0)
    assert target == tmp_path / "CA927-FB-RAI7-No3-20260909-131530.mp4"
    assert download.saved_to == target


def test_save_download_avoids_overwrite(tmp_path):
    existing = tmp_path / "CA927-20260909-131530.mp4"
    existing.write_bytes(b"x")
    download = _FakeDownload("CA927-20260909-131530.mp4")
    target = gv.save_download(download, tmp_path, "CA927", 0)
    assert target == tmp_path / "CA927-20260909-131530_2.mp4"


def test_save_download_falls_back_to_indexed_name(tmp_path):
    download = _FakeDownload("")
    target = gv.save_download(download, tmp_path, "CA927-FB-RAI7-No3", 2)
    assert target == tmp_path / "CA927-FB-RAI7-No3_03.mp4"
    assert download.saved_to == target


# -- fetch_all_from_csv --------------------------------------------------------

def _write_csv(tmp_path: Path, cameras: list[str]) -> Path:
    rows = "".join(f"{tmp_path / cam},config/config_4.yaml\n" for cam in cameras)
    csv_path = tmp_path / "watch_folders.csv"
    csv_path.write_text("folder,config\n" + rows, encoding="utf-8")
    return csv_path


def test_fetch_all_visits_every_camera(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path, ["CA927-FB-RAI7-No3", "CA857-FB-RAI3-No1"])
    calls = []

    def fake_fetch(camera_name, download_dir, begin_dt, end_dt, **kwargs):
        calls.append((camera_name, Path(download_dir)))
        return [Path(download_dir) / "clip.mp4"]

    monkeypatch.setattr(gv, "fetch_one_camera", fake_fetch)

    state_path = tmp_path / "fetch_state.json"
    summary = gv.fetch_all_from_csv(
        csv_path,
        datetime(2026, 9, 9, 13, 0),
        datetime(2026, 9, 9, 14, 0),
        state_path=state_path,
        log=lambda _message: None,
    )

    assert [c[0] for c in calls] == ["CA927-FB-RAI7-No3", "CA857-FB-RAI3-No1"]
    assert calls[0][1] == tmp_path / "CA927-FB-RAI7-No3"
    assert calls[1][1] == tmp_path / "CA857-FB-RAI3-No1"
    assert summary["ok"] == 2
    assert summary["skipped"] == 0
    assert summary["failed"] == 0

    state = gv.load_fetch_state(state_path)
    assert set(state["cameras"]) == {"CA927-FB-RAI7-No3", "CA857-FB-RAI3-No1"}
    assert state["cameras"]["CA927-FB-RAI7-No3"] == "2026-09-09 14:00:00"


def test_fetch_all_skips_already_fetched_window(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path, ["CA927-FB-RAI7-No3"])
    calls = []

    def fake_fetch(camera_name, download_dir, begin_dt, end_dt, **kwargs):
        calls.append(camera_name)
        return []

    monkeypatch.setattr(gv, "fetch_one_camera", fake_fetch)

    state_path = tmp_path / "fetch_state.json"
    window = (datetime(2026, 9, 9, 13, 0), datetime(2026, 9, 9, 14, 0))
    gv.fetch_all_from_csv(csv_path, *window, state_path=state_path, log=lambda _m: None)
    summary = gv.fetch_all_from_csv(
        csv_path, *window, state_path=state_path, log=lambda _m: None
    )

    assert calls == ["CA927-FB-RAI7-No3"]  # lần 2 không tải lại
    assert summary["skipped"] == 1
    assert summary["ok"] == 0


def test_fetch_all_continues_after_camera_error(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path, ["CA927-FB-RAI7-No3", "CA857-FB-RAI3-No1"])

    def fake_fetch(camera_name, download_dir, begin_dt, end_dt, **kwargs):
        if camera_name == "CA927-FB-RAI7-No3":
            raise RuntimeError("không mở được trình duyệt")
        return []

    monkeypatch.setattr(gv, "fetch_one_camera", fake_fetch)

    summary = gv.fetch_all_from_csv(
        csv_path,
        datetime(2026, 9, 9, 13, 0),
        datetime(2026, 9, 9, 14, 0),
        state_path=tmp_path / "fetch_state.json",
        log=lambda _message: None,
    )

    assert summary["ok"] == 1
    assert summary["failed"] == 1
    assert summary["results"][0]["status"] == "error"
    assert summary["results"][1]["status"] == "ok"
    # Camera lỗi không được ghi vào state để lần sau thử lại.
    state = gv.load_fetch_state(tmp_path / "fetch_state.json")
    assert state["cameras"] == {"CA857-FB-RAI3-No1": "2026-09-09 14:00:00"}


def test_fetch_all_stops_between_cameras(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path, ["CA927-FB-RAI7-No3", "CA857-FB-RAI3-No1"])
    calls = []

    def fake_fetch(camera_name, download_dir, begin_dt, end_dt, **kwargs):
        calls.append(camera_name)
        return []

    monkeypatch.setattr(gv, "fetch_one_camera", fake_fetch)

    summary = gv.fetch_all_from_csv(
        csv_path,
        datetime(2026, 9, 9, 13, 0),
        datetime(2026, 9, 9, 14, 0),
        should_stop=lambda: len(calls) >= 1,
        state_path=tmp_path / "fetch_state.json",
        log=lambda _message: None,
    )

    assert calls == ["CA927-FB-RAI7-No3"]
    assert summary["ok"] == 1
    assert summary["total"] == 2