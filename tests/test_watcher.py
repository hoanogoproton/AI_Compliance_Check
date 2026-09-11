import json
from pathlib import Path

import numpy as np
import pytest

import watch_folders as wf


def _write_csv(path, rows, header="folder,config"):
    lines = [header] + list(rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_pipeline_ok(calls):
    def fn(video_path, output_dir, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        (out / f"{stem}_metadata.json").write_text(
            json.dumps({"events": [{"id": 1}, {"id": 2}, {"id": 3}]}),
            encoding="utf-8",
        )
        calls.append(video_path)

    return fn


def _fake_pipeline_progress(calls, total=1000):
    """Fake pipeline that reports per-frame progress and logs one line."""

    def fn(video_path, output_dir, progress_callback=None, log_callback=None, **kwargs):
        calls.append(video_path)
        if log_callback:
            log_callback("fake pipeline log line")
        for frame in range(1, total + 1):
            if progress_callback:
                progress_callback(frame, total)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        (out / f"{stem}_metadata.json").write_text(
            json.dumps({"events": [{"id": 1}, {"id": 2}]}),
            encoding="utf-8",
        )

    return fn


def _make_runner(tmp_path, pipeline_fn, **runner_kwargs):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("behaviors: []\n", encoding="utf-8")
    folder = tmp_path / "drop"
    folder.mkdir()
    csv_file = tmp_path / "watch.csv"
    _write_csv(csv_file, [f"{folder},{cfg}"])
    runner = wf.WatchRunner(
        csv_path=csv_file,
        poll_interval=0.01,
        output_dir=tmp_path / "out",
        journal_path=tmp_path / "state.json",
        pipeline_fn=pipeline_fn,
        **runner_kwargs,
    )
    return runner, folder, cfg


# --------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------

def test_load_watch_csv_basic(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.touch()
    folder = tmp_path / "drop"
    folder.mkdir()
    _write_csv(tmp_path / "w.csv", [f"{folder},{cfg}"])
    entries, errors = wf.load_watch_csv(tmp_path / "w.csv")
    assert errors == []
    assert len(entries) == 1
    assert entries[0]["folder"] == folder
    assert entries[0]["config"] == cfg
    assert entries[0]["output_dir"] is None


def test_load_watch_csv_wrong_columns(tmp_path):
    _write_csv(tmp_path / "w.csv", ["a,b"], header="video,config")
    with pytest.raises(ValueError):
        wf.load_watch_csv(tmp_path / "w.csv")


def test_load_watch_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        wf.load_watch_csv(tmp_path / "nope.csv")


def test_load_watch_csv_relative_paths_and_output_dir(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.touch()
    folder = tmp_path / "drop"
    folder.mkdir()
    _write_csv(
        tmp_path / "w.csv",
        ["drop,cfg.yaml,./myout", "drop2,cfg.yaml"],
        header="folder,config,output_dir",
    )
    entries, errors = wf.load_watch_csv(tmp_path / "w.csv")
    assert errors == []
    assert len(entries) == 2
    entry = entries[0]
    assert entry["folder"] == tmp_path / "drop"
    assert entry["config"] == tmp_path / "cfg.yaml"
    assert entry["output_dir"] == tmp_path / "myout"
    # row without output_dir falls back to the default (None)
    assert entries[1]["output_dir"] is None


def test_load_watch_csv_blank_and_duplicate_rows(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.touch()
    folder = tmp_path / "drop"
    folder.mkdir()
    _write_csv(
        tmp_path / "w.csv",
        [f"{folder},{cfg}", f"{folder},{cfg}", ",,", f"{folder},"],
    )
    entries, errors = wf.load_watch_csv(tmp_path / "w.csv")
    assert len(entries) == 1
    assert len(errors) == 3


# --------------------------------------------------------------------------
# File stability
# --------------------------------------------------------------------------

def test_is_file_stable(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"0" * 100)
    sizes = {}
    assert wf.is_file_stable(f, sizes) is False  # first sighting
    assert wf.is_file_stable(f, sizes) is True   # unchanged since last poll
    f.write_bytes(b"1" * 120)
    assert wf.is_file_stable(f, sizes) is False  # size changed (still copying)
    assert wf.is_file_stable(f, sizes) is True
    assert wf.is_file_stable(tmp_path / "gone.mp4", sizes) is False


# --------------------------------------------------------------------------
# Journal
# --------------------------------------------------------------------------

def test_journal_round_trip(tmp_path):
    jp = tmp_path / "state.json"
    journal = wf.load_journal(jp)
    assert journal == {"processed": [], "errors": {}, "ignored_initial": {}}
    journal["processed"].append({"video": "x.mp4"})
    wf.save_journal(jp, journal)
    assert wf.load_journal(jp)["processed"][0]["video"] == "x.mp4"


def test_journal_corrupt_file_recovers(tmp_path):
    jp = tmp_path / "state.json"
    jp.write_text("{not json", encoding="utf-8")
    assert wf.load_journal(jp)["processed"] == []


def test_count_events(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "v_metadata.json").write_text(
        json.dumps({"events": [1, 2]}), encoding="utf-8"
    )
    assert wf.count_events(out, "v") == 2
    assert wf.count_events(out, "missing") is None


# --------------------------------------------------------------------------
# WatchRunner behavior
# --------------------------------------------------------------------------

def test_existing_videos_are_ignored(tmp_path):
    calls = []
    runner, folder, cfg = _make_runner(tmp_path, _fake_pipeline_ok(calls))
    (folder / "old.mp4").write_bytes(b"x" * 10)
    runner.run_cycle()
    assert calls == []
    assert runner.journal["ignored_initial"][str(folder)] == ["old.mp4"]
    runner.run_cycle()
    assert calls == []  # still ignored, never processed


def test_new_video_processed_then_deleted(tmp_path):
    calls = []
    runner, folder, cfg = _make_runner(tmp_path, _fake_pipeline_ok(calls))
    runner.run_cycle()  # startup scan — folder empty
    (folder / "new.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()  # first sighting: size recorded, not yet stable
    assert calls == []
    runner.run_cycle()  # stable now -> processed
    assert len(calls) == 1
    assert not (folder / "new.mp4").exists()  # original deleted
    processed = runner.journal["processed"][0]
    assert processed["events"] == 3
    assert processed["deleted"] is True
    assert (tmp_path / "out" / "new" / "new_metadata.json").exists()
    runner.run_cycle()
    assert len(calls) == 1  # no reprocessing


@pytest.mark.parametrize("send_email", [True, False])
def test_results_email_toggle(tmp_path, monkeypatch, send_email):
    """``send_email=False`` skips the results email; the default sends it."""
    sent = []
    monkeypatch.setattr(
        wf.email_notifier,
        "send_results_email_async",
        lambda subject, body, log=None: sent.append(subject),
    )
    runner, folder, cfg = _make_runner(
        tmp_path, _fake_pipeline_ok([]), send_email=send_email
    )
    assert runner.send_email is send_email
    runner.run_cycle()               # folder first seen, empty
    (folder / "v.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()               # sighting (not stable yet)
    runner.run_cycle()               # stable -> processed, 3 events in metadata
    if send_email:
        assert len(sent) == 1
        assert sent[0] == "Kết quả xử lý - v.mp4"
    else:
        assert sent == []


def test_error_keeps_video_then_gives_up(tmp_path):
    calls = []

    def failing(video_path, **kwargs):
        calls.append(video_path)
        raise RuntimeError("boom")

    runner, folder, cfg = _make_runner(tmp_path, failing)
    runner.run_cycle()
    (folder / "bad.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()  # sighting
    runner.run_cycle()  # attempt 1 -> kept, will retry
    assert len(calls) == 1
    assert (folder / "bad.mp4").exists()
    runner.run_cycle()  # attempt 2 -> give up
    assert len(calls) == 2
    assert (folder / "bad.mp4").exists()
    err = runner.journal["errors"][str(folder / "bad.mp4")]
    assert err["attempts"] == 2
    assert err["gave_up"] is True
    runner.run_cycle()  # no more retries
    assert len(calls) == 2


def test_hidden_and_non_video_files_skipped(tmp_path):
    calls = []
    runner, folder, cfg = _make_runner(tmp_path, _fake_pipeline_ok(calls))
    (folder / ".hidden.mp4").write_bytes(b"x" * 5)
    (folder / "notes.txt").write_bytes(b"x" * 5)
    (folder / "vid.mp4.part").write_bytes(b"x" * 5)
    runner.run_cycle()
    runner.run_cycle()
    assert calls == []


def test_same_stem_from_two_folders_no_collision(tmp_path):
    calls = []
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("behaviors: []\n", encoding="utf-8")
    f1 = tmp_path / "cam_one"
    f1.mkdir()
    f2 = tmp_path / "cam_two"
    f2.mkdir()
    csv_file = tmp_path / "watch.csv"
    _write_csv(csv_file, [f"{f1},{cfg}", f"{f2},{cfg}"])
    runner = wf.WatchRunner(
        csv_path=csv_file,
        poll_interval=0.01,
        output_dir=tmp_path / "out",
        journal_path=tmp_path / "state.json",
        pipeline_fn=_fake_pipeline_ok(calls),
    )
    runner.run_cycle()
    (f1 / "clip.mp4").write_bytes(b"a" * 5)
    (f2 / "clip.mp4").write_bytes(b"b" * 5)
    runner.run_cycle()  # sighting
    runner.run_cycle()  # process both
    assert len(calls) == 2
    assert (tmp_path / "out" / "clip").exists()
    assert (tmp_path / "out" / f"{f2.name}_clip").exists()


def test_csv_reload_picks_up_new_folder(tmp_path):
    calls = []
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("behaviors: []\n", encoding="utf-8")
    f1 = tmp_path / "drop1"
    f1.mkdir()
    f2 = tmp_path / "drop2"
    f2.mkdir()
    (f2 / "v.mp4").write_bytes(b"z" * 10)
    csv_file = tmp_path / "watch.csv"
    _write_csv(csv_file, [f"{f1},{cfg}"])
    runner = wf.WatchRunner(
        csv_path=csv_file,
        poll_interval=0.01,
        output_dir=tmp_path / "out",
        journal_path=tmp_path / "state.json",
        pipeline_fn=_fake_pipeline_ok(calls),
    )
    runner.run_cycle()  # only drop1 known; drop2 not in the CSV yet
    assert calls == []
    _write_csv(csv_file, [f"{f1},{cfg}", f"{f2},{cfg}"])
    runner.run_cycle()  # drop2 first seen -> its existing video is ignored
    assert calls == []
    (f2 / "v2.mp4").write_bytes(b"z" * 10)
    runner.run_cycle()  # sighting
    runner.run_cycle()  # processed
    assert len(calls) == 1
    assert calls[0].endswith("v2.mp4")


# --------------------------------------------------------------------------
# Event callback (GUI realtime feed)
# --------------------------------------------------------------------------

def test_events_for_successful_video(tmp_path):
    calls = []
    events = []
    runner, folder, cfg = _make_runner(
        tmp_path, _fake_pipeline_progress(calls), event_callback=events.append
    )
    runner.run_cycle()                     # startup scan: folder first seen, empty
    (folder / "new.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()                     # first sighting: not stable yet
    runner.run_cycle()                     # stable -> processed
    runner._print_summary()                # emits the final `stopped` event

    types = [e["type"] for e in events]
    assert "folder_seen" in types
    assert any(e["type"] == "log" and "PROCESSING" in e["message"] for e in events)
    assert types.index("video_new") < types.index("video_start")
    assert types.index("video_start") < types.index("video_done")
    assert types.index("video_done") < types.index("stopped")

    folder_seen = events[types.index("folder_seen")]
    assert folder_seen["ignored"] == 0

    new_ev = events[types.index("video_new")]
    assert new_ev["video_name"] == "new.mp4"
    assert new_ev["folder"] == str(folder)
    assert new_ev["config"] == str(cfg)
    assert new_ev["key"] == wf.canonical(folder / "new.mp4")

    # progress events: throttled to ~1% steps, last one reports completion
    start = types.index("video_start")
    done = types.index("video_done")
    progress = [e for e in events[start:done] if e["type"] == "video_progress"]
    assert progress, "expected progress events between start and done"
    assert len(progress) <= 105
    assert progress[-1]["frame"] == progress[-1]["total"] == 1000
    assert all(e["key"] == new_ev["key"] for e in progress)

    # the pipeline received the log callback and its message was mirrored
    assert any(
        e["type"] == "log" and "fake pipeline log line" in e["message"]
        for e in events
    )

    done_ev = events[done]
    assert done_ev["events"] == 2
    assert done_ev["deleted"] is True

    stopped = events[-1]
    if stopped["type"] == "log":
        # The async results-email thread may log after the summary.
        stopped = [e for e in events if e["type"] == "stopped"][-1]
    assert stopped["type"] == "stopped"
    assert stopped["processed"] == 1
    assert stopped["deleted"] == 1
    assert stopped["failed"] == 0


def test_folder_seen_event_lists_ignored_videos(tmp_path):
    events = []
    runner, folder, cfg = _make_runner(
        tmp_path, _fake_pipeline_ok([]), event_callback=events.append
    )
    (folder / "old.mp4").write_bytes(b"x" * 10)
    runner.run_cycle()
    folder_seen = [e for e in events if e["type"] == "folder_seen"][0]
    assert folder_seen["ignored"] == 1
    assert folder_seen["videos"] == ["old.mp4"]


def test_startup_and_stopped_events(tmp_path):
    events = []
    runner, folder, cfg = _make_runner(
        tmp_path, _fake_pipeline_ok([]), event_callback=events.append, max_cycles=0
    )
    (folder / "old.mp4").write_bytes(b"x" * 10)
    runner.run()  # no poll cycles: just startup + summary
    types = [e["type"] for e in events]
    assert "startup" in types
    assert types[-1] == "stopped"
    startup = events[types.index("startup")]
    assert startup["csv_path"] == str(runner.csv_path)
    assert startup["poll_interval"] == runner.poll_interval > 0
    assert startup["visualize"] is True
    assert events[-1]["processed"] == 0
    assert events[-1]["deleted"] == 0
    assert events[-1]["failed"] == 0


def test_events_for_failing_video(tmp_path):
    events = []
    calls = []

    def failing(video_path, output_dir, **kwargs):
        calls.append(video_path)
        raise RuntimeError("boom")

    runner, folder, cfg = _make_runner(
        tmp_path, failing, event_callback=events.append
    )
    runner.run_cycle()                     # startup scan: folder empty
    (folder / "bad.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()                     # sighting
    runner.run_cycle()                     # attempt 1 -> kept, will retry
    runner.run_cycle()                     # attempt 2 -> gave up
    runner._print_summary()

    errors = [e for e in events if e["type"] == "video_error"]
    assert len(errors) == 2
    assert errors[0]["attempts"] == 1
    assert errors[0]["gave_up"] is False
    assert errors[0]["max_retries"] == 2
    assert errors[1]["attempts"] == 2
    assert errors[1]["gave_up"] is True
    assert "boom" in errors[1]["error"]
    assert all(e["video_name"] == "bad.mp4" for e in errors)
    assert all(e["key"] == wf.canonical(folder / "bad.mp4") for e in errors)

    stopped = events[-1]
    assert stopped["type"] == "stopped"
    assert stopped["processed"] == 0
    assert stopped["failed"] == 1
    assert len(calls) == 2  # retried once before giving up


def test_event_callback_none_keeps_cli_behavior(tmp_path):
    calls = []
    received_kwargs = []

    def fn(video_path, output_dir, **kwargs):
        calls.append(video_path)
        received_kwargs.append(kwargs)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        (out / f"{stem}_metadata.json").write_text(
            json.dumps({"events": []}), encoding="utf-8"
        )

    runner, folder, cfg = _make_runner(tmp_path, fn)
    assert runner.event_callback is None
    runner.run_cycle()
    (folder / "x.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()
    runner.run_cycle()
    assert len(calls) == 1
    # CLI path: no GUI callbacks are wired into run_pipeline (tqdm stays as-is)
    assert "progress_callback" not in received_kwargs[0]
    assert "log_callback" not in received_kwargs[0]
    assert runner.journal["processed"][0]["deleted"] is True
    assert "frame_callback" not in received_kwargs[0]


def test_frame_callback_wired_in_gui_mode(tmp_path):
    """GUI mode: the preview callback is passed through and bound to the key."""
    calls = []
    received = []

    def fn(video_path, output_dir, frame_callback=None, **kwargs):
        assert frame_callback is not None
        frame_callback(
            0, np.zeros((2, 2, 3), dtype=np.uint8), {"people": 1, "events": 0}
        )
        calls.append(video_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        (out / f"{stem}_metadata.json").write_text(
            json.dumps({"events": []}), encoding="utf-8"
        )

    runner, folder, cfg = _make_runner(
        tmp_path, fn, event_callback=lambda e: None,
        frame_callback=lambda *a: received.append(a),
    )
    runner.run_cycle()                     # folder first seen (empty)
    (folder / "v.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()                     # sighting (not stable yet)
    runner.run_cycle()                     # stable -> queued and processed
    assert len(calls) == 1
    assert len(received) == 1
    key, frame_idx, rgb, info = received[0]
    assert key == wf.canonical(folder / "v.mp4")
    assert frame_idx == 0
    assert rgb.shape == (2, 2, 3)
    assert info == {"people": 1, "events": 0}


def test_stop_aborts_current_video_and_reprocesses_next_start(tmp_path):
    """Stop must abort the video being processed immediately.

    The unfinished video stays on disk, is journalled as `interrupted`, and a
    fresh watcher sharing the same journal re-processes it instead of
    ignoring it like a pre-existing video.
    """
    import threading
    import time

    calls = []
    events = []
    entered = threading.Event()
    seen_abort = {}

    def aborting_fn(video_path, output_dir, abort_event=None, **kwargs):
        calls.append(video_path)
        seen_abort["event"] = abort_event
        entered.set()
        for _ in range(500):
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("aborted by stop request")
            time.sleep(0.005)

    def ok_fn(video_path, output_dir, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        (out / f"{stem}_metadata.json").write_text(
            json.dumps({"events": []}), encoding="utf-8"
        )

    runner, folder, cfg = _make_runner(
        tmp_path, aborting_fn, event_callback=events.append
    )
    runner.run_cycle()                     # folder first seen (empty)
    video = folder / "mid.mp4"
    video.write_bytes(b"v" * 50)
    runner.run_cycle()                     # sighting (not stable yet)

    worker = threading.Thread(target=runner.run_cycle)
    worker.start()
    assert entered.wait(5.0), "pipeline was never started"
    runner.stop()
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "watcher thread did not stop promptly"

    # Aborted mid-video: kept on disk, not counted as processed nor failed.
    assert video.exists()
    assert seen_abort["event"] is runner._stop
    stopped = [e for e in events if e["type"] == "video_stopped"]
    assert len(stopped) == 1
    assert stopped[0]["video_name"] == "mid.mp4"
    assert stopped[0]["key"] == wf.canonical(video)
    assert runner.stats["processed"] == 0
    assert runner.stats["failed"] == 0
    assert runner.journal["processed"] == []
    assert wf.canonical(video) in runner.journal.get("interrupted", {})

    # A fresh runner sharing the journal re-processes the interrupted video
    # (it is NOT ignored like other pre-existing videos) and deletes it.
    events2 = []
    runner2 = wf.WatchRunner(
        csv_path=runner.csv_path,
        poll_interval=0.01,
        output_dir=tmp_path / "out",
        journal_path=tmp_path / "state.json",
        pipeline_fn=ok_fn,
        event_callback=events2.append,
    )
    runner2.run_cycle()                    # first seen: interrupted NOT ignored
    runner2.run_cycle()                    # stable -> queued, processed, deleted
    assert not video.exists()
    assert any(e["type"] == "video_done" for e in events2)
    assert wf.canonical(video) not in runner2.journal.get("interrupted", {})
    assert not any(e["type"] == "video_error" for e in events2)