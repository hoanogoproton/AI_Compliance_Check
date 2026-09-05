import json
from pathlib import Path

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


def _make_runner(tmp_path, pipeline_fn):
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