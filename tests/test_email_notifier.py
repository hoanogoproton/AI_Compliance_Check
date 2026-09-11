import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import email_notifier
import watch_folders as wf


EVENTS = [
    {
        "event_id": 1,
        "behavior": "vay tay",
        "start_time_sec": 1.5,
        "end_time_sec": 3.2,
    },
    {
        "event_id": 2,
        "behavior": "",
        "start_time_sec": 10.0,
        "end_time_sec": 12.4,
    },
]


class _FakeSocket:
    def __init__(self, response=b"OK", recv_error=None):
        self.response = response
        self.recv_error = recv_error
        self.sent = b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sendall(self, data):
        self.sent += data

    def recv(self, bufsize):
        if self.recv_error is not None:
            raise self.recv_error
        return self.response


def _write_csv(path, rows, header="folder,config"):
    lines = [header] + list(rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_pipeline_raw(content):
    def fn(video_path, output_dir, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        if content is not None:
            (out / f"{stem}_metadata.json").write_text(content, encoding="utf-8")

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


def _collect_sends(monkeypatch):
    sent = []
    monkeypatch.setattr(
        email_notifier,
        "send_results_email_async",
        lambda subject, body, log=None: sent.append((subject, body, log)),
    )
    return sent


# --------------------------------------------------------------------------
# send_email (socket protocol)
# --------------------------------------------------------------------------

def test_send_email_message_format(monkeypatch):
    fake = _FakeSocket(b"QUEUED")
    seen = {}

    def fake_create_connection(address, timeout=None):
        seen["address"] = address
        seen["timeout"] = timeout
        return fake

    monkeypatch.setattr(
        email_notifier,
        "socket",
        SimpleNamespace(create_connection=fake_create_connection),
    )
    message, response = email_notifier.send_email("sub ject", "line1\nline2")

    assert seen["address"] == (email_notifier.SERVER_IP, email_notifier.SERVER_PORT)
    assert seen["timeout"] == email_notifier.SOCKET_TIMEOUT
    assert message == f"[SendEmail_KTTT]{email_notifier.FACTORY}|sub ject|line1\nline2"
    assert fake.sent.decode("utf-8") == message
    assert response == "QUEUED"


def test_send_email_raises_on_connection_error(monkeypatch):
    def fake_create_connection(address, timeout=None):
        raise OSError("unreachable")

    monkeypatch.setattr(
        email_notifier,
        "socket",
        SimpleNamespace(create_connection=fake_create_connection),
    )
    with pytest.raises(OSError):
        email_notifier.send_email("s", "b")


# --------------------------------------------------------------------------
# build_results_html (professional single-line HTML body)
# --------------------------------------------------------------------------

def test_build_results_html_layout():
    events = [
        {"event_id": 1, "behavior": "vay tay", "start_time_sec": 1.5, "end_time_sec": 3.2},
        {"event_id": 2, "behavior": "", "start_time_sec": 10.0, "end_time_sec": 12.4},
    ]
    body = email_notifier.build_results_html(
        "cam.mp4", "D:/out", events, processed_at="01/01/2026 08:00:00"
    )

    assert body.startswith("<!DOCTYPE html>")
    assert "\n" not in body
    assert "cam.mp4" in body
    assert "D:/out" in body
    assert "01/01/2026 08:00:00" in body
    assert "vay tay" in body
    assert ">1.5<" in body
    assert ">3.2<" in body
    assert ">10<" in body
    assert ">12.4<" in body
    assert ">-<" in body
    assert ">2 sự kiện" in body
    # STT and duration columns are removed.
    assert "STT" not in body
    assert "Thời lượng" not in body
    assert ">1.7<" not in body
    assert ">2.4<" not in body


def test_build_results_html_escapes_and_sanitizes_pipes():
    events = [
        {"event_id": 1, "behavior": "<b>x|y</b>", "start_time_sec": 0, "end_time_sec": 1}
    ]
    body = email_notifier.build_results_html("a<b>.mp4", "D:/out", events)

    assert "|" not in body
    assert "&lt;b&gt;x/y&lt;/b&gt;" in body
    assert "a&lt;b&gt;.mp4" in body


def test_build_results_html_defaults_to_current_time():
    body = email_notifier.build_results_html("v.mp4", "D:/out", [])
    assert "Thời gian xử lý" in body
    assert "v.mp4" in body


def test_build_results_html_short_list_shows_every_row():
    events = [
        {"event_id": i, "behavior": f"be {i}",
         "start_time_sec": float(i), "end_time_sec": float(i) + 1}
        for i in range(1, 4)
    ]
    body = email_notifier.build_results_html("cam.mp4", "D:/out", events)
    assert "sự kiện khác" not in body
    for i in range(1, 4):
        assert f"<td>{i}</td>" in body  # event_id cell, one per row
        assert f"be {i}" in body


def test_build_results_html_keeps_request_under_service_limit():
    events = [
        {"event_id": f"EVT-{i:03d}", "behavior": "Khong deo mu bao hiem",
         "start_time_sec": float(i), "end_time_sec": float(i) + 2.5}
        for i in range(1, 201)
    ]
    video = "CA442-FB-PC_No2-20260904-081130.mp4"
    body = email_notifier.build_results_html(
        video, rf"D:\out\{Path(video).stem}", events
    )
    subject = f"Kết quả xử lý - {video}"
    total = (
        len(f"[SendEmail_KTTT]{email_notifier.FACTORY}|{subject}|".encode("utf-8"))
        + len(body.encode("utf-8"))
    )
    assert total <= email_notifier.MAX_MESSAGE_BYTES
    assert "sự kiện khác" in body
    assert "EVT-001" in body
    assert "EVT-200" not in body
    assert "\n" not in body
    assert "|" not in body


def test_build_results_html_folder_is_clickable_file_link():
    body = email_notifier.build_results_html(
        "cam.mp4", r"D:\out\CA442 results", EVENTS
    )

    assert (
        'Thư mục kết quả: <a href="file:///D:/out/CA442%20results">'
        "D:\\out\\CA442 results</a>" in body
    )
    assert "\n" not in body
    assert "|" not in body


def test_build_results_html_folder_posix_and_unc_paths():
    posix = email_notifier.build_results_html("v.mp4", "/data/out", [])
    assert '<a href="file:///data/out">/data/out</a>' in posix

    unc = email_notifier.build_results_html("v.mp4", r"\\srv\share\out", [])
    assert '<a href="file://srv/share/out">\\\\srv\\share\\out</a>' in unc


def test_build_results_html_folder_relative_path_stays_plain_text():
    body = email_notifier.build_results_html("v.mp4", "outputs/cam", [])

    assert "<a href" not in body
    assert "outputs/cam" in body


def test_build_results_html_folder_http_url_is_linked():
    body = email_notifier.build_results_html(
        "v.mp4", "http://172.17.108.169/results/cam", []
    )

    assert '<a href="http://172.17.108.169/results/cam">http://' in body
    assert "172.17.108.169/results/cam</a>" in body


def test_send_email_returns_empty_response_when_service_does_not_reply(
    monkeypatch,
):
    fake = _FakeSocket(recv_error=socket.timeout("timed out"))

    def fake_create_connection(address, timeout=None):
        return fake

    monkeypatch.setattr(
        email_notifier,
        "socket",
        SimpleNamespace(create_connection=fake_create_connection),
    )
    _message, response = email_notifier.send_email("s", "b")
    assert response == ""


def test_send_email_roundtrip_delivers_full_message(monkeypatch):
    # socketpair: deterministic two-way connection, no listener/accept race,
    # and the real factory server is never touched.
    server_side, client_side = socket.socketpair()
    received = {}

    def fake_create_connection(address, timeout=None):
        return client_side

    monkeypatch.setattr(
        email_notifier,
        "socket",
        SimpleNamespace(create_connection=fake_create_connection),
    )

    def _reader():
        server_side.settimeout(2)
        chunks = b""
        try:
            while not chunks.rstrip().endswith(b"</html>"):
                part = server_side.recv(65536)
                if not part:
                    break
                chunks += part
        except OSError:
            pass
        received["data"] = chunks
        server_side.sendall(b"QUEUED")

    thread = threading.Thread(target=_reader)
    thread.start()
    body = email_notifier.build_results_html(
        "roundtrip.mp4", r"D:\out\roundtrip", EVENTS * 50
    )
    subject = "Kết quả xử lý - roundtrip.mp4"
    message, response = email_notifier.send_email(subject, body)
    thread.join(5)
    server_side.close()
    client_side.close()

    assert response == "QUEUED"
    assert received["data"].decode("utf-8") == message
    assert len(received["data"]) <= email_notifier.MAX_MESSAGE_BYTES


# --------------------------------------------------------------------------
# send_results_email_async (fire-and-forget thread)
# --------------------------------------------------------------------------

def test_send_results_email_async_logs_success(monkeypatch):
    finished = threading.Event()
    logs = []

    def fake_send(subject, body):
        return ("MSG", "RESP-OK")

    def capture(msg):
        logs.append(msg)
        if len(logs) >= 2:
            finished.set()

    monkeypatch.setattr(email_notifier, "send_email", fake_send)
    email_notifier.send_results_email_async("subj", "body", log=capture)

    assert finished.wait(5.0)
    assert logs == ["[email] Đã gửi: subj", "[email] Response: RESP-OK"]


def test_send_results_email_async_logs_failure(monkeypatch):
    finished = threading.Event()
    logs = []

    def fake_send(subject, body):
        raise OSError("connection refused")

    def capture(msg):
        logs.append(msg)
        finished.set()

    monkeypatch.setattr(email_notifier, "send_email", fake_send)
    email_notifier.send_results_email_async("subj", "body", log=capture)

    assert finished.wait(5.0)
    assert len(logs) == 1
    assert logs[0] == "[email] Lỗi kết nối: connection refused"


def test_send_results_email_async_without_log_callback(monkeypatch):
    called = threading.Event()

    def fake_send(subject, body):
        called.set()
        return ("MSG", "RESP")

    monkeypatch.setattr(email_notifier, "send_email", fake_send)
    email_notifier.send_results_email_async("s", "b")

    assert called.wait(5.0)


# --------------------------------------------------------------------------
# load_metadata_events
# --------------------------------------------------------------------------

def test_load_metadata_events(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "v_metadata.json").write_text(
        json.dumps(
            {
                "source_video": "v.mp4",
                "fps": 30.0,
                "total_frames": 100,
                "events": EVENTS,
            }
        ),
        encoding="utf-8",
    )
    assert wf.load_metadata_events(out, "v") == EVENTS
    assert wf.load_metadata_events(out, "missing") is None
    (out / "c_metadata.json").write_text("{not json", encoding="utf-8")
    assert wf.load_metadata_events(out, "c") is None
    (out / "n_metadata.json").write_text("[1, 2]", encoding="utf-8")
    assert wf.load_metadata_events(out, "n") is None
    (out / "e_metadata.json").write_text('{"events": "nope"}', encoding="utf-8")
    assert wf.load_metadata_events(out, "e") is None


def test_count_events_still_works_via_helper(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "v_metadata.json").write_text(
        json.dumps({"events": EVENTS}), encoding="utf-8"
    )
    assert wf.count_events(out, "v") == 2
    assert wf.count_events(out, "missing") is None


# --------------------------------------------------------------------------
# WatchRunner sends the results email
# --------------------------------------------------------------------------

def _process_video(tmp_path, monkeypatch, content, video_name="cam.mp4"):
    sent = _collect_sends(monkeypatch)
    runner, folder, cfg = _make_runner(tmp_path, _fake_pipeline_raw(content))
    runner.run_cycle()  # folder first seen (empty)
    (folder / video_name).write_bytes(b"v" * 50)
    runner.run_cycle()  # first sighting: not stable yet
    runner.run_cycle()  # stable -> processed
    return runner, sent


def test_email_sent_once_on_success_with_events(tmp_path, monkeypatch):
    metadata = json.dumps(
        {"source_video": "cam.mp4", "fps": 30.0, "total_frames": 300, "events": EVENTS}
    )
    runner, sent = _process_video(tmp_path, monkeypatch, metadata)

    assert len(sent) == 1
    subject, body, log = sent[0]
    assert subject == "Kết quả xử lý - cam.mp4"
    output_dir = runner.journal["processed"][0]["output_dir"]
    assert body.startswith("<!DOCTYPE html>")
    assert "\n" not in body
    assert "cam.mp4" in body
    assert output_dir in body
    assert "vay tay" in body
    assert ">1.5<" in body
    assert ">3.2<" in body
    assert ">12.4<" in body
    assert log == runner._log


def test_email_body_sanitizes_pipe_characters(tmp_path, monkeypatch):
    events = [
        {"event_id": 1, "behavior": "vay|tay", "start_time_sec": 0.0, "end_time_sec": 1.0}
    ]
    metadata = json.dumps({"events": events})
    _runner, sent = _process_video(tmp_path, monkeypatch, metadata)

    assert len(sent) == 1
    subject, body, _log = sent[0]
    assert "|" not in subject
    assert "|" not in body
    assert "vay/tay" in body


def test_no_email_when_zero_events(tmp_path, monkeypatch):
    metadata = json.dumps({"events": []})
    runner, sent = _process_video(tmp_path, monkeypatch, metadata)

    assert sent == []
    assert runner.journal["processed"][0]["events"] == 0


def test_no_email_when_metadata_corrupt(tmp_path, monkeypatch):
    _runner, sent = _process_video(tmp_path, monkeypatch, "{not json")
    assert sent == []


def test_no_email_when_metadata_missing(tmp_path, monkeypatch):
    _runner, sent = _process_video(tmp_path, monkeypatch, None)

    assert sent == []


def test_no_email_for_failed_video(tmp_path, monkeypatch):
    sent = _collect_sends(monkeypatch)
    calls = []

    def failing(video_path, **kwargs):
        calls.append(video_path)
        raise RuntimeError("boom")

    runner, folder, cfg = _make_runner(tmp_path, failing)
    runner.run_cycle()  # folder first seen (empty)
    (folder / "bad.mp4").write_bytes(b"v" * 50)
    runner.run_cycle()  # sighting
    runner.run_cycle()  # attempt 1 -> retry
    runner.run_cycle()  # attempt 2 -> gave up

    assert len(calls) == 2
    assert runner.journal["errors"][str(folder / "bad.mp4")]["gave_up"] is True
    assert sent == []
