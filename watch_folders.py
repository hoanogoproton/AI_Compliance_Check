"""
Folder Watcher: monitor folders listed in a CSV file and automatically run the
detection pipeline on every NEW video that appears, using that folder's own
config.

CSV format (2 required columns + 1 optional):

    folder,config[,output_dir]
    D:/CameraDrop/RAI7,config/config_4.yaml
    ./videos/watch_demo,./config/config_3.yaml,./outputs/custom

Behavior:
    * Videos already inside a folder when the watcher first sees it are
      IGNORED (recorded in the journal as `ignored_initial`). Only videos
      that appear afterwards are processed.
    * A new video is picked up once its file size has stayed unchanged
      between two polls (guards against videos that are still being copied).
    * After a successful run the original video is DELETED. On failure the
      video is kept and retried up to --max-retries times.
    * A journal (default: outputs/watch_state.json) records processed /
      failed / ignored videos for auditing (the originals are deleted).

Usage:
    python watch_folders.py watch_folders.csv
    python watch_folders.py watch_folders.csv --poll-interval 15 --no-visualize

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts"}
JOURNAL_NAME = "watch_state.json"
DEFAULT_MAX_RETRIES = 2


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def canonical(path: str | Path) -> str:
    """Absolute, case-normalized form of a path (stable on Windows)."""
    return os.path.normcase(str(Path(path).resolve()))


def _resolve_path(value: str, base_dirs: list[Path]) -> Path:
    """Resolve a CSV path: absolute is kept; relative tried against base_dirs."""
    p = Path(value.strip())
    if p.is_absolute():
        return p
    for base in base_dirs:
        candidate = base / p
        if candidate.exists():
            return candidate
    return base_dirs[0] / p


def load_watch_csv(csv_path: str | Path) -> tuple[list[dict], list[str]]:
    """Parse the watch CSV (columns: folder,config[,output_dir]).

    Returns (entries, errors). Relative paths are resolved against the CSV
    file's directory first, then the current working directory.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    base_dirs = [csv_path.resolve().parent, Path.cwd()]
    entries: list[dict] = []
    errors: list[str] = []
    seen_folders: set[str] = set()

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = [(x or "").strip().lower() for x in (reader.fieldnames or [])]
        if len(fields) < 2 or fields[0] != "folder" or fields[1] != "config":
            raise ValueError(
                "CSV must have columns: folder,config (optional third column: output_dir)"
            )
        has_output_col = "output_dir" in fields

        for line_no, row in enumerate(reader, start=2):
            folder = (row.get("folder") or "").strip()
            config = (row.get("config") or "").strip()
            output_dir = (row.get("output_dir") or "").strip() if has_output_col else ""

            if not folder or not config:
                errors.append(f"line {line_no}: 'folder' and 'config' are required")
                continue

            folder_path = _resolve_path(folder, base_dirs)
            key = canonical(folder_path)
            if key in seen_folders:
                errors.append(f"line {line_no}: duplicate folder '{folder}'")
                continue
            seen_folders.add(key)

            entries.append({
                "folder": folder_path,
                "config": _resolve_path(config, base_dirs),
                "output_dir": _resolve_path(output_dir, base_dirs) if output_dir else None,
                "folder_raw": folder,
                "config_raw": config,
            })

    return entries, errors


def is_file_stable(path: str | Path, last_sizes: dict[str, int]) -> bool:
    """True when the file size is unchanged since the previous poll.

    The first time a file is seen it is never stable, so videos that are
    still being copied into the folder are not processed too early.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        return False
    key = canonical(path)
    previous = last_sizes.get(key)
    last_sizes[key] = size
    return previous is not None and previous == size


def load_journal(path: str | Path) -> dict:
    """Load the watcher journal, returning a fresh structure when missing/corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("processed", [])
            data.setdefault("errors", {})
            data.setdefault("ignored_initial", {})
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"processed": [], "errors": {}, "ignored_initial": {}}


def save_journal(path: str | Path, journal: dict) -> None:
    """Atomically write the journal (best-effort; failures are logged only)."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(journal, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        _log(f"WARNING: could not save journal {path}: {e}")


def count_events(output_dir: str | Path, video_stem: str) -> int | None:
    """Read `<stem>_metadata.json` written by the pipeline and count events."""
    meta_path = Path(output_dir) / f"{video_stem}_metadata.json"
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events") if isinstance(data, dict) else None
        return len(events) if isinstance(events, list) else None
    except (OSError, json.JSONDecodeError):
        return None


class WatchRunner:
    """Polls the folders from the CSV and processes every new video.

    An optional ``event_callback`` (used by the realtime GUI) receives a dict
    per lifecycle event (``startup``, ``folder_seen``, ``video_new``,
    ``video_start``, ``video_progress``, ``video_done``, ``video_error``,
    ``log``, ``stopped``). When it is ``None`` (CLI) nothing is emitted and
    behavior is unchanged.
    """

    def __init__(
        self,
        csv_path: str | Path,
        poll_interval: float = 10.0,
        output_dir: str | Path = "./outputs",
        visualize: bool = True,
        model_path: str = "yolo11n-pose.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        context_seconds: int = 5,
        crop_padding: int = 20,
        debug_keypoints: bool = False,
        journal_path: str | Path | None = None,
        max_cycles: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        pipeline_fn=None,
        event_callback: Callable[[dict], None] | None = None,
        frame_callback: Callable | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.poll_interval = max(0.1, float(poll_interval))
        self.output_base = Path(output_dir)
        self.visualize = bool(visualize)
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.context_seconds = context_seconds
        self.crop_padding = crop_padding
        self.debug_keypoints = debug_keypoints
        self.max_cycles = max_cycles
        self.max_retries = max(1, int(max_retries))
        self.journal_path = (
            Path(journal_path) if journal_path else self.output_base / JOURNAL_NAME
        )
        self.journal = load_journal(self.journal_path)

        # The heavy pipeline import (ultralytics/torch, takes seconds) is
        # deferred until the first video is actually processed, so the
        # startup snapshot runs immediately and no file dropped during model
        # loading can be mistaken for a pre-existing video.
        self._pipeline = pipeline_fn  # imported on first use when None
        self.event_callback = event_callback
        # Optional realtime preview bridge called as (key, frame_idx, rgb, info)
        # with `key` being the canonical path of the video being processed.
        self.frame_callback = frame_callback

        self.entries: list[dict] = []
        self.known_folders: set[str] = set()
        self.snapshot: set[str] = set()        # videos present when a folder is first seen
        self.last_sizes: dict[str, int] = {}   # canonical path -> size at previous poll
        self.queue: deque = deque()
        self.queued_keys: set[str] = set()
        self.processing: set[str] = set()
        self.failed: set[str] = set()          # gave up after max_retries (this run)
        self.retry_counts: dict[str, int] = {}
        self.warned: set[str] = set()          # one-time warnings
        self.claimed_stems: dict[str, str] = {}  # video stem -> folder key
        self._output_bases: set[str] = {canonical(self.output_base)}
        self.stats = {"processed": 0, "deleted": 0, "failed": 0}
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    # -- event callback ------------------------------------------------------

    def _emit(self, event_type: str, **fields) -> None:
        """Forward a lifecycle event to the GUI callback (best effort)."""
        if self.event_callback is None:
            return
        try:
            self.event_callback({"type": event_type, **fields})
        except Exception as e:  # noqa: BLE001 — a dead GUI must not kill the watcher
            self.event_callback = None
            _log(f"WARNING: event callback disabled after error: {e}")

    def _log(self, message: str) -> None:
        """Log to stdout like the CLI and mirror the message to the GUI."""
        _log(message)
        self._emit("log", message=message)

    # -- CSV / entries ------------------------------------------------------

    def _load_entries(self) -> list[dict]:
        entries, errors = load_watch_csv(self.csv_path)
        for err in errors:
            warn_key = f"csv:{err}"
            if warn_key not in self.warned:
                self.warned.add(warn_key)
                self._log(f"CSV ERROR: {err}")
        valid: list[dict] = []
        for entry in entries:
            if not entry["config"].exists():
                warn_key = f"cfg:{canonical(entry['config'])}"
                if warn_key not in self.warned:
                    self.warned.add(warn_key)
                    self._log(
                        f"ERROR: config not found for folder "
                        f"'{entry['folder_raw']}': {entry['config']}"
                    )
                continue
            if not entry["folder"].exists():
                warn_key = f"dir:{canonical(entry['folder'])}"
                if warn_key not in self.warned:
                    self.warned.add(warn_key)
                    self._log(
                        f"WARNING: folder does not exist yet (will keep polling): "
                        f"{entry['folder']}"
                    )
            valid.append(entry)
        return valid

    def _refresh_output_bases(self) -> None:
        bases = {canonical(self.output_base)}
        for entry in self.entries:
            if entry["output_dir"]:
                bases.add(canonical(entry["output_dir"]))
        self._output_bases = bases


    # -- scanning -----------------------------------------------------------

    def _iter_videos(self, folder: Path) -> list[Path]:
        try:
            children = sorted(folder.iterdir())
        except OSError as e:
            self._log(f"WARNING: cannot read folder {folder}: {e}")
            return []
        videos: list[Path] = []
        for child in children:
            try:
                if not child.is_file():
                    continue
            except OSError:
                continue
            name = child.name
            if name.startswith(".") or name.startswith("~$"):
                continue  # hidden / lock files
            if child.suffix.lower() not in VIDEO_EXTENSIONS:
                continue  # also skips *.tmp / *.part partial downloads
            videos.append(child)
        return videos

    def _is_under_output(self, path: Path) -> bool:
        p = canonical(path)
        for base in self._output_bases:
            if p == base or p.startswith(base + os.sep):
                return True
        return False

    def _scan_folders(self) -> None:
        for entry in self.entries:
            folder = entry["folder"]
            if not folder.is_dir():
                continue  # already warned in _load_entries
            folder_key = canonical(folder)
            videos = self._iter_videos(folder)
            if folder_key not in self.known_folders:
                # First time the folder is visible: ignore whatever is already
                # in it — only videos that appear afterwards are processed.
                self.known_folders.add(folder_key)
                ignored_names = [v.name for v in videos]
                if videos:
                    self.snapshot.update(canonical(v) for v in videos)
                    self.journal.setdefault("ignored_initial", {})[str(folder)] = [
                        v.name for v in videos
                    ]
                    save_journal(self.journal_path, self.journal)
                    self._log(
                        f"Folder first seen: {folder} - "
                        f"{len(videos)} existing video(s) will be IGNORED."
                    )
                self._emit(
                    "folder_seen",
                    folder=str(folder),
                    folder_raw=entry["folder_raw"],
                    ignored=len(ignored_names),
                    videos=ignored_names,
                )
            for video in videos:
                self._maybe_enqueue(video, entry)

    def _output_dir_for(self, entry: dict, video: Path) -> Path:
        base = entry["output_dir"] or self.output_base
        stem = video.stem
        stem_key = stem.lower()
        owner = self.claimed_stems.get(stem_key)
        if owner is not None and owner != canonical(entry["folder"]):
            # Same video name from a different folder -> prefix the folder name.
            out = base / f"{entry['folder'].name}_{stem}"
        else:
            out = base / stem
        self.claimed_stems.setdefault(stem_key, canonical(entry["folder"]))
        return out

    def _maybe_enqueue(self, video: Path, entry: dict) -> None:
        key = canonical(video)
        if (
            key in self.snapshot
            or key in self.queued_keys
            or key in self.processing
            or key in self.failed
        ):
            return
        if self._is_under_output(video):
            return  # never re-ingest our own exported clips
        if not is_file_stable(video, self.last_sizes):
            return  # first sighting or file still being copied
        item = {
            "video": video,
            "config": entry["config"],
            "config_raw": entry["config_raw"],
            "folder_raw": entry["folder_raw"],
            "output_dir": self._output_dir_for(entry, video),
            "key": key,
        }
        self._log(f"NEW video: {video}")
        self._emit(
            "video_new",
            key=key,
            video=str(video),
            video_name=video.name,
            folder=entry["folder_raw"],
            config=entry["config_raw"],
            output_dir=str(item["output_dir"]),
        )
        self.queue.append(item)
        self.queued_keys.add(key)


    # -- processing -----------------------------------------------------------

    def _process_queue(self) -> None:
        while self.queue and not self._stop.is_set():
            item = self.queue.popleft()
            key = item["key"]
            self.queued_keys.discard(key)
            self.processing.add(key)
            video = item["video"]
            if self._pipeline is None:
                from detection.pipeline import run_pipeline  # heavy, first use only

                self._pipeline = run_pipeline
            try:
                self._log(f"PROCESSING: {video.name} (config: {item['config_raw']})")
                self._log(f"  output: {item['output_dir']}")
                self._emit(
                    "video_start",
                    key=key,
                    video_name=video.name,
                    config=item["config_raw"],
                    output_dir=str(item["output_dir"]),
                )
                extra_kwargs: dict = {}
                if self.event_callback is not None:
                    def on_progress(
                        frame: int, total: int, _key: str = key, _name: str = video.name
                    ) -> None:
                        # Throttle to ~1% steps so the GUI is not flooded.
                        if total <= 0:
                            return
                        if frame == total or frame % max(1, total // 100) == 0:
                            self._emit(
                                "video_progress",
                                key=_key,
                                video_name=_name,
                                frame=frame,
                                total=total,
                            )

                    extra_kwargs["progress_callback"] = on_progress
                    extra_kwargs["log_callback"] = self._log
                if self.frame_callback is not None:
                    def on_frame(
                        frame_idx: int, frame_rgb, info: dict, _key: str = key
                    ) -> None:
                        # Bind the video key so the GUI knows which video the
                        # realtime preview frame belongs to.
                        self.frame_callback(_key, frame_idx, frame_rgb, info)

                    extra_kwargs["frame_callback"] = on_frame
                self._pipeline(
                    video_path=str(video.resolve()),
                    model_path=self.model_path,
                    output_dir=str(item["output_dir"]),
                    conf=self.conf,
                    iou=self.iou,
                    visualize=self.visualize,
                    context_seconds=self.context_seconds,
                    crop_padding=self.crop_padding,
                    debug_keypoints=self.debug_keypoints,
                    config_path=str(item["config"]),
                    **extra_kwargs,
                )
                self._on_success(item)
            except KeyboardInterrupt:
                self._log(f"INTERRUPTED while processing {video.name} - video kept on disk.")
                raise
            except Exception as e:  # noqa: BLE001 — one bad video must not stop the watcher
                self._on_error(item, e)
            finally:
                self.processing.discard(key)

    def _on_success(self, item: dict) -> None:
        video = item["video"]
        key = item["key"]
        events = count_events(item["output_dir"], video.stem)
        deleted = False
        delete_error = None
        try:
            video.unlink()
            deleted = True
        except OSError as e:
            delete_error = str(e)
        if deleted:
            self.last_sizes.pop(key, None)
            self._log(
                f"DONE: {video.name} - "
                f"{events if events is not None else '?'} event(s); original deleted."
            )
        else:
            # Keep the processed video out of future scans so it is not
            # re-processed endlessly when deletion fails (e.g. file locked).
            self.snapshot.add(key)
            self._log(f"DONE: {video.name} but could NOT delete it: {delete_error}")
        self.stats["processed"] += 1
        if deleted:
            self.stats["deleted"] += 1
        self.journal["processed"].append({
            "video": str(video),
            "folder": item["folder_raw"],
            "config": item["config_raw"],
            "output_dir": str(item["output_dir"]),
            "events": events,
            "deleted": deleted,
            "delete_error": delete_error,
            "finished_at": _now(),
        })
        save_journal(self.journal_path, self.journal)
        self._emit(
            "video_done",
            key=key,
            video_name=video.name,
            events=events,
            deleted=deleted,
            delete_error=delete_error,
        )

    def _on_error(self, item: dict, exc: Exception) -> None:
        video = item["video"]
        key = item["key"]
        attempts = self.retry_counts.get(key, 0) + 1
        self.retry_counts[key] = attempts
        gave_up = attempts >= self.max_retries
        if gave_up:
            self.failed.add(key)
            self.stats["failed"] += 1
            self._log(
                f"ERROR: {video.name} failed {attempts} time(s) - giving up, "
                f"video kept on disk: {exc}"
            )
        else:
            self._log(
                f"ERROR: {video.name} (attempt {attempts}/{self.max_retries}) - "
                f"will retry on the next poll: {exc}"
            )
        self.journal.setdefault("errors", {})[str(video)] = {
            "attempts": attempts,
            "last_error": str(exc),
            "last_at": _now(),
            "gave_up": gave_up,
        }
        save_journal(self.journal_path, self.journal)
        self._emit(
            "video_error",
            key=key,
            video_name=video.name,
            attempts=attempts,
            max_retries=self.max_retries,
            gave_up=gave_up,
            error=str(exc),
        )


    # -- main loop ------------------------------------------------------------

    def run_cycle(self) -> None:
        """One poll: reload CSV, scan folders, process whatever is queued."""
        self.entries = self._load_entries()
        self._refresh_output_bases()
        self._scan_folders()
        self._process_queue()

    def run(self) -> int:
        ignored = sum(
            len(names) for names in self.journal.get("ignored_initial", {}).values()
        )
        self._log("Folder watcher started.")
        self._log(f"  watch CSV  : {self.csv_path}")
        self._log(f"  poll every : {self.poll_interval:g}s | visualize: {self.visualize}")
        self._log(f"  journal    : {self.journal_path}")
        self._log("Press Ctrl+C to stop.")
        self._emit(
            "startup",
            csv_path=str(self.csv_path),
            poll_interval=self.poll_interval,
            visualize=self.visualize,
            journal_path=str(self.journal_path),
            journal_processed=len(self.journal.get("processed", [])),
            journal_errors=len(self.journal.get("errors", {})),
            ignored_initial=ignored,
        )
        cycles = 0
        try:
            while not self._stop.is_set():
                if self.max_cycles is not None and cycles >= self.max_cycles:
                    break
                started = time.monotonic()
                try:
                    self.run_cycle()
                except (FileNotFoundError, ValueError) as e:
                    self._log(f"ERROR: {e} - CSV will be retried on the next poll.")
                cycles += 1
                if self.max_cycles is not None and cycles >= self.max_cycles:
                    break
                remaining = self.poll_interval - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except KeyboardInterrupt:
            self._log("Interrupted by user (Ctrl+C).")
        finally:
            self._print_summary()
        return 0

    def _print_summary(self) -> None:
        ignored = sum(
            len(names) for names in self.journal.get("ignored_initial", {}).values()
        )
        self._log(
            "Watcher stopped. "
            f"processed={self.stats['processed']} "
            f"(deleted={self.stats['deleted']}), "
            f"failed={self.stats['failed']}, ignored_existing={ignored}."
        )
        self._log(f"Journal: {self.journal_path}")
        self._emit(
            "stopped",
            processed=self.stats["processed"],
            deleted=self.stats["deleted"],
            failed=self.stats["failed"],
            ignored=ignored,
            journal_path=str(self.journal_path),
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Watch folders from a CSV (folder,config[,output_dir]) and "
            "automatically run detection on every new video."
        )
    )
    p.add_argument(
        "csv",
        help="Path to the watch CSV file (columns: folder,config[,output_dir]).",
    )
    p.add_argument(
        "--poll-interval", type=float, default=10.0,
        help="Seconds between folder scans (default: 10).",
    )
    p.add_argument(
        "--output-dir", default="./outputs",
        help="Base output directory (default: ./outputs).",
    )
    p.add_argument(
        "--no-visualize", action="store_true",
        help="Skip the full annotated video (event clips are always exported).",
    )
    p.add_argument(
        "--model", default="yolo11n-pose.pt",
        help="YOLO pose model path or name (default: yolo11n-pose.pt).",
    )
    p.add_argument("--conf", type=float, default=0.3,
                   help="Detection confidence threshold (default: 0.3).")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold (default: 0.5).")
    p.add_argument("--context-seconds", type=int, default=5,
                   help="Context seconds around event clips (default: 5).")
    p.add_argument("--crop-padding", type=int, default=20,
                   help="Padding pixels around crops (default: 20).")
    p.add_argument("--debug-keypoints", action="store_true",
                   help="Export extra debug clips with skeleton overlay.")
    p.add_argument(
        "--journal", default=None,
        help=f"Journal file path (default: <output-dir>/{JOURNAL_NAME}).",
    )
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help="Retries per video before giving up (default: 2).")
    p.add_argument(
        "--max-cycles", type=int, default=None,
        help="Stop after N poll cycles (for testing; default: run forever).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        load_watch_csv(args.csv)  # early validation of the CSV
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    runner = WatchRunner(
        csv_path=args.csv,
        poll_interval=args.poll_interval,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        context_seconds=args.context_seconds,
        crop_padding=args.crop_padding,
        debug_keypoints=args.debug_keypoints,
        journal_path=args.journal,
        max_cycles=args.max_cycles,
        max_retries=args.max_retries,
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())