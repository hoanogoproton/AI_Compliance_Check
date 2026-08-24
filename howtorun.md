# Usage

## Basic (hand_to_head only, no config needed)

```bash
python cli.py --video "videos/1.mp4" --conf 0.3 --iou 0.5 --visualize
```

## With config YAML (multiple behaviors + zones)

Create `config.yaml` (see example in project root), then:

```bash
python cli.py --video "videos/1.mp4" --config config.yaml --visualize
```

## Zone definition tool

```bash
python cli.py --video "videos/1.mp4" --define-zones
```

Saves zones into `config.yaml` (or specified `--config` file).

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--video` | (required) | Input video path |
| `--config` | None | YAML config with behaviors/zones |
| `--define-zones` | False | Launch interactive zone polygon tool |
| `--model` | yolo11n-pose.pt | YOLO pose model |
| `--output-dir` | ./outputs | Output directory |
| `--conf` | 0.3 | Detection confidence |
| `--iou` | 0.5 | NMS IoU threshold |
| `--visualize` | False | Generate annotated video |
| `--context-seconds` | 5 | Context before/after event clips |
| `--crop-padding` | 20 | Padding around cropped events |
| `--debug-keypoints` | False | Export clips with skeleton overlay |

## Running tests

```bash
pytest tests/ -v
```
python -m gui.app
python debug_tracks.py --video "videos/1.mp4" --config config/config.yaml
python cut_video.py videos/RAI7-1.mp4 10.5 30.2