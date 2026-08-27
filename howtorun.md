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
python cut_video.py "D:\Video\CA170-FB-PC-No5-20260824-114244.mp4" 0 180

## Keypoint classifier (one model per behavior)

The optional keypoint classifier refines rule-based detections with a trained
MLP (one-vs-rest, one model per behavior). Three stages:

### 1. Collect a dataset (`dataset_tool.py`)

```bash
python dataset_tool.py --model yolo26n-pose.pt --dataset ./dataset/
```

Load a video, run detection over a frame range, pick a track, set a start/end
range, choose a behavior (or tick "Negative example"), and "Add sample".
Each sample is stored as `dataset/samples/sample_XXXX.npz` + a row in
`dataset/metadata.csv`.

### 2. Train a per-behavior model (`train_classifier.py`)

```bash
python train_classifier.py --dataset ./dataset/ --behavior hand_to_head --output ./models/
python train_classifier.py --dataset ./dataset/ --behavior body_turn --output ./models/ --mode agg
```

Modes: `temporal` (resample to 32 frames + flatten, ~3306 dims, needs more
data) or `agg` (statistics aggregation, ~422 dims, better for small datasets).
Saves `models/<behavior>/{classifier.pkl, scaler.pkl, metadata.json}`.

### 3. Enable in pipeline config

Add a `classifier` section to your config YAML:

```yaml
classifier:
  behaviors:
    hand_to_head:
      model_path: ./models/hand_to_head/classifier.pkl
      scaler_path: ./models/hand_to_head/scaler.pkl
      metadata_path: ./models/hand_to_head/metadata.json
      threshold: 0.82
```

Events whose behavior has a classifier are kept only if `P(positive) >= threshold`
(computed from the event's keypoint sequence). Behaviors without a model, or
sequences shorter than `min_sequence_frames`, pass through unfiltered.


Cách dùng:

Build lại khi cần: powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1
Kết quả: thư mục dist\HandHeadGUI\ (~1.24 GB) chứa HandHeadGUI.exe + _internal\ + config\, models\, yolo*.pt
Copy cho khách: zip cả thư mục dist\HandHeadGUI → khách giải nén và double-click HandHeadGUI.exe. Không cần cài Python hay internet.