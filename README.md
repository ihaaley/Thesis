# Skateboard / Longboard Pose Analysis & Trick Classification

A real-time computer-vision system that tracks skaters, analyses body mechanics,
and classifies tricks — using YOLOv8-Pose for skeleton detection and a trained
BiLSTM neural network for trick recognition.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Step 1 — Real-Time Analysis (main.py)](#5-step-1--real-time-analysis-mainpy)
6. [Step 2 — Feature Extraction (extract_features.py)](#6-step-2--feature-extraction-extract_featurespy)
7. [Step 3 — Model Training (train.py)](#7-step-3--model-training-trainpy)
8. [Step 4 — Run with Trained Classifier](#8-step-4--run-with-trained-classifier)
9. [Shared Utilities (trick_utils.py)](#9-shared-utilities-trick_utilspy)
10. [Analytics & Metrics Explained](#10-analytics--metrics-explained)
11. [Visual Enhancements](#11-visual-enhancements)
12. [Gamification & Scoring](#12-gamification--scoring)
13. [Session Database (session_db.py)](#13-session-database-session_dbpy)
14. [HTML Report (report.py)](#14-html-report-reportpy)
15. [Web Dashboard (app.py)](#15-web-dashboard-apppy)
16. [Dataset Preparation](#16-dataset-preparation)
17. [Configuration & Tuning](#17-configuration--tuning)
18. [Output Files](#18-output-files)
19. [Key Design Decisions](#19-key-design-decisions)

---

## 1. Project Overview

The system solves two related problems:

| Problem                     | Solution                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------- |
| **Real-time body tracking** | YOLOv8-Pose extracts 17 skeleton joints per player every frame                        |
| **Trick recognition**       | A custom-trained BiLSTM classifies trick sequences from normalized joint trajectories |

The pipeline works on pre-recorded video files **and** live webcam feeds. Multiple
skaters are tracked simultaneously, each receiving their own colour-coded overlay,
panel with analytics, and trick label.

---

## 2. System Architecture

```
Video / Webcam
      │
      ▼
┌─────────────────────┐
│  YOLOv8-Pose        │  → 17 COCO keypoints per person, with confidence
│  (yolov8s-pose.pt)  │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  YOLOv8-Det         │  → Skateboard bounding box (COCO class 36)
│  (yolov8s.pt)       │
└─────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  Per-Player Analytics                                    │
│   • Speed (ankle displacement → km/h proxy)              │
│   • Knee angles (hip–knee–ankle vector maths)            │
│   • Jump height (skateboard bottom vs ground)            │
│   • Airtime timer + landing quality score                │
│   • Jump angle (arc-tangent of peak displacement)        │
│   • Rule-based trick detection (height + angle)          │
│   • Phase classification (ground/rising/peak/landing)    │
│   • Stance detection (regular / goofy foot)              │
│   • Bail detection (fall heuristic)                      │
│   • Trick scoring + session leaderboard                  │
│   • Achievement badges                                   │
│   • Coaching hints                                       │
└──────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  Visual Overlays                                         │
│   • Phase-coloured skeleton (green/cyan/blue/lime)       │
│   • Joint motion trails (fading per-frame history)       │
│   • Mini speed + height line graphs                      │
│   • Trick score flash banner                             │
│   • Achievement golden-stripe banner                     │
│   • Bottom-left leaderboard                              │
└──────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  Persistence & Reporting                                 │
│   • SQLite session DB (session_db.py)                    │
│   • Slo-mo highlight clip export                         │
│   • HTML session report (report.py)                      │
│   • Flask live web dashboard (app.py)                    │
└──────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────┐
│  BiLSTM Trick Classifier         │
│   Sliding window: 30 frames      │
│   Input: 51-d normalised joints  │
│   Output: trick label + prob %   │
└──────────────────────────────────┘
      │
      ▼
Annotated frame displayed / saved
```

---

## 3. Project Structure

```
Thesis/
├── main.py               ← Real-time pipeline (run this to watch video)
├── extract_features.py   ← Step 1 of ML training: video → numpy arrays
├── train.py              ← Step 2 of ML training: arrays → trained model
├── trick_utils.py        ← Shared: normalisation function + model definition
├── session_db.py         ← SQLite session logging module
├── report.py             ← HTML session summary report generator
├── app.py                ← Flask live web dashboard
│
├── data/
│   └── tricks/           ← Your labeled video dataset (see §16)
│       ├── carve/
│       ├── ollie/
│       └── ...
│
├── features/             ← Created by extract_features.py
│   ├── X.npy             ← Feature windows  (N, 30, 51)
│   ├── y.npy             ← Integer labels   (N,)
│   ├── label_map.json    ← {"carve": 0, "slide": 1, ...}
│   ├── trick_classifier.pt  ← Best model weights (created by train.py)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── training_report.txt
│
├── highlights/           ← Slo-mo highlight clips (created by --highlights-dir)
├── session.db            ← SQLite session database (created by --db)
│
├── yolov8s-pose.pt       ← Downloaded automatically by ultralytics
├── yolov8s.pt
└── venv/                 ← Python virtual environment
```

---

## 4. Installation

### Requirements

- Python 3.10 or later
- macOS / Linux / Windows
- GPU optional (Apple Silicon MPS or NVIDIA CUDA accelerates training)

### Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install all dependencies
pip install ultralytics opencv-python numpy torch torchvision \
            scikit-learn matplotlib seaborn

# 3. YOLOv8 weights download automatically on first run.
#    To download manually:
python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt'); YOLO('yolov8s.pt')"
```

---

## 5. Step 1 — Real-Time Analysis (`main.py`)

This is the main entry point. It plays a video (or opens a webcam), draws
skeleton overlays and per-player panels, and — optionally — shows predictions
from the trained classifier.

### Basic usage

```bash
# Analyse a video file
python main.py --source data/video.mp4

# Use webcam (device 0)
python main.py --source 0

# Save the annotated result to a file
python main.py --source data/video.mp4 --save --output result.mp4

# Export per-frame analytics to CSV
python main.py --source data/video.mp4 --log-csv

# Use a trained trick classifier (see §7–8)
python main.py --source data/video.mp4 \
  --trick-model features/trick_classifier.pt \
  --label-map   features/label_map.json
```

### All CLI arguments

| Argument           | Default      | Description                                                |
| ------------------ | ------------ | ---------------------------------------------------------- |
| `--source`         | `0`          | Video file path or `0` for webcam                          |
| `--model`          | `s`          | YOLOv8 size: `n` nano · `s` small · `m` medium · `l` large |
| `--conf`           | `0.25`       | Detection confidence threshold (0–1)                       |
| `--iou`            | `0.45`       | NMS IoU overlap threshold (0–1)                            |
| `--save`           | off          | Save annotated video to `--output`                         |
| `--output`         | `output.mp4` | Output video file path                                     |
| `--log-csv`        | off          | Write per-frame CSV data log                               |
| `--trick-model`    | none         | Path to `trick_classifier.pt`                              |
| `--label-map`      | none         | Path to `label_map.json`                                   |
| `--highlights-dir` | none         | Directory to save 4× slo-mo highlight clips                |
| `--db`             | none         | Path for SQLite session database (e.g. `session.db`)       |
| `--ref-poses`      | none         | Path to reference poses JSON for form grading              |

### Keyboard controls during playback

| Key | Action                                          |
| --- | ----------------------------------------------- |
| `q` | Quit                                            |
| `p` | Pause / Resume                                  |
| `r` | Reset ground baseline and all player statistics |

### What the overlay shows

**Per-player panel** (top-right corner, one panel per tracked person):

```
┌─────────────────────────────┐
│ PLAYER  P001                │  ← Track ID + player-colour border
│ Conf     : 0.92             │  ← YOLOv8 detection confidence
│ Speed    : 12.3 km/h        │  ← Ankle-displacement proxy
│ Ankle Y  : 487.2 px         │  ← Vertical ankle position
│ Knee Ang : L142° R138°      │  ← Left/right knee angles
│ Phase    : RISING           │  ← GROUND / RISING / PEAK / LANDING
│ Stance   : REGULAR          │  ← REGULAR or GOOFY (majority vote)
│ Bail     : --               │  ← Flashes "BAIL!!!" on fall detection
│ ── Skateboard ──            │
│ Status   : IN AIR           │  ← ON GROUND or IN AIR
│ Height   :  8.3 %  (60px)   │  ← Skateboard height above ground
│ Max Ht   : 12.1 %           │  ← Session peak height
│ Airtime  : 0.42s  max:0.68s │  ← Current and best airtime
│ JumpAng  : 34.7 deg         │  ← Estimated jump trajectory angle
│ Trick(ML): CARVE 87%        │  ← ML classifier output (or rule-based)
│ LandScr  : 82/100           │  ← Landing quality (knee-angle smoothness)
│ Score    : 12,450           │  ← Cumulative session score
│ Coach    : Bend knees more  │  ← Contextual coaching hint
└─────────────────────────────┘
```

When the ML classifier is not loaded, the Trick line falls back to rule-based
detection using height thresholds and knee angles.

---

## 6. Step 2 — Feature Extraction (`extract_features.py`)

Before training the trick classifier you must convert your video dataset into
numerical feature arrays. This script does that automatically.

### How it works

1. Walks every subfolder in `--dataset`; the **folder name becomes the class label**
2. For each video, samples every `--frame-skip`-th frame
3. Runs YOLOv8-Pose on each sampled frame
4. Picks the highest-confidence person detection
5. Normalises the 17 keypoints (see §9 for normalisation details)
6. Groups frames into overlapping sliding windows of length `--window`
7. Saves all windows as numpy arrays ready for training

### Run feature extraction

```bash
# Minimal — uses defaults (frame-skip=2, window=30, stride=15)
python extract_features.py --dataset data/tricks --output features/

# Custom settings
python extract_features.py \
  --dataset   data/tricks \
  --output    features/ \
  --frame-skip 3       \   # use every 3rd frame (faster, less overlap)
  --window    30       \   # 30-frame input sequences
  --stride    10       \   # 10-frame step between windows (more data, more overlap)
  --conf      0.30         # YOLO confidence threshold for person detection
```

### CLI arguments

| Argument       | Default    | Description                                    |
| -------------- | ---------- | ---------------------------------------------- |
| `--dataset`    | required   | Root folder with one subfolder per trick class |
| `--output`     | `features` | Destination folder for output files            |
| `--model`      | `s`        | YOLOv8-Pose model size                         |
| `--frame-skip` | `2`        | Process every N-th frame                       |
| `--window`     | `30`       | Sliding window length (must match training)    |
| `--stride`     | `15`       | Step between consecutive windows               |
| `--conf`       | `0.25`     | YOLO detection confidence threshold            |

### Output files

| File             | Shape         | Description                                                    |
| ---------------- | ------------- | -------------------------------------------------------------- |
| `X.npy`          | `(N, 30, 51)` | Feature windows — N samples, 30 frames each, 51 values         |
| `y.npy`          | `(N,)`        | Integer class label for each window                            |
| `label_map.json` | —             | Maps trick name → integer index                                |
| `stats.json`     | —             | Per-class counts: videos processed, windows extracted, skipped |

The 51 values per frame come from 17 joints × 3 values each: **normalised x, y, confidence**.

---

## 7. Step 3 — Model Training (`train.py`)

Trains a Bidirectional LSTM neural network on the feature arrays produced in §6.

### How it works

1. Loads `X.npy`, `y.npy`, `label_map.json` from the features folder
2. Performs a **stratified 70 / 15 / 15** train / validation / test split
   (each class has the same proportion in every split)
3. Computes class weights to handle imbalanced datasets
4. Trains a **BiLSTM → LayerNorm → Linear(64) → Linear(num_classes)** model
5. Saves the **best checkpoint** (highest validation accuracy) automatically
6. Applies early stopping if validation accuracy stops improving
7. Produces training plots and a full classification report

### Run training

```bash
# Default settings (recommended starting point)
python train.py --features features/

# Fine-tuned run
python train.py \
  --features features/ \
  --epochs   100        \
  --batch    64         \
  --lr       5e-4       \
  --hidden   128        \   # LSTM hidden units per direction
  --layers   2          \   # number of stacked LSTM layers
  --dropout  0.3        \
  --patience 20             # stop if no improvement for 20 epochs
```

### CLI arguments

| Argument         | Default    | Description                                 |
| ---------------- | ---------- | ------------------------------------------- |
| `--features`     | `features` | Folder containing the numpy arrays          |
| `--epochs`       | `80`       | Maximum training epochs                     |
| `--batch`        | `64`       | Mini-batch size                             |
| `--lr`           | `1e-3`     | Initial learning rate                       |
| `--hidden`       | `128`      | LSTM hidden state size (per direction)      |
| `--layers`       | `2`        | Number of stacked LSTM layers               |
| `--dropout`      | `0.3`      | Dropout probability                         |
| `--patience`     | `15`       | Early stopping — epochs without improvement |
| `--weight-decay` | `1e-4`     | L2 regularisation                           |

### What training produces

| File                            | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| `features/trick_classifier.pt`  | Best model weights (use with `--trick-model`)     |
| `features/training_curves.png`  | Loss and accuracy curves for train + validation   |
| `features/confusion_matrix.png` | Normalised per-class confusion matrix on test set |
| `features/training_report.txt`  | Precision, recall, F1 per class on test set       |

### Training techniques used

- **Class-weighted loss**: rarer tricks get higher loss weight, preventing the model
  from ignoring minority classes
- **CosineAnnealingLR**: learning rate anneals smoothly to near-zero, avoiding
  sharp oscillations late in training
- **Gradient clipping** (`max_norm=1.0`): prevents exploding gradients in the LSTM
- **Early stopping**: training halts automatically when validation accuracy plateaus,
  avoiding overfitting
- **Device auto-selection**: uses Apple MPS → CUDA → CPU in that priority order

---

## 8. Step 4 — Run with Trained Classifier

Once `train.py` finishes, you have `features/trick_classifier.pt` and
`features/label_map.json`. Pass both paths to `main.py`:

```bash
python main.py \
  --source      data/video.mp4 \
  --trick-model features/trick_classifier.pt \
  --label-map   features/label_map.json
```

The panel will now show **`Trick(ML): CARVE 87%`** instead of the rule-based label.
If the model is not provided, the rule-based fallback remains active.

### Full pipeline summary

```
┌─────────────────────────────────────────────────────────────┐
│  data/tricks/                                               │
│    carve/  slide/  ollie/  tuck/  ...                       │
│    (1 subfolder per trick, videos inside)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
          python extract_features.py
          --dataset data/tricks --output features/
                      │
                      ▼
          features/X.npy  y.npy  label_map.json
                      │
                      ▼
          python train.py --features features/
                      │
                      ▼
          features/trick_classifier.pt
                      │
                      ▼
          python main.py --source video.mp4
            --trick-model features/trick_classifier.pt
            --label-map   features/label_map.json
```

---

## 9. Shared Utilities (`trick_utils.py`)

This module is imported by all three other scripts. Keeping the normalisation
function and model class in one place guarantees that the data seen during
training is **identical** in format to the data seen during inference.

### Keypoint Normalisation

Raw pixel coordinates vary with camera angle, distance and person height.
Before feeding joints to the classifier, each frame is normalised so the
representation is **translation- and scale-invariant**:

1. **Translate**: subtract the hip midpoint so the hips sit at the origin (0, 0)
2. **Scale**: divide by torso length (hip midpoint → shoulder midpoint distance)
   so that a person close to or far from the camera has the same numeric range
3. **Mask low-confidence joints**: joints with confidence < 0.30 are zeroed out
4. **Flatten**: the 17 joints × (x, y, conf) become a single 51-dimensional vector

```
Raw:         (17, 2) pixel coords  +  (17,) confidence
Normalised:  (51,)  =  [x₀, y₀, c₀, x₁, y₁, c₁, ...]
```

### BiLSTM Model Architecture

```
Input:  (batch, 30, 51)          — 30-frame window, 51 features
   ↓
BiLSTM(hidden=128, layers=2)     — 256-d output (bidirectional)
   ↓
LayerNorm(256)                   — stabilises activations
   ↓
Linear(256 → 64)  +  ReLU
   ↓
Dropout(0.3)
   ↓
Linear(64 → num_classes)
   ↓
Output: (batch, num_classes)     — raw logits; softmax for probabilities
```

Bidirectional LSTMs read the sequence both forward and backward, which helps
recognise tricks where the peak motion occurs in the middle of the window.

---

## 10. Analytics & Metrics Explained

### Speed (km/h proxy)

```
speed = |ankle_y[t] − ankle_y[t−10]| × SPEED_PX_TO_KMH
```

Uses a 10-frame rolling buffer of ankle Y-position. The constant
`SPEED_PX_TO_KMH = 0.05` is a rough calibration factor — tune it for your
camera's field of view and distance.

### Knee Angle

Computed as the angle at the knee joint using the hip–knee–ankle triangle:

$$\theta = \arccos\!\left(\frac{\vec{v_1} \cdot \vec{v_2}}{\|\vec{v_1}\| \cdot \|\vec{v_2}\|}\right)$$

where $\vec{v_1}$ = (hip − knee) and $\vec{v_2}$ = (ankle − knee).
A locked straight leg ≈ 180°; a deep crouch ≈ 90°.

### Jump Height

The skateboard's bottom edge is tracked each frame. Height above ground is:

```
h_px = ground_y − skateboard_bottom_y
```

`ground_y` is automatically established from the first skateboard detection and
can be reset with the `r` key. A rolling average over 7 frames (`HEIGHT_SMOOTH_N`)
smooths out detection jitter.

### Airtime

A timer starts when `h_px > 12` (the `AIRBORNE_THRESH` constant) and stops
when the skateboard returns to ground. The session maximum is tracked and shown
alongside the current airtime.

### Jump Angle

```
θ = arctan2(Δy_ankle_peak, h_px_at_landing)
```

Captures whether the skater jumps straight up (90°) or at an angle.

### Rule-Based Trick Detection

When no trained model is loaded, tricks are detected from physical rules:

| Trick      | Condition                                                        |
| ---------- | ---------------------------------------------------------------- |
| `OLLIE`    | Airborne + height > `TRICK_OLLIE_MIN_H` (20 px)                  |
| `KICKFLIP` | Airborne + at least one knee angle < `TRICK_KICKFLIP_KNEE` (60°) |
| `GRIND`    | Very low height (< 5 px) + knee > `TRICK_GRIND_KNEE` (150°)      |
| `LANDING`  | Just transitioned from airborne to ground                        |

---

## 11. Visual Enhancements

### Phase-coloured skeleton

Bone and joint colours change automatically based on the current movement phase:

| Phase     | Colour | Meaning                         |
| --------- | ------ | ------------------------------- |
| `ground`  | Green  | Stationary on the ground        |
| `rising`  | Cyan   | Skateboard ascending            |
| `peak`    | Blue   | At or near the apex of the jump |
| `landing` | Lime   | Descending back to ground       |

### Joint motion trails

The last 25 positions of each wrist and ankle joint are drawn as a fading trail,
making rotation patterns and foot placement visible across time.

### Real-time mini graphs

Two compact line graphs are drawn next to each player's panel:

- **Speed** (km/h) — last 90 frames
- **Board height** (px) — last 90 frames

### Slo-mo highlight clips

When a trick is detected and `--highlights-dir` is set, the system automatically
saves the last `SLO_MO_BUFFER_SECS` (default 2 s) of raw frames as a 4× slow-motion
MP4 clip:

```bash
python main.py --source data/video.mp4 --highlights-dir highlights/
# saves: highlights/highlight_P001_<timestamp>.mp4
```

### Trick score banner

When a trick is scored, a large centred banner flashes on screen for 2.5 seconds
showing the trick name and points awarded.

### Achievement banners

A golden stripe appears across the top of the frame when a milestone is reached:

| Achievement     | Trigger                        |
| --------------- | ------------------------------ |
| FIRST TRICK!    | First trick of the session     |
| NEW HEIGHT PB!  | New personal best board height |
| NEW AIRTIME PB! | New personal best airtime      |
| 5 TRICK STREAK! | Five consecutive tricks        |
| CLEAN LANDING!  | Landing quality score ≥ 80/100 |

### Leaderboard

A ranked scoreboard in the bottom-left corner shows all active players sorted by
cumulative session score, updated live every frame.

---

## 12. Gamification & Scoring

### Trick Score

Each trick is scored using a composite formula:

```
trick_score = base × height_factor × airtime_factor × landing_factor × angle_bonus
```

| Component      | Description                                     |
| -------------- | ----------------------------------------------- |
| Base           | `TRICK_SCORE_BASE` (1000 pts)                   |
| Height factor  | Board height as fraction of frame height (0–2×) |
| Airtime factor | Airtime in seconds × 2 (capped at 3×)           |
| Landing factor | Landing quality score / 100 (0–1)               |
| Angle bonus    | Small multiplier for jump angles close to 90°   |

The final score is capped at 9999 per trick. Scores accumulate into `session_score`.

### Landing Quality

For `LANDING_STABLE_FRAMES` (10) frames after touch-down, knee angles are
sampled. A score of 100 indicates perfectly smooth, controlled absorption;
high variance (wobble or slamming) reduces the score.

### Stance Detection

Over a rolling window of 30 frames, the relative foot position (left vs right
ankle forward) is voted on. The majority vote determines REGULAR or GOOFY stance.

### Bail Detection

A bail is flagged when the nose keypoint drops below the hip midpoint —
indicating the skater has fallen forward. The panel row turns red.

### Coaching Hints

After each trick the system analyses the state dictionary and emits a short
text hint, for example:

- _"Bend knees more on landing"_ — when landing score is low
- _"Jump higher!"_ — when height is below average
- _"Keep that streak going!"_ — when trick_streak ≥ 3

---

## 13. Session Database (`session_db.py`)

All per-frame metrics and trick events can be saved to a SQLite database for
offline analysis and report generation.

### Enable

```bash
python main.py --source data/video.mp4 --db session.db
```

### Schema

**`frames` table** — one row per player per frame:

| Column          | Type    | Description                   |
| --------------- | ------- | ----------------------------- |
| `frame_idx`     | INTEGER | Frame number                  |
| `player_id`     | INTEGER | Track ID                      |
| `speed_kmh`     | REAL    | Estimated speed               |
| `height_px`     | REAL    | Board height in pixels        |
| `airtime_s`     | REAL    | Airtime duration              |
| `trick_label`   | TEXT    | Active trick name             |
| `session_score` | INTEGER | Cumulative score              |
| `phase`         | TEXT    | Movement phase                |
| `stance`        | TEXT    | REGULAR or GOOFY              |
| `bail`          | INTEGER | 1 if bail detected this frame |

**`tricks` table** — one row per completed trick:

| Column      | Type    | Description                |
| ----------- | ------- | -------------------------- |
| `player_id` | INTEGER | Track ID                   |
| `trick`     | TEXT    | Trick name                 |
| `score`     | INTEGER | Points awarded             |
| `airtime_s` | REAL    | Airtime for this trick     |
| `height_px` | REAL    | Peak height for this trick |
| `ts`        | REAL    | Unix timestamp             |

### Python API

```python
import session_db

conn = session_db.init_db("session.db")
session_db.log_frame(conn, frame_idx, player_id, metrics_dict)
session_db.log_trick(conn, player_id, trick, score, airtime, height)
summary = session_db.get_session_summary(conn)
conn.close()
```

---

## 14. HTML Report (`report.py`)

Generates a self-contained HTML file from a session database, with embedded
matplotlib charts and a per-player trick breakdown table.

### Usage

```bash
python report.py --db session.db --output report.html
```

Or from Python:

```python
from report import generate_report
generate_report("session.db", "report.html")
```

### Report contents

- Per-player **stat cards**: max speed, max height, max airtime, final score, trick variety, bails
- **Speed over time** chart (matplotlib, embedded as base64 PNG)
- **Board height over time** chart
- **Trick frequency** horizontal bar chart
- **Trick breakdown table**: name · count · best score

The report is a single `.html` file with no external dependencies — open it in
any browser or share it directly.

---

## 15. Web Dashboard (`app.py`)

A Flask web application that streams the annotated video live and displays
real-time player stats in a browser.

### Start the dashboard

```bash
# Terminal 1 — run the analysis
python main.py --source data/video.mp4 --db session.db

# Terminal 2 — start the dashboard
python app.py --port 5000
```

Then open **http://127.0.0.1:5000/** in a browser.

> `push_frame()` and `push_stats()` in `app.py` are the integration points —
> call them from `process_video` when you want live streaming.

### Endpoints

| URL                     | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| `GET /`                 | Live dashboard — MJPEG stream + auto-refreshing stat cards |
| `GET /stream`           | Raw MJPEG video stream (embed in `<img src="/stream">`)    |
| `GET /api/stats`        | JSON snapshot of all current player stats                  |
| `GET /api/leaderboard`  | JSON ranked leaderboard                                    |
| `GET /sessions`         | Browse `.db` files in the working directory                |
| `GET /report/<name.db>` | Generate and serve an HTML report on demand                |

### Install Flask

```bash
pip install flask
```

---

## 16. Dataset Preparation

Organise your videos so that each trick type has its own subfolder:

```
data/tricks/
├── carve/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
├── slide/
│   └── ...
├── ollie/
│   └── ...
└── tuck/
    └── ...
```

**Rules:**

- The **folder name** is the class label — it must be consistent
- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Videos can be any length; short clips (2–10 seconds) work well
- Aim for **at least 50–100 clips per class** for reliable training
- The more variation in camera angle, lighting and skater physique, the better

**Tips for better accuracy:**

- Trim clips so each video contains **one trick only** (no long idle sections)
- Include different skaters performing the same trick to avoid person-specific overfitting
- Balance class sizes where possible (equal number of clips per trick)

---

## 17. Configuration & Tuning

Key constants at the top of `main.py` that can be adjusted without retraining:

```python
KP_CONF_THRESH   = 0.40   # min joint confidence to draw / use — raise to reduce noise
HEIGHT_SMOOTH_N  = 7      # rolling average window for skateboard height
AIRBORNE_THRESH  = 12     # px above ground to count as airborne
SPEED_PX_TO_KMH  = 0.05   # calibrate to your camera distance and FOV

# Rule-based trick thresholds
TRICK_OLLIE_MIN_H   = 20   # px
TRICK_KICKFLIP_KNEE = 60   # degrees (max during kickflip)
TRICK_GRIND_KNEE    = 150  # degrees (min for grind)

# Visual / gamification
TRAIL_LENGTH          = 25    # frames of joint motion trail to keep
GRAPH_WIDTH           = 200   # px — width of mini speed/height graphs
GRAPH_HEIGHT          = 60    # px — height of mini speed/height graphs
GRAPH_HISTORY         = 90    # frames of history shown in graphs
SLO_MO_BUFFER_SECS    = 2.0   # seconds of raw frames kept for highlight exports
HIGHLIGHT_CONF_THRESH = 0.70  # min ML confidence to trigger a highlight save
LANDING_STABLE_FRAMES = 10    # frames after landing to measure knee quality
TRICK_SCORE_BASE      = 1000  # base points that get multiplied per trick
BADGE_DISPLAY_SECS    = 3.0   # seconds an achievement badge stays on screen
```

Constants in `trick_utils.py` that **must stay the same** between extraction and inference:

```python
WINDOW_SIZE = 30    # frames per input sample — change only if you re-extract + retrain
FEATURE_DIM = 51    # always 17 × 3  (do not change)
MIN_KP_CONF = 0.30  # joints below this are zeroed out
```

---

## 18. Output Files

### Annotated video (`--save`)

An `.mp4` copy of the input video with all overlays burned in.

### CSV data log (`--log-csv`)

Written as `<output_name>_data.csv`. One row per player per frame:

| Column                 | Description                             |
| ---------------------- | --------------------------------------- |
| `frame`                | Frame number                            |
| `player_id`            | Player track ID (e.g. `P001`)           |
| `conf`                 | YOLOv8 detection confidence             |
| `ankle_y_px`           | Average ankle Y position                |
| `knee_angle_left_deg`  | Left knee angle in degrees              |
| `knee_angle_right_deg` | Right knee angle in degrees             |
| `speed_kmh`            | Estimated speed                         |
| `skate_height_px`      | Skateboard height above ground (pixels) |
| `skate_height_pct`     | Height as % of frame height             |
| `skate_max_height_px`  | Session maximum height                  |
| `airtime_s`            | Current airtime (if in air)             |
| `jump_angle_deg`       | Estimated jump trajectory angle         |
| `trick`                | Rule-based trick label for this frame   |

### Training outputs (in `features/`)

| File                   | Description                                   |
| ---------------------- | --------------------------------------------- |
| `trick_classifier.pt`  | Best model weights — use with `--trick-model` |
| `training_curves.png`  | Loss + accuracy curves across epochs          |
| `confusion_matrix.png` | Per-class accuracy heatmap                    |
| `training_report.txt`  | Precision / recall / F1 for every trick class |

### Session database (`--db`)

A SQLite file (e.g. `session.db`) containing `frames` and `tricks` tables.
See [§13](#13-session-database-session_dbpy) for the full schema.

### Highlight clips (`--highlights-dir`)

Per-player 4× slo-mo MP4 files saved when a trick is detected:

```
highlights/highlight_P001_<unix_timestamp>.mp4
```

### HTML report

Run `python report.py --db session.db` to generate `report.html` —
a self-contained browser-viewable summary. See [§14](#14-html-report-reportpy).

---

## 19. Key Design Decisions

**Why YOLOv8-Pose instead of MediaPipe?**
YOLOv8 runs both person detection and pose estimation in a single pass, supports
`track(persist=True)` for stable multi-person IDs across frames (ByteTrack), and
the same ultralytics package handles skateboard detection.

**Why BiLSTM instead of a simple frame classifier?**
A single frame cannot distinguish an ollie takeoff from an ollie peak from a
landing — the temporal sequence is the trick. A BiLSTM reads the 30-frame window
both forward and backward, capturing the full motion arc.

**Why normalise keypoints?**
Without normalisation the model would learn camera-specific pixel positions rather
than body movement patterns. Hip-centred, torso-scaled coordinates let the same
model work on different cameras, distances and player heights.

**Why rule-based fallback?**
The trained classifier requires `--trick-model` and a trained dataset. The
rule-based fallback works on any skateboard video out of the box, without needing
to label and train on custom data first.

**Why a sliding window instead of full-sequence input?**
Sliding windows of fixed length allow the same architecture to process videos of
any duration. A 30-frame window at 30 fps covers 1 second — enough to capture
most skateboard tricks.

**Why phase-coloured skeleton instead of a single colour?**
Colour-coding per phase (ground/rising/peak/landing) gives an immediate visual
cue about where in the trick arc the skater is, without needing to read the panel.

**Why landing quality instead of just airtime?**
A high bail-risk landing (wobble, bent knee spike) is penalised in the score.
This encourages technically clean tricks over just jumping high.

**Why SQLite instead of CSV for the session DB?**
SQLite allows efficient aggregate queries (max speed, trick frequency) without
loading the entire log into memory, and the same file feeds the HTML report and
Flask dashboard with a single `session_db.get_session_summary()` call.

**Why Flask for the dashboard instead of a desktop GUI?**
Flask runs on any device on the same network — a tablet or second monitor can
display the leaderboard and stats without any additional software installation.
