# Project Description — Longboard Trick Recognition System

## What This Project Does

This project is a **real-time longboard/skateboard trick recognition system** built for a thesis.  
It uses computer vision and deep learning to:

1. Watch a video feed (live camera or file)
2. Detect and track each rider's body skeleton
3. Classify what trick they are performing (e.g. Caveman, Manual, HippieJump)
4. Display live stats per player — speed, airtime, trick label, confidence
5. Record sessions and serve a web dashboard with a leaderboard

---

## System Architecture Overview

```             Video Input
                     │
                     ▼
    ┌─────────────────────────────────┐
    │  YOLOv8s-pose  (main.py)        │  → detects person + 17 body keypoints
    │  YOLOv8s       (main.py)        │  → detects skateboard bounding box
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │  Keypoint Normalization         │  → removes camera/position/size variance
    │  (trick_utils.py)               │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │  Sliding Window Buffer (30 fr.) │  → collects 30 frames of normalized pose
    │  (main.py)                      │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │  BiLSTM Trick Classifier        │  → outputs: trick label + confidence
    │  (trick_utils.py / train.py)    │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │  Flask Web Dashboard (app.py)   │  → live stream, leaderboard, stats API
    │  SQLite Session Log (session_db)│  → stores per-session trick history
    │  HTML Report (report.py)        │  → exportable session reports
    └─────────────────────────────────┘
```

---

## Step-by-Step Pipeline

### Step 1 — Data Collection (`data/tricks/`)

**What:** Labeled video clips of longboard tricks organized into class folders.  
**Purpose:** Provide supervised training data. Each folder name = one trick class.  
**How:** `setup_dataset.py` copies clips from the external drive into `data/tricks/<ClassName>/`.  
**Output:** 107 class folders, ~1522 video clips total. 19 classes have ≥20 clips (usable for training).

---

### Step 2 — Feature Extraction (`extract_features.py`)

**What:** Processes every video clip and converts it into numerical feature arrays.  
**Purpose:** The neural network cannot take raw video — it needs structured numerical input.

**How it works:**

1. Opens each video clip frame by frame
2. Runs **YOLOv8s-pose** on every Nth frame (`--frame-skip 2`) to get 17 body keypoints (COCO skeleton)
3. **Normalizes** keypoints using `trick_utils.normalize_keypoints()`:
   - Translates hip midpoint to origin → removes camera position
   - Scales by torso length → removes person size/distance from camera
4. Stacks 30 consecutive normalized frames into a **sliding window** (stride 15 → 50% overlap)
5. Each window becomes one training sample: shape `(30, 51)` where 51 = 17 joints × (x, y, confidence)
6. Saves per-class cache files to `features/tricks/cache/<ClassName>.npz`
7. After all classes done, merges into `X.npy` (shape: N×30×51) and `y.npy` (integer class labels)

**Output:** `features/tricks/X.npy`, `y.npy`, `label_map.json`

---

### Step 3 — Model Training (`train.py`)

**What:** Trains a BiLSTM neural network to classify tricks from pose sequences.  
**Purpose:** Learn the temporal pattern of body movement that distinguishes one trick from another.

**How it works:**

1. Loads `X.npy` and `y.npy`
2. Splits data **70% train / 15% validation / 15% test** (stratified by class)
3. Applies **class-weighted CrossEntropyLoss** to handle imbalance (Caveman has 436 clips vs 20 for others)
4. Architecture (`TrickClassifier` in `trick_utils.py`):
   - Input: `(batch, 30, 51)`
   - **BiLSTM** (2 layers, 128 hidden units, bidirectional) → captures motion forward and backward in time
   - **LayerNorm** → stabilizes training
   - **Linear(256 → 64)** → ReLU
   - **Linear(64 → num_classes)** → trick prediction
5. Trains with **early stopping** (stops if validation accuracy doesn't improve for N epochs)
6. Saves the best model by validation accuracy

**Output:** `trick_classifier.pt`, `training_curves.png`, `confusion_matrix.png`, `training_report.txt`

---

### Step 4 — Real-Time Inference (`main.py`)

**What:** Runs the full pipeline on live video or a file.  
**Purpose:** The actual product — recognize tricks in real time.

**How it works:**

1. Opens webcam or video file
2. Per frame:
   - YOLOv8s-pose → detect riders → extract 17 keypoints each
   - YOLOv8s → detect skateboard → estimate height above ground, compute speed
   - Normalize keypoints (same as training — critical for consistency)
   - Append to per-player sliding window buffer (30 frames)
   - Once buffer is full, run `TrickClassifier` → get trick label + confidence
3. Computes per-player stats: speed (km/h), airtime (seconds), jump height (px), trick score
4. Draws bounding boxes, skeleton overlay, speed graph, trick label on screen
5. Saves highlight clips when confidence > threshold
6. Logs everything to SQLite via `session_db.py`

---

### Step 5 — Web Dashboard (`app.py` + `session_db.py` + `report.py`)

**What:** A Flask web server providing a live dashboard.  
**Purpose:** Display real-time stats, leaderboard, and session history in a browser.

**Endpoints:**

- `/` — leaderboard + live stats HTML page
- `/stream` — MJPEG live video stream
- `/api/stats` — JSON snapshot of all player stats
- `/api/leaderboard` — JSON ranked leaderboard
- `/sessions` — list past session databases
- `/report/<name>` — generate and serve an HTML session report

---

## What the Project Produces

By the end of the pipeline the system can:

- **Recognize 19 longboard trick classes** in real time from a standard camera
- **Track multiple riders simultaneously**, each with independent trick history
- **Display live overlays**: skeleton, trick label, confidence, speed, airtime
- **Rank riders** on a leaderboard by trick score
- **Export session reports** as HTML documents
- **Save highlight clips** automatically when a high-confidence trick is detected

The core insight is that a trick is a **temporal pattern of body pose** — the BiLSTM learns to recognize these patterns by seeing hundreds of labeled examples.

---

## How to Make This Production-Ready

### 1. Model Quality

| Issue                                              | Fix                                                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Only 19 classes, some with just 20 clips           | Collect 100–500 clips per class. Quality > quantity.                                             |
| Heavy class imbalance (Caveman: 436 vs others: 20) | Augment under-represented classes: horizontal flip, speed variation, brightness jitter           |
| No data augmentation during training               | Add random noise, time-warp, and dropout augmentation to `extract_features.py`                   |
| Single train/test split                            | Use k-fold cross-validation for more reliable accuracy estimates                                 |
| Fixed window size (30 frames)                      | Tricks have different durations — consider variable-length input with attention or a Transformer |

### 2. Inference Performance

| Issue                                  | Fix                                                                               |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| All models run sequentially on CPU/MPS | Run pose and object detection in parallel threads                                 |
| YOLOv8s is relatively large            | Export to ONNX or CoreML for 3–5× faster inference: `model.export(format="onnx")` |
| No frame-level batching                | Process frames in batches of 4–8 for GPU throughput                               |
| Sliding window runs every frame        | Only run classifier when buffer has changed by ≥ stride frames                    |

### 3. System Reliability

| Issue                                              | Fix                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| No error recovery on model load failure            | Wrap model loading in try/except with graceful fallback             |
| SQLite is single-writer, unsuitable for multi-user | Switch to PostgreSQL for concurrent session writes                  |
| No authentication on Flask dashboard               | Add JWT or session-based auth before any public deployment          |
| Highlight clips saved to local disk                | Upload to object storage (S3 / Azure Blob) with a background thread |

### 4. Deployment

| Step              | Recommendation                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| Containerize      | Dockerfile with CUDA or MPS base image; pin all dependency versions                            |
| Model versioning  | Store model artifacts in MLflow or DVC with metadata (accuracy, training date, classes)        |
| Config management | Move all constants (thresholds, paths, window sizes) to a single `config.yaml`                 |
| CI/CD             | On every new model trained, auto-run `train.py` eval and gate deployment on accuracy threshold |
| Monitoring        | Log prediction confidence distribution over time — drift indicates the model needs retraining  |
| Multi-camera      | Abstract the video source behind an interface; support RTSP streams for IP cameras             |

### 5. User Experience

| Issue                                | Fix                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------- |
| Flask dashboard is basic HTML        | Build a React or Vue frontend consuming the `/api/` endpoints          |
| No mobile support                    | Responsive layout + PWA for tablet/phone use at skate parks            |
| Trick labels are internal code names | Map to human-readable display names in `label_map.json`                |
| No replay                            | Store last N seconds in a circular buffer; expose a `/replay` endpoint |

---

## Summary

This project is a complete end-to-end computer vision pipeline:  
**raw video → pose extraction → feature engineering → deep learning → real-time classification → web dashboard**.

The thesis demonstrates that skeleton-based pose sequences, when properly normalized and fed into a BiLSTM, can reliably distinguish longboard tricks without needing raw pixel data — making the approach robust to clothing, lighting, and camera angle changes.
