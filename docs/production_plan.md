# Production-Ready UI Improvement Plan

## Local Device Only — No External APIs

All changes are self-contained. Run everything with `python3 main.py` or `python3 app.py`.

---

## Phase 1 — Real-Time Overlay Improvements (main.py)

_Visual upgrades to the OpenCV camera/video window_

### Step 1.1 — Trick Name Pop Animation

**What:** When a trick is detected with confidence ≥ 0.60, flash the trick name large in the center of the frame for ~40 frames, then fade out.
**How:**

- Keep a `trick_flash` state variable: `{name, confidence, frames_left}`
- Each frame: if `frames_left > 0`, draw the trick name centered with `cv2.putText` at large font size, decrease counter
- Color: white text with a semi-transparent black background rectangle behind it

### Step 1.2 — Top-3 Confidence Bar Panel

**What:** A panel in the bottom-left corner showing the top 3 predicted tricks with horizontal confidence bars.
**How:**

- After BiLSTM inference, `torch.topk(logits, 3)` → get top 3 class names + probabilities
- Draw a dark semi-transparent background rectangle (bottom-left, ~280×90px)
- For each of the 3 entries: draw trick name as text + a filled rectangle scaled to confidence %
- Color code: green > 0.70, yellow 0.40–0.70, red < 0.40

### Step 1.3 — Skeleton Color by Trick Class

**What:** Each trick class gets its own unique skeleton color instead of always green.
**How:**

- Build a `TRICK_COLORS` dict at startup: hash each class name to a BGR color from a fixed palette of 19 distinct colors
- When drawing the skeleton, use `TRICK_COLORS[current_trick]` instead of the fixed green

### Step 1.4 — Joint Motion Trails

**What:** Fading trails behind wrists and ankles showing the last 20 positions — visualizes trick trajectory.
**How:**

- Keep a `deque(maxlen=20)` for each of the 4 joints (left wrist, right wrist, left ankle, right ankle)
- Each frame: append current joint (x, y) to its deque
- Draw circles along the trail with decreasing radius and alpha to simulate fade

### Step 1.5 — Splash Screen on Startup

**What:** Show model info for 2 seconds before the camera feed starts.
**How:**

- Create a black frame (same resolution as camera)
- Draw: project name (top), model file name, number of classes, device (MPS/CPU), date
- Display for 60 frames (at 30fps ≈ 2 seconds), then switch to live feed

### Step 1.6 — Rep Counter HUD

**What:** Corner display showing how many times each trick was detected this session.
**How:**

- Keep a `session_counts: dict[str, int]` — increment each time a trick is confirmed
- Draw in top-right: show top 5 most frequent tricks with counts
- Reset on session restart

---

## Phase 2 — Web Dashboard Improvements (app.py + templates)

_Upgrade the Flask dashboard — no public hosting, only `localhost:5000`_

### Step 2.1 — Dark Theme + Clean Layout

**What:** Replace default HTML with a dark-themed professional layout.
**How:**

- Add a `static/style.css` with dark background (#0f1117), card components, monospace font for stats
- Update the base HTML template to use this stylesheet
- Use a consistent color palette matching the overlay (green/yellow/red for confidence)

### Step 2.2 — Per-Trick Session Timeline

**What:** Horizontal timeline bar showing when each trick was detected during the session.
**How:**

- Store trick detections in SQLite with timestamp (already done via `session_db.py`)
- On the session detail page, render a horizontal `<div>` timeline using CSS widths proportional to session duration
- Each trick event = a colored tick mark with hover tooltip showing trick name + confidence

### Step 2.3 — Trick Frequency Chart

**What:** Bar chart showing which tricks were detected most in a session.
**How:**

- Query SQLite for trick counts per session
- Render as pure CSS bar chart (no external library needed) or use Chart.js served locally
- Download Chart.js once (`static/chart.min.js`) — no CDN needed

### Step 2.4 — Highlight Reel Gallery

**What:** Grid of auto-saved highlight video clips with trick label, confidence, and timestamp.
**How:**

- `--highlights-dir clips/` already saves `.mp4` clips when a trick fires above threshold
- Add a `/highlights` route in `app.py` that lists all `.mp4` files in the highlights dir
- Render as a responsive grid: `<video>` thumbnail + trick name + confidence badge

### Step 2.5 — Confusion Matrix Viewer

**What:** Display the training confusion matrix as an interactive HTML table in the dashboard.
**How:**

- `train.py` already saves `confusion_matrix.png` — show it as an `<img>` on a "Model Info" page
- Also parse `training_report.txt` to extract per-class accuracy and render as a sortable HTML table

### Step 2.6 — Model Info Page

**What:** A dedicated `/model` page showing everything about the trained model.
**How:**

- Read `label_map.json` → list all 19 classes
- Read `training_report.txt` → overall accuracy, per-class F1
- Show `training_curves.png` and `confusion_matrix.png` inline
- Show file size of `trick_classifier.pt`, training date (file mtime)

---

## Phase 3 — Demo Mode (main.py --demo)

_For thesis presentations — no live camera needed_

### Step 3.1 — `--demo` Flag

**What:** Loop through all clips in `data/tricks/` automatically, showing ground-truth label vs predicted label side by side.
**How:**

- Add `--demo` argument to `main.py`
- In demo mode: walk `data/tricks/<ClassName>/` folders, feed each clip through the pipeline
- Overlay: left side = "Ground Truth: Caveman", right side = "Predicted: Caveman (87%)"
- Color the predicted label green if correct, red if wrong

### Step 3.2 — Per-Clip Accuracy Summary

**What:** At the end of a demo run, print (and optionally save) a summary: X/Y clips correct per class.
**How:**

- Accumulate correct/total counts per class in a dict
- At end: print a formatted table to terminal + save as `demo_results.txt`

---

## Phase 4 — Launch Script

_Single command to start everything_

### Step 4.1 — `start.sh`

**What:** One script that activates the venv, starts the Flask dashboard in background, then launches the camera.

```bash
#!/bin/bash
source venv/bin/activate
python3 app.py &           # dashboard at localhost:5000
open http://localhost:5000  # open browser automatically
python3 main.py --source 0  # start camera (foreground)
```

---

## Recommended Implementation Order

| Priority | Step                            | Impact                            | Effort   |
| -------- | ------------------------------- | --------------------------------- | -------- |
| 1        | Step 1.1 — Trick pop animation  | High visual impact for demo       | Low      |
| 2        | Step 1.2 — Confidence bar panel | Very informative                  | Low      |
| 3        | Step 3.1 — Demo mode            | Essential for thesis presentation | Medium   |
| 4        | Step 2.1 — Dark theme dashboard | Looks professional                | Low      |
| 5        | Step 2.4 — Highlight gallery    | Shows best moments                | Medium   |
| 6        | Step 1.4 — Motion trails        | Cool visual                       | Low      |
| 7        | Step 2.2 — Session timeline     | Informative                       | Medium   |
| 8        | Step 2.6 — Model info page      | Academic credibility              | Low      |
| 9        | Step 1.3 — Skeleton by trick    | Subtle but nice                   | Low      |
| 10       | Step 4.1 — Launch script        | Convenience                       | Very Low |

---

## Notes

- All changes are additive — nothing in the existing ML pipeline (`extract_features.py`, `train.py`, `trick_utils.py`) needs to change
- Chart.js can be downloaded once: `curl -o static/chart.min.js https://cdn.jsdelivr.net/npm/chart.js/dist/chart.min.js`
- Demo mode is the most important for a thesis defense — it lets you show the model working without needing a live skater
- Implement Phase 1 first (overlay) since it has the most visual impact and lowest risk
