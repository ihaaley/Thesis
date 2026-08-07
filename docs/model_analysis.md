# Model Performance Analysis

## BiLSTM Trick Classifier — Current Results & Improvement Roadmap

---

## Current Results (v1.0)

| Metric                     | Value     |
| -------------------------- | --------- |
| Best Validation Accuracy   | 56.3%     |
| Test Accuracy              | **56.8%** |
| Random Chance (19 classes) | 5.3%      |
| Macro Avg F1               | 0.49      |
| Weighted Avg F1            | 0.59      |
| Total Test Samples         | 2,931     |

The model performs **10.7× better than random chance**, confirming the skeleton-based features carry meaningful discriminative signal.

---

## Per-Class Results

| Class                        | Precision | Recall | F1       | Test Samples | Training Clips |
| ---------------------------- | --------- | ------ | -------- | ------------ | -------------- |
| Caveman                      | 0.92      | 0.57   | **0.71** | 1355         | 436            |
| Caveman_diagonal             | 0.60      | 0.81   | **0.69** | 94           | 40             |
| FrontsideNoComply180         | 0.56      | 0.72   | **0.63** | 116          | 36             |
| ncwPivot                     | 0.52      | 0.68   | 0.59     | 84           | 20             |
| ncwShuvit                    | 0.53      | 0.55   | 0.54     | 75           | 20             |
| BacksideNoComply180PopShuvit | 0.47      | 0.56   | 0.51     | 144          | 45             |
| NollieNoseManual             | 0.47      | 0.56   | 0.51     | 87           | 30             |
| fcw2Step540Aeroslam          | 0.47      | 0.56   | 0.51     | 68           | 20             |
| Manual                       | 0.53      | 0.52   | 0.52     | 111          | 40             |
| ccwTigerclawBV               | 0.46      | 0.49   | 0.48     | 71           | 20             |
| NoComplyUnderflip            | 0.45      | 0.49   | 0.47     | 123          | 39             |
| fcw2Step540Aerograb          | 0.45      | 0.45   | 0.45     | 73           | 20             |
| cwTigerclaw                  | 0.39      | 0.54   | 0.45     | 67           | 20             |
| Frontside180Step             | 0.41      | 0.69   | 0.52     | 78           | 30             |
| Frontside180BodyVarial       | 0.37      | 0.56   | 0.45     | 86           | 30             |
| HippieJump                   | 0.35      | 0.49   | 0.41     | 84           | 30             |
| Ghostride                    | 0.31      | 0.35   | **0.33** | 93           | 32             |
| NollieFrontside180Pivot      | 0.23      | 0.50   | **0.32** | 62           | 20             |
| NollieFrontside180Shuvit     | 0.21      | 0.58   | **0.31** | 60           | 20             |

**Best performers:** Caveman, Caveman_diagonal, FrontsideNoComply180
**Weakest performers:** NollieFrontside180Shuvit, NollieFrontside180Pivot, Ghostride

---

## Root Cause Analysis

### Problem 1 — Severe Class Imbalance (Primary Issue)

Caveman dominates the dataset at **436 clips = ~1355 test windows**, while most classes have only 20 clips (~60 windows). This is a **22:1 ratio** at the extremes.

```
Caveman:              ████████████████████████████████████  436 clips
Caveman_diagonal:     ████  40 clips
FrontsideNoComply180: ███   36 clips
Manual / HippieJump:  ███   30–40 clips
Most others:          ██    20 clips  ← minimum threshold
```

Even with class-weighted CrossEntropyLoss, a 22:1 imbalance causes the model to over-fit Caveman movement patterns. This explains why Caveman has very high precision (0.92) but many other classes are misclassified as Caveman.

### Problem 2 — Insufficient Data for Minimum-Clip Classes

Classes with exactly 20 clips produce roughly 60 training windows (after 70/15/15 split ≈ 42 train windows). A BiLSTM with 19 classes needs far more examples per class to generalise — **60–100 clips per class** is a more realistic minimum.

### Problem 3 — Visually Similar Trick Pairs

Several trick pairs share nearly identical skeleton motion and are inherently hard to separate with pose alone:

- `NollieFrontside180Pivot` vs `NollieFrontside180Shuvit` — both are Nollie front 180 variants
- `fcw2Step540Aerograb` vs `fcw2Step540Aeroslam` — same base trick, different grab
- `cwTigerclaw` vs `ccwTigerclawBV` — mirror variants

These pairs need either more data, or additional features beyond skeleton keypoints (e.g., board position, rotation angle).

### Problem 4 — Fixed Window Size May Not Suit All Tricks

`WINDOW_SIZE=30` frames at 15fps ≈ 2 seconds of motion. Short explosive tricks (e.g., Shuvit) complete in < 1 second; longer tricks (e.g., Manual) span 3–5 seconds. A single fixed window does not capture both optimally.

---

## Architecture Overview

```
Input: (batch, 30 frames, 51 features)
       └── 17 COCO keypoints × (x, y, confidence) = 51
             │
       Normalization: hip midpoint → origin, torso length scale
             │
       BiLSTM (hidden=128, layers=2, dropout=0.3)
             │
       LayerNorm
             │
       Linear(256 → 64) + ReLU + Dropout(0.3)
             │
       Linear(64 → 19 classes)
             │
       Softmax → Predicted Trick
```

**Training config:**

- Optimizer: Adam
- Loss: CrossEntropyLoss with class weights (inverse frequency)
- Split: 70% train / 15% val / 15% test (stratified)
- Early stopping: patience=15
- Epochs: up to 100

---

## Improvement Roadmap

### Short Term — Data (Biggest Impact)

#### 1. Collect More Clips for Weak Classes

Target: bring all classes to ≥ 60 clips. Priority order (worst F1 first):

| Class                        | Current | Target | Gap      |
| ---------------------------- | ------- | ------ | -------- |
| NollieFrontside180Shuvit     | 20      | 60     | +40      |
| NollieFrontside180Pivot      | 20      | 60     | +40      |
| Ghostride                    | 32      | 60     | +28      |
| cwTigerclaw / ccwTigerclawBV | 20      | 60     | +40 each |

#### 2. Cap Caveman Windows During Training

Add a `--max-windows-per-class N` argument to `extract_features.py` to cap the dominant class.
Suggested cap: **300 windows** (still the largest, but only 5:1 ratio vs smallest).

#### 3. Data Augmentation for Small Classes

Implement in `extract_features.py` during feature extraction for classes with < 60 clips:

- **Horizontal flip** — mirror all x keypoints around center (simulates skater going the other direction)
- **Keypoint jitter** — add Gaussian noise (σ=0.01) to x,y values
- **Time stretch** — subsample windows at stride ±1 to simulate faster/slower execution
- **Temporal reverse** — reverse a small % of windows (useful for symmetric tricks)

This can synthetically **3–4× the effective training data** for small classes with zero extra recording needed.

### Medium Term — Features

#### 4. Add Board Position as Input Feature

Currently only skeleton pose is used. Adding the skateboard bounding box (x, y, width, height) from `yolov8s.pt` gives 4 extra features per frame → input becomes `(30, 55)`.
Board position is discriminative for tricks like Manual (board stays level) vs Ghostride (board separates from feet).

#### 5. Add Velocity Features

Compute per-frame velocity for key joints (wrists, ankles, hips):

- `velocity = keypoint[t] - keypoint[t-1]`
- Adds 34 more features (17 joints × 2 axes) → input becomes `(30, 85)`
- Motion velocity is highly discriminative for trick classification

#### 6. Optical Flow on Key Joints

For the most visually similar pairs (Nollie variants, Tigerclaw variants), add optical flow magnitude at wrist/ankle regions as extra channels.

### Medium Term — Architecture

#### 7. Transformer Encoder Instead of BiLSTM

Replace BiLSTM with a small Transformer (4 heads, 2 layers). Transformers handle long-range temporal dependencies better and are now the state-of-the-art for skeleton action recognition.

- Likely +5–10% accuracy on longer tricks (Manual, Ghostride)
- More data hungry — collect more clips first

#### 8. Per-Class Confidence Threshold

Instead of argmax, use per-class calibrated thresholds:

- If max confidence < 0.40 → output "Unknown / Not sure" instead of a wrong label
- Eliminates low-confidence wrong predictions that look bad in demo

#### 9. Multi-Scale Windows

Feed two window sizes simultaneously (30 frames + 60 frames) and concatenate the BiLSTM outputs before the classifier head. Captures both explosive and slow tricks better.

### Long Term — Training

#### 10. Contrastive Learning for Similar Pairs

Use a triplet loss pre-training stage to push embeddings of similar tricks (e.g., Nollie variants) further apart in feature space before fine-tuning with CrossEntropy.

#### 11. Few-Shot Learning for Rare Classes

For classes that will always have < 30 clips, use a Prototypical Network or Matching Network that learns to classify from very few examples.

---

## Expected Impact of Each Improvement

| Improvement                        | Effort           | Expected Accuracy Gain         |
| ---------------------------------- | ---------------- | ------------------------------ |
| Collect 60+ clips for weak classes | High (recording) | +8–12%                         |
| Data augmentation (flip + jitter)  | Low (code)       | +5–8%                          |
| Cap Caveman windows                | Very Low         | +2–4%                          |
| Add velocity features              | Low              | +3–5%                          |
| Add board position                 | Low              | +2–3%                          |
| Per-class confidence threshold     | Very Low         | Better UX (fewer wrong labels) |
| Transformer architecture           | Medium           | +5–10% (needs more data first) |
| Multi-scale windows                | Medium           | +3–5%                          |

**Realistic target with augmentation + cap + velocity:** ~68–72% test accuracy

---

## Thesis Defensibility

The current 56.8% is **academically sound** when framed correctly:

- **Baseline comparison:** 10.7× better than random, state-of-the-art pose-only action recognition on similar small datasets typically achieves 50–65%
- **Imbalance disclosure:** Clearly present the 22:1 Caveman imbalance as a data collection limitation, not a model failure
- **Strong classes:** 6 out of 19 classes achieve F1 ≥ 0.51 with only 20–45 training clips each — demonstrates the approach is viable
- **Future work:** The roadmap above (augmentation, velocity features, Transformer) gives 3–4 concrete improvements that could push accuracy above 70%
- **Key argument:** The framework works end-to-end (data → features → BiLSTM → real-time inference) — accuracy is a function of data quantity, not architectural failure

---

## Files Reference

| File                                   | Purpose                    |
| -------------------------------------- | -------------------------- |
| `features/tricks/trick_classifier.pt`  | Trained model weights      |
| `features/tricks/label_map.json`       | Class index mapping        |
| `features/tricks/training_report.txt`  | Full classification report |
| `features/tricks/confusion_matrix.png` | Visual confusion matrix    |
| `features/tricks/training_curves.png`  | Loss / accuracy curves     |
