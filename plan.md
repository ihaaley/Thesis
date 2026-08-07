# Feature Extraction & Training Run Plan

## Setup (once)

```bash
source venv/bin/activate
```

---

## Feature Extraction — 19 Classes (≥20 clips each)

Run each command one at a time. Wait for it to finish before running the next.
Completed classes are auto-skipped on re-run (cached in `features/tricks/cache/`).

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 1 / 19 — BacksideNoComply180PopShuvit** (45 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes BacksideNoComply180PopShuvit -->
<!-- ``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 2 / 19 — Caveman** (436 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes Caveman
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 3 / 19 — Caveman_diagonal** (40 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes Caveman_diagonal
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 4 / 19 — Frontside180BodyVarial** (30 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes Frontside180BodyVarial
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 5 / 19 — Frontside180Step** (30 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes Frontside180Step
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 6 / 19 — FrontsideNoComply180** (36 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes FrontsideNoComply180
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 7 / 19 — Ghostride** (32 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes Ghostride
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 8 / 19 — HippieJump** (30 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes HippieJump
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 9 / 19 — Manual** (40 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes Manual
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 10 / 19 — NoComplyUnderflip** (39 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes NoComplyUnderflip
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 11 / 19 — NollieFrontside180Pivot** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes NollieFrontside180Pivot
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 12 / 19 — NollieFrontside180Shuvit** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes NollieFrontside180Shuvit
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 13 / 19 — NollieNoseManual** (30 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes NollieNoseManual
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 14 / 19 — ccwTigerclawBV** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes ccwTigerclawBV
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 15 / 19 — cwTigerclaw** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes cwTigerclaw
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 16 / 19 — fcw2Step540Aerograb** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes fcw2Step540Aerograb
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 17 / 19 — fcw2Step540Aeroslam** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes fcw2Step540Aeroslam
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 18 / 19 — ncwPivot** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes ncwPivot
``` -->

<!-- ----------------- ∆ DONE ∆ -----------------
**Class 19 / 19 — ncwShuvit** (20 clips)

```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --min-clips 20 --classes ncwShuvit
``` -->

---

## Merge All Cached Classes

Run after all 19 classes are done:

<!-- ----------------- ∆ DONE ∆ -----------------
```bash
python3 extract_features.py --dataset data/tricks --output features/tricks --merge-only
``` -->

<!-- Output: `features/tricks/X.npy` + `features/tricks/y.npy` -->

---

## Train the Model

<!-- ----------------- ∆ DONE ∆ -----------------
```bash
python3 train.py --features features/tricks --epochs 100 --batch 32 --patience 15
``` -->

Output:

- `features/tricks/trick_classifier.pt` — trained model
- `features/tricks/label_map.json` — class index map
- `features/tricks/training_curves.png`
- `features/tricks/confusion_matrix.png`
- `features/tricks/training_report.txt`

---

## Notes

- 88 other folders exist with only 1–19 clips — excluded from training (`--min-clips 20`)
- To lower threshold: change `--min-clips 20` to `--min-clips 10` (adds ~11 more classes)
- Check cache progress: `ls features/tricks/cache/`
- If a run is killed, just rerun the same command — completed classes are auto-skipped
- `main.py` defaults are already set to `features/tricks/trick_classifier.pt` and `features/tricks/label_map.json`
