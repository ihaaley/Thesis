"""
extract_features.py
-------------------
SCHRITT 2 der Pipeline: Verwandelt beschriftete Videoclips in Zahlen, mit denen
ein neuronales Netz trainiert werden kann.

GRUNDIDEE:
Ein Video ist für ein LSTM zu groß und zu unspezifisch. Deshalb wird pro Bild
nur die KÖRPERHALTUNG extrahiert (17 Gelenkpunkte via YOLOv8-Pose) und diese
Punkte werden normalisiert. Aus der langen Folge solcher Haltungen werden dann
kurze, überlappende Ausschnitte ("Fenster") von je 30 Bildern geschnitten.
Jedes Fenster ist ein Trainingsbeispiel.

Erwartete Ordnerstruktur des Datensatzes
--------------------------
  <dataset_root>/
    carve/
      clip_001.mp4
      clip_002.mp4
      ...
    slide/
      clip_001.mp4
      ...
    tuck/
      ...

Der Name jedes Unterordners wird automatisch zur Klassenbezeichnung.

Ergebnisse (landen in <output_dir>/)
---------
  X.npy           float32  Form (N, WINDOW_SIZE, 51)  — die Bewegungsfenster
  y.npy           int64    Form (N,)                  — die Klasse je Fenster
  label_map.json  { "carve": 0, "slide": 1, ... }     — Name ↔ Zahl
  stats.json      Zusammenfassung der Verarbeitung

Aufruf
-----
  python extract_features.py --dataset data/tricks --output features/
  python extract_features.py --dataset data/tricks --output features/ --frame-skip 3 --window 30 --stride 15
"""

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore

# Zentrale Vorverarbeitung — MUSS dieselbe sein wie später bei der Erkennung
from trick_utils import normalize_keypoints, WINDOW_SIZE, FEATURE_DIM

# ── Standardwerte ─────────────────────────────────────────────────────────────
# Nur jedes N-te Bild verarbeiten (2 → aus 30 fps werden ~15 fps).
# Aufeinanderfolgende Bilder sind fast identisch; das Überspringen halbiert die
# Rechenzeit, ohne dass Bewegungsinformation verloren geht.
DEFAULT_FRAME_SKIP = 2
# Schrittweite des Fensters (15 bei Fenstergröße 30 → 50 % Überlappung).
# Überlappung erzeugt mehr Trainingsbeispiele aus demselben Material und macht
# das Modell unabhängig davon, WANN im Fenster der Trick beginnt.
DEFAULT_STRIDE = 15
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_MIN_CLIPS = 20    # Klassenordner mit weniger Videos werden übersprungen


def should_skip(name: str) -> bool:
    """Prüft, ob eine Datei vom Training ausgeschlossen werden soll.

    Zwei Fälle:
      '._xyz.mp4' — macOS-Hilfsdateien (Resource Forks), keine echten Videos
      '..._failed' — misslungene Versuche; sie würden dem Modell falsche
                     Bewegungsmuster für einen Trick beibringen."""
    import re
    if name.startswith("._"):
        return True
    # Regulärer Ausdruck: "_fail" oder "_failed", gefolgt von Punkt,
    # Unterstrich oder Dateiende. IGNORECASE = Groß-/Kleinschreibung egal.
    if re.search(r"_fail(ed)?(\.|_|$)", name, re.IGNORECASE):
        return True
    return False


# ── Globale Variablen der Arbeitsprozesse (ein YOLO-Modell pro Prozess) ───────
# Beim parallelen Arbeiten startet Python mehrere unabhängige Prozesse. Jeder
# braucht sein eigenes YOLO-Modell — Modelle lassen sich nicht zwischen
# Prozessen teilen. Diese globale Variable hält es je Prozess bereit.
_worker_model: "YOLO | None" = None


def _init_worker(model_name: str) -> None:
    """Wird EINMAL beim Start jedes Arbeitsprozesses ausgeführt.
    Lädt YOLO genau einmal (nicht pro Video – das wäre extrem langsam)."""
    import torch
    global _worker_model
    _worker_model = YOLO(model_name)
    # Zwingend CPU: Grafikkarte/MPS kann nur ein Prozess sinnvoll belegen,
    # mehrere Prozesse würden sich gegenseitig blockieren oder abstürzen.
    _worker_model.to("cpu")
    # Threads pro Prozess begrenzen. Sonst startet jeder der z. B. 8 Prozesse
    # nochmals 8 Threads = 64 – das bremst durch ständiges Umschalten.
    torch.set_num_threads(2)


def _process_video_task(task: tuple) -> tuple:
    """Die Funktion, die ein Arbeitsprozess je Video ausführt.
    Muss auf oberster Ebene stehen (nicht verschachtelt), weil Python sie
    zum Versenden an andere Prozesse serialisieren ("picklen") muss."""
    video_path_str, label_idx, label_name, frame_skip, window, stride = task
    windows = extract_windows_from_video(
        Path(video_path_str), _worker_model,
        frame_skip=frame_skip, window=window, stride=stride,
    )
    return label_name, label_idx, video_path_str, windows


# ── Merkmalsextraktion für ein einzelnes Video ────────────────────────────────

def extract_windows_from_video(video_path: Path, model: YOLO,
                               frame_skip: int, window: int, stride: int
                               ) -> list[np.ndarray]:
    """
    Lässt YOLOv8-Pose auf jedem frame_skip-ten Bild eines Videos laufen,
    normalisiert die Gelenkpunkte und schneidet daraus überlappende Fenster.

    Rückgabe: Liste von np.ndarray, jedes mit der Form (window, 51)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path.name}")
        return []

    frame_vecs = []   # Liste von 51-Zahlen-Vektoren, einer je verarbeitetem Bild
    frame_idx = 0

    # ── Phase 1: Video Bild für Bild durchgehen ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break          # Videoende erreicht
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue       # dieses Bild überspringen

        vec = None
        results = model(frame, verbose=False)   # YOLO-Posenerkennung

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Bei mehreren erkannten Personen: die mit der höchsten Sicherheit
            # nehmen. In den Trainingsclips ist das die fahrende Person.
            best_i = int(results[0].boxes.conf.argmax())
            kps_xy = kps_conf = None
            if (results[0].keypoints is not None
                    and results[0].keypoints.xy is not None
                    and best_i < len(results[0].keypoints.xy)):
                # .cpu().numpy(): vom Tensor (evtl. auf GPU) zu normalem numpy
                kps_xy = results[0].keypoints.xy[best_i].cpu().numpy()
                kps_conf = (results[0].keypoints.conf[best_i].cpu().numpy()
                            if results[0].keypoints.conf is not None else None)
            vec = normalize_keypoints(kps_xy, kps_conf)

        # Wenn keine Person gefunden oder die Normalisierung scheiterte, wird ein
        # Nullvektor eingesetzt. Wichtig: die Zeitachse darf keine Lücken haben,
        # sonst würde die Bewegung künstlich beschleunigt wirken.
        frame_vecs.append(vec if vec is not None else np.zeros(
            FEATURE_DIM, dtype=np.float32))

    cap.release()

    if len(frame_vecs) < window:
        return []   # Video zu kurz für auch nur ein Fenster

    # ── Phase 2: Gleitendes Fenster über die Bildfolge schieben ──
    # Beispiel bei window=30, stride=15:
    #   Fenster 1 = Frames  0–29
    #   Fenster 2 = Frames 15–44   (Hälfte überlappt mit Fenster 1)
    #   Fenster 3 = Frames 30–59 ...
    windows = []
    for start in range(0, len(frame_vecs) - window + 1, stride):
        w = np.stack(frame_vecs[start: start + window]).astype(np.float32)
        windows.append(w)

    return windows


# ── Zusammenführen der Zwischenspeicher ───────────────────────────────────────

def _merge_cache(cache_dir: Path, output_dir: Path, label_map: dict) -> None:
    """Fügt alle klassenweisen .npz-Zwischendateien zu X.npy / y.npy zusammen.

    WARUM ZWISCHENSPEICHER?
    Die Extraktion dauert bei vielen Videos Stunden. Jede fertige Klasse wird
    einzeln abgelegt. Bricht der Lauf ab, gehen nur die Daten der gerade
    laufenden Klasse verloren — der Rest wird beim Neustart übersprungen."""
    all_X, all_y = [], []
    missing = []

    # Nach Index sortiert durchgehen, damit die Reihenfolge stets identisch ist
    for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
        cache_file = cache_dir / f"{name}.npz"
        if not cache_file.exists():
            missing.append(name)
            continue
        data = np.load(cache_file)
        all_X.append(data["X"])
        all_y.append(data["y"])
        print(f"  [MERGE] {name:<35} {len(data['X']):>5} windows")

    if missing:
        print(f"\n[WARN] Missing cache for: {missing}")
        print(f"       These classes will NOT be in the merged dataset.")

    if not all_X:
        print("[ERROR] No cache files found. Run extraction first.")
        return

    # Alle Klassen zu einem großen Datensatz stapeln
    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)

    print(f"\n{'='*55}")
    print(f"  MERGE COMPLETE")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Saved to: {output_dir}/")
    print(f"{'='*55}")


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract pose keypoint windows from labeled trick videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",    type=str, required=True,
                        help="Root folder containing one subfolder per trick class")
    parser.add_argument("--output",     type=str, default="features",
                        help="Directory to save X.npy, y.npy, label_map.json")
    # n = nano (schnellste), s = small, m = medium, l = large (genaueste)
    parser.add_argument("--model",      type=str, default="s",
                        choices=["n", "s", "m", "l"],
                        help="YOLOv8-Pose model size")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP,
                        help="Process every N-th frame")
    parser.add_argument("--window",     type=int, default=WINDOW_SIZE,
                        help="Sliding window length in frames")
    parser.add_argument("--stride",     type=int, default=DEFAULT_STRIDE,
                        help="Sliding window stride")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="YOLOv8 detection confidence threshold")
    parser.add_argument("--min-clips",  type=int,   default=DEFAULT_MIN_CLIPS,
                        help="Skip class folders that have fewer than this many video files")
    # Ein Kern bleibt frei, damit der Rechner bedienbar bleibt
    parser.add_argument("--workers",    type=int,   default=max(1, cpu_count() - 1),
                        help="Parallel worker processes (default: CPU cores - 1)")
    parser.add_argument("--classes",    type=str,   default=None, nargs="+",
                        help="Process only these class name(s). Omit to process all eligible classes.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip extraction — just merge existing cache files into X.npy/y.npy")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_dir = Path(args.output)
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)   # exist_ok: kein Fehler, falls schon da
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── ALLE geeigneten Klassenordner ermitteln (für eine stabile label_map) ──
    # Ordner, die mit "." beginnen, sind versteckte Systemordner
    all_class_dirs = sorted([d for d in dataset_root.iterdir()
                             if d.is_dir() and not d.name.startswith(".")])
    eligible_dirs = []
    for d in all_class_dirs:
        clip_count = sum(1 for f in d.iterdir()
                         if f.suffix.lower() in VIDEO_EXTS and not should_skip(f.name))
        if clip_count >= args.min_clips:
            eligible_dirs.append(d)
        else:
            # Zu wenige Clips: Das Modell könnte diese Klasse nicht verlässlich
            # lernen und die Statistik würde verzerrt.
            print(
                f"[SKIP] {d.name} — only {clip_count} clips (< --min-clips {args.min_clips})")

    if not eligible_dirs:
        print(
            f"[ERROR] No class folders passed the --min-clips={args.min_clips} threshold.")
        return

    # WICHTIG: Die Zuordnung Name→Zahl wird über ALLE geeigneten Klassen gebildet,
    # nicht nur über die heute verarbeiteten. Sonst bekäme "Shuvit" bei einem
    # Teillauf die Zahl 0 und bei einem anderen die 3 — die Daten wären unbrauchbar.
    label_map = {d.name: idx for idx, d in enumerate(eligible_dirs)}

    # Zuordnung sofort speichern, damit train.py immer das vollständige Bild hat
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # ── Auf die gewünschten Klassen einschränken (Option --classes) ──
    if args.classes:
        unknown = [c for c in args.classes if c not in label_map]
        if unknown:
            print(f"[ERROR] Unknown class(es): {unknown}")
            print(f"        Available: {list(label_map.keys())}")
            return
        work_dirs = [d for d in eligible_dirs if d.name in args.classes]
    else:
        work_dirs = eligible_dirs

    # ── Betriebsart "nur zusammenführen" ──
    if args.merge_only:
        _merge_cache(cache_dir, output_dir, label_map)
        return

    print(f"\n[INFO] Full label map: {len(label_map)} classes")
    print(f"[INFO] Processing this run: {[d.name for d in work_dirs]}")

    model_name = f"yolov8{args.model}-pose.pt"

    # ── Jede Klasse getrennt verarbeiten ──
    for class_dir in work_dirs:
        label = class_dir.name
        label_idx = label_map[label]
        cache_file = cache_dir / f"{label}.npz"

        # Bereits vorhandene Ergebnisse nicht neu berechnen
        if cache_file.exists():
            print(f"\n[CACHED] {label} — already extracted, skipping. "
                  f"(delete {cache_file} to re-run)")
            continue

        video_files = [vf for vf in sorted(class_dir.iterdir())
                       if vf.suffix.lower() in VIDEO_EXTS and not should_skip(vf.name)]
        total_videos = len(video_files)

        print(f"\n[CLASS] {label}  ({total_videos} videos, label={label_idx})")

        # Arbeitspakete vorbereiten: ein Tupel je Video
        tasks = [(str(vf), label_idx, label, args.frame_skip, args.window, args.stride)
                 for vf in video_files]
        # Nie mehr Prozesse als Videos starten
        workers = min(args.workers, total_videos)

        class_X, class_y = [], []
        skipped = 0

        if workers == 1:
            # ── Variante A: seriell, ein Video nach dem anderen ──
            print(
                f"[INFO] Loading YOLOv8{args.model}-pose model (single worker)...")
            model = YOLO(model_name)
            for i, (vp, li, lbl, fs, win, st) in enumerate(tasks):
                print(f"  [{i+1}/{total_videos}] {Path(vp).name} ...",
                      end=" ", flush=True)   # flush: sofort anzeigen, nicht puffern
                windows = extract_windows_from_video(Path(vp), model,
                                                     frame_skip=fs, window=win, stride=st)
                if not windows:
                    print("skipped")
                    skipped += 1
                else:
                    class_X.extend(windows)
                    # Jedes Fenster bekommt dieselbe Klassennummer wie sein Video
                    class_y.extend([li] * len(windows))
                    print(f"{len(windows)} windows")
        else:
            # ── Variante B: parallel über mehrere Prozessorkerne ──
            print(f"[INFO] Using {workers} parallel workers...")
            with Pool(processes=workers,
                      initializer=_init_worker,     # lädt YOLO je Prozess einmal
                      initargs=(model_name,)) as pool:
                # imap_unordered: Ergebnisse kommen, sobald sie fertig sind
                # (Reihenfolge egal → kein Warten auf ein langsames Video)
                for i, (lbl, li, vp, windows) in enumerate(
                        pool.imap_unordered(_process_video_task, tasks)):
                    status = f"{len(windows)} windows" if windows else "skipped"
                    print(
                        f"  [{i+1}/{total_videos}] {Path(vp).name} → {status}")
                    if windows:
                        class_X.extend(windows)
                        class_y.extend([li] * len(windows))
                    else:
                        skipped += 1

        if not class_X:
            print(
                f"  [WARN] No windows extracted for {label}, skipping cache save.")
            continue

        # Ergebnisse dieser Klasse komprimiert zwischenspeichern
        X_arr = np.stack(class_X).astype(np.float32)
        y_arr = np.array(class_y, dtype=np.int64)
        np.savez_compressed(cache_file, X=X_arr, y=y_arr)
        print(f"  [SAVED] {cache_file.name}  —  {len(X_arr)} windows  "
              f"({skipped} videos skipped)")

    # ── Automatisch zusammenführen, sobald alle Klassen fertig sind ──
    cached = sorted(cache_dir.glob("*.npz"))
    done_classes = {f.stem for f in cached}    # .stem = Dateiname ohne Endung
    all_classes = set(label_map.keys())
    remaining = all_classes - done_classes     # Mengendifferenz: was fehlt noch?

    if remaining:
        print(
            f"\n[INFO] {len(done_classes)}/{len(all_classes)} classes cached so far.")
        print(f"       Still needed: {sorted(remaining)}")
        print(f"       Run with --merge-only once all classes are done.")
    else:
        print(
            f"\n[INFO] All {len(all_classes)} classes cached. Merging now...")
        _merge_cache(cache_dir, output_dir, label_map)


if __name__ == "__main__":
    main()
