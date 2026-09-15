#!/usr/bin/env python3
"""
Merkmalsextraktion auf Grundlage eines Manifests.

Unterschied zu `extract_features.py`:
  - Die Videoliste kommt aus einer CSV-Datei, nicht aus einem Ordnerbaum.
    Damit sind Dubletten bereits ausgeschlossen und die Klassenzuordnung
    ist nachvollziehbar dokumentiert.
  - Zu jedem Fenster wird die `clip_id` des Ursprungsvideos mitgespeichert.
    Erst dadurch kann `train.py` die Aufteilung clipweise vornehmen, sodass
    überlappende Fenster desselben Videos nicht auf Training und Test
    verteilt werden.

Die eigentliche Verarbeitung eines Videos übernimmt unverändert
`extract_windows_from_video` aus `extract_features.py` — die Vorverarbeitung
bleibt damit identisch zum bisherigen Vorgehen.

WICHTIG: Liegen die Videos auf einer externen Platte, wird davon nur gelesen.
Sämtliche Ausgaben landen im angegebenen Ausgabeverzeichnis.

Beispiel:
  python3 extract_features_manifest.py \
      --manifest data/manifest_20klassen.csv \
      --output   features/tricks_v2
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np  # type: ignore

from extract_features import (
    DEFAULT_FRAME_SKIP,
    DEFAULT_STRIDE,
    _init_worker,
    extract_windows_from_video,
)
from trick_utils import WINDOW_SIZE, FEATURE_DIM

_worker_model = None   # wird in jedem Arbeitsprozess von _init_worker gesetzt


def _aufgabe(task: tuple) -> tuple:
    """Verarbeitet ein Video in einem Arbeitsprozess.

    Muss auf oberster Ebene stehen, damit Python die Funktion zum Versenden
    an andere Prozesse serialisieren kann.
    """
    import extract_features as ef
    pfad, klasse, clip_id, frame_skip, window, stride = task
    fenster = extract_windows_from_video(
        Path(pfad), ef._worker_model,
        frame_skip=frame_skip, window=window, stride=stride,
    )
    return klasse, clip_id, fenster


def manifest_lesen(pfad: Path, umschreiben=None) -> list[dict]:
    """Liest das Manifest und prüft, ob alle Videos erreichbar sind."""
    with open(pfad, newline="", encoding="utf-8") as fh:
        zeilen = list(csv.DictReader(fh))

    if umschreiben:
        zeilen = [umschreiben(z) for z in zeilen]

    noetig = {"clip_id", "klasse", "pfad"}
    fehlend = noetig - set(zeilen[0].keys() if zeilen else [])
    if fehlend:
        sys.exit(f"[FEHLER] Im Manifest fehlen die Spalten: {sorted(fehlend)}")

    unerreichbar = [z for z in zeilen if not Path(z["pfad"]).is_file()]
    if unerreichbar:
        print(f"[WARN] {len(unerreichbar)} Videos nicht erreichbar "
              f"(externe Platte angeschlossen?). Beispiel:")
        for z in unerreichbar[:3]:
            print(f"        {z['pfad']}")
        if len(unerreichbar) == len(zeilen):
            sys.exit("[FEHLER] Kein einziges Video erreichbar — Abbruch.")
        zeilen = [z for z in zeilen if Path(z["pfad"]).is_file()]

    doppelt = len(zeilen) - len({z["clip_id"] for z in zeilen})
    if doppelt:
        print(f"[WARN] {doppelt} doppelte clip_id im Manifest — werden übersprungen.")
        gesehen, eindeutig = set(), []
        for z in zeilen:
            if z["clip_id"] not in gesehen:
                gesehen.add(z["clip_id"]); eindeutig.append(z)
        zeilen = eindeutig

    return zeilen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV mit clip_id, klasse, pfad")
    p.add_argument("--output", required=True, help="Zielverzeichnis für X.npy, y.npy, clips.npy")
    p.add_argument("--video-root", default=None,
                   help="Ersetzt den Laufwerksteil der Pfade im Manifest. "
                        "Noetig, wenn die Platte anders eingehaengt ist, z. B. "
                        "--video-root /Volumes/HHN_LBSBLab1")
    p.add_argument("--model", default="yolov8s-pose.pt")
    p.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    p.add_argument("--classes", nargs="*", default=None,
                   help="nur diese Klassen verarbeiten (sonst alle)")
    args = p.parse_args()

    if args.video_root:
        # Pfade auf den neuen Einhaengepunkt umschreiben (nur im Speicher)
        import re as _re
        def _um(z):
            z["pfad"] = _re.sub(r"^/Volumes/[^/]+", args.video_root.rstrip("/"), z["pfad"])
            return z
        globals()["_umschreiben"] = _um
    zeilen = manifest_lesen(Path(args.manifest), 
                            umschreiben=globals().get("_umschreiben"))
    nach_klasse: dict[str, list[dict]] = defaultdict(list)
    for z in zeilen:
        nach_klasse[z["klasse"]].append(z)

    if args.classes:
        nach_klasse = {k: v for k, v in nach_klasse.items() if k in args.classes}

    ausgabe = Path(args.output)
    cache = ausgabe / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"Manifest : {args.manifest}")
    print(f"Klassen  : {len(nach_klasse)}   Videos: {sum(len(v) for v in nach_klasse.values())}")
    print(f"Prozesse : {args.workers}\n")

    # ── Je Klasse verarbeiten, Zwischenstand sichern ──
    # Bereits fertige Klassen werden übersprungen, damit ein Abbruch nicht
    # die gesamte Arbeit vernichtet.
    for klasse in sorted(nach_klasse):
        ziel = cache / f"{klasse}.npz"
        if ziel.exists():
            print(f"[SKIP] {klasse} — Zwischenstand vorhanden")
            continue

        videos = nach_klasse[klasse]
        aufgaben = [(z["pfad"], klasse, z["clip_id"],
                     args.frame_skip, args.window, args.stride) for z in videos]

        beginn = time.time()
        alle_fenster, alle_clips = [], []
        with Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.model,)) as pool:
            for i, (_, clip_id, fenster) in enumerate(
                    pool.imap_unordered(_aufgabe, aufgaben), 1):
                for f in fenster:
                    alle_fenster.append(f)
                    alle_clips.append(clip_id)
                if i % 10 == 0 or i == len(aufgaben):
                    print(f"   {klasse:32} {i:4}/{len(aufgaben)} Videos, "
                          f"{len(alle_fenster):6} Fenster", end="\r", flush=True)

        dauer = time.time() - beginn
        if not alle_fenster:
            print(f"\n[WARN] {klasse}: kein einziges Fenster erzeugt")
            continue

        X = np.stack(alle_fenster).astype(np.float32)          # (N, window, 51)
        clips = np.array(alle_clips, dtype=object)
        np.savez_compressed(ziel, X=X, clips=clips)
        print(f"\n[OK]   {klasse:32} {X.shape[0]:6} Fenster aus "
              f"{len(videos)} Videos  ({dauer/60:.1f} min)")

    # ── Zusammenführen ──
    dateien = sorted(cache.glob("*.npz"))
    if not dateien:
        sys.exit("[FEHLER] Keine Zwischenstände gefunden.")

    label_map = {d.stem: i for i, d in enumerate(dateien)}
    X_teile, y_teile, clip_teile = [], [], []
    for d in dateien:
        daten = np.load(d, allow_pickle=True)
        X_teile.append(daten["X"])
        y_teile.append(np.full(len(daten["X"]), label_map[d.stem], dtype=np.int64))
        clip_teile.append(daten["clips"])

    X = np.concatenate(X_teile)
    y = np.concatenate(y_teile)
    clips = np.concatenate(clip_teile)

    np.save(ausgabe / "X.npy", X)
    np.save(ausgabe / "y.npy", y)
    np.save(ausgabe / "clips.npy", clips)
    with open(ausgabe / "label_map.json", "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2, ensure_ascii=False)

    print(f"\nX      : {X.shape}")
    print(f"y      : {y.shape}   Klassen: {len(label_map)}")
    print(f"clips  : {len(set(clips.tolist()))} verschiedene Clips")
    print(f"Ablage : {ausgabe}")


if __name__ == "__main__":
    main()
