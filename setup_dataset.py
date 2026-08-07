#!/usr/bin/env python3
"""
Datensatz-Aufbereitung
======================
SCHRITT 1 der Pipeline: Kopiert und sortiert die Videoclips von der externen
Festplatte in die Projektstruktur — OHNE etwas auf der externen Platte zu ändern.

WARUM DIESER SCHRITT?
Die Rohaufnahmen liegen nach Aufnahmeterminen sortiert vor ("20251110_videos_..."),
teils in Unterordnern, teils flach mit dem Trickname im Dateinamen. Das
Trainingsskript braucht dagegen eine Struktur "ein Ordner pro Trickklasse".
Dieses Skript übersetzt zwischen beiden Welten.

Zielstruktur:
    data/tricks/<ClassName>/*.mp4   – Trick- und Step-Klassen fürs Modelltraining
    data/basics/<ClassName>/*.mp4   – grundlegende Bewegungsklassen

Regeln für jede Datei:
  • macOS-Hilfsdateien überspringen (beginnen mit '._')
  • Misslungene Versuche überspringen ('_failed' oder '_fail.' im Namen)
  • _blurred / _meh / _unclear BEHALTEN — geringere Qualität, aber gültige
    Trainingsdaten; sie machen das Modell robuster gegenüber schlechten Aufnahmen
  • Niemals eine bereits vorhandene Zieldatei überschreiben

Ausführen:
    python setup_dataset.py
    python setup_dataset.py --dry-run     # nur Vorschau, es wird nichts kopiert
"""

import argparse
import re
import shutil
from pathlib import Path

# ── Pfade ─────────────────────────────────────────────────────────────────────
# Quelle: die externe Festplatte mit den Originalaufnahmen (nur Lesezugriff)
BASE = Path("/Volumes/HHN_LBSBLab/HHN_GECKO_LBSBLab/HHN_GECKO_LBSBLab_videos")
# Ziele im Projekt. Die Pfade werden relativ zum Speicherort DIESER Datei
# gebildet (Path(__file__).resolve().parent = der Projektordner). Dadurch
# funktioniert das Skript unabhängig vom Benutzernamen und davon, aus welchem
# Verzeichnis heraus es gestartet wird.
PROJECT_ROOT = Path(__file__).resolve().parent
TRICKS_DEST = PROJECT_ROOT / "data" / "tricks"
BASICS_DEST = PROJECT_ROOT / "data" / "basics"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────


def should_skip(name: str) -> bool:
    """Gibt True zurück, wenn diese Datei nicht in die Trainingsdaten gehört."""
    if name.startswith("._"):
        return True     # macOS-Metadatei, kein echtes Video
    stem = Path(name).stem
    # Erkennt '_failed' oder '_fail' am Ende oder vor einem weiteren Namensteil
    if re.search(r"_fail(ed)?(\.|_|$)", name, re.IGNORECASE):
        return True
    return False


def extract_class_from_filename(name: str) -> str | None:
    """
    Liest den Trick-/Klassennamen aus dem Dateinamen heraus.

    Die Aufnahmegeräte haben über die Sessions hinweg unterschiedliche
    Namensschemata erzeugt. Deshalb werden hier zwei Muster geprüft.

    Unterstützte Muster:
      VID_20XXXXXX_NNNNNNNNN_ClassName_NNN_L[_tag...].mp4   (die meisten Sessions)
      20260309_HHMMSS_ClassName_L.mp4                        (Step-Sessions 2026)
      20260316_HHMMSS_ClassName_N_L.mp4
      20260323_HHMMSS_Caveman_NNN_L.mp4
    """
    stem = Path(name).stem   # Dateiname ohne Endung

    # Muster 1: VID_JJJJMMTT_Zeitstempel_Klassenname_NNN_...
    # \d{8} = genau 8 Ziffern (Datum), ([A-Za-z][A-Za-z0-9]+) = der gesuchte Name.
    # Die Klammern markieren die Fanggruppe, die group(1) zurückgibt.
    m = re.match(r"VID_\d{8}_\d+_([A-Za-z][A-Za-z0-9]+)_\d{3}", stem)
    if m:
        return m.group(1)

    # Muster 2: JJJJMMTT_HHMMSS_Klassenname_N[_L]   (20260309, 20260316, 20260323)
    m = re.match(r"\d{8}_\d{6}_([A-Za-z][A-Za-z0-9]+)", stem)
    if m:
        return m.group(1)

    return None   # Kein Muster passt → Datei wird nicht einsortiert


def copy_file(src: Path, dest_root: Path, class_name: str, dry_run: bool) -> bool:
    """Kopiert src nach dest_root/class_name/.
    Rückgabe: True, wenn kopiert wurde (bzw. im Testlauf würde)."""
    if Path(src).suffix.lower() not in VIDEO_EXTS:
        return False    # keine Videodatei
    if should_skip(src.name):
        return False    # ausgeschlossene Datei

    dst_dir = dest_root / class_name
    dst = dst_dir / src.name
    if dst.exists():
        return False    # schon vorhanden — nie überschreiben

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        # copy2 (statt copy) überträgt auch Zeitstempel und Metadaten
        shutil.copy2(src, dst)
    else:
        # needed to show path correctly
        dst_dir.mkdir(parents=True, exist_ok=True)

    return True


def process_flat_dir(src_dir: Path, dest_root: Path, dry_run: bool,
                     fixed_class: str | None = None) -> int:
    """
    Verarbeitet einen Ordner, in dem die Videos ohne Unterordner nebeneinander liegen.

    Zwei Betriebsarten:
      fixed_class gesetzt → alle Dateien kommen in genau diesen Klassenordner
                            (z. B. eine reine Caveman-Session)
      fixed_class = None  → der Klassenname wird aus jedem Dateinamen gelesen
    """
    copied = 0
    for f in sorted(src_dir.glob("*")):
        if not f.is_file():
            continue
        # "or" nutzt fixed_class, falls gesetzt, sonst den Namen aus der Datei
        cls = fixed_class or extract_class_from_filename(f.name)
        if cls:
            copied += copy_file(f, dest_root, cls, dry_run)   # True zählt als 1
    return copied


def process_subfolder_dir(src_dir: Path, dest_root: Path, dry_run: bool,
                          prefix_to_strip: str = "") -> int:
    """
    Verarbeitet einen Ordner, in dem jeder Unterordner bereits eine Klasse ist.

    Optional wird ein Session-Präfix entfernt (z. B. '20251110_'), damit aus
    '20251110_Shuvit' der saubere Klassenname 'Shuvit' wird. Ohne das würde
    derselbe Trick aus verschiedenen Sessions als verschiedene Klassen gelten.
    """
    copied = 0
    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        if prefix_to_strip and cls.startswith(prefix_to_strip):
            cls = cls[len(prefix_to_strip):]   # Präfix vorne abschneiden
        for f in sorted(class_dir.glob("*")):
            if f.is_file():
                copied += copy_file(f, dest_root, cls, dry_run)
    return copied


# ── Hauptprogramm ────────────────────────────────────────────────────────────
def main(dry_run: bool) -> None:
    """Arbeitet alle Aufnahmesessions der Reihe nach ab.

    Jede Session hat eine eigene Ordnerstruktur, deshalb wird sie einzeln und
    ausdrücklich behandelt statt über eine allgemeine Automatik — so bleibt
    nachvollziehbar, welche Daten woher stammen (wichtig für die Dokumentation)."""
    if not BASE.exists():
        print(f"[ERROR] External drive not found at {BASE}")
        print("        Make sure the drive is connected and mounted.")
        return

    label = "[DRY-RUN]" if dry_run else "[COPY]"
    total = 0

    # ── 1. Session 20251103 — nur Caveman (flache Dateien) ────────────────────
    print(f"\n{label} 20251103 → Caveman")
    src = BASE / "20251103_videos_LDF_Caveman"
    n = process_flat_dir(src, TRICKS_DEST, dry_run, fixed_class="Caveman")
    print(f"         {n} files")
    total += n

    # ── 2. Session 20251110 — nach Unterordnern organisiert ───────────────────
    s110 = BASE / "20251110_videos_LDF_Caveman-Tricks-Steps_lightissues"

    print(f"\n{label} 20251110 → Caveman")
    n = process_flat_dir(s110 / "20251110_Caveman", TRICKS_DEST, dry_run,
                         fixed_class="Caveman")
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20251110 → Tricks (subfolders)")
    n = process_subfolder_dir(
        s110 / "20251110_other" / "20251110_other_tricks",
        TRICKS_DEST, dry_run, prefix_to_strip="20251110_")
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20251110 → Steps (subfolders)")
    n = process_subfolder_dir(
        s110 / "20251110_other" / "20251110_other_steps",
        TRICKS_DEST, dry_run, prefix_to_strip="20251110_")
    print(f"         {n} files")
    total += n

    # ── 3. Session 20251201 — flache Dateien in tricks-/steps-Ordnern ─────────
    s201 = BASE / "20251201_videos_LDF_Caveman-Tricks-Steps"

    print(f"\n{label} 20251201 → Caveman")
    n = process_flat_dir(s201 / "20251201_Caveman", TRICKS_DEST, dry_run,
                         fixed_class="Caveman")
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20251201 → Tricks (flat)")
    n = process_flat_dir(
        s201 / "20251201_other" / "20251201_other_tricks",
        TRICKS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20251201 → Steps (flat)")
    n = process_flat_dir(
        s201 / "20251201_other" / "20251201_other_steps",
        TRICKS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    # ── 4. Session 20251208 — Caveman + Basics + Tricks ───────────────────────
    s208 = BASE / "20251208_videos_LDF_Caveman-Basics-Tricks_wetfloor"

    print(f"\n{label} 20251208 → Caveman")
    n = process_flat_dir(s208 / "20251208_Caveman", TRICKS_DEST, dry_run,
                         fixed_class="Caveman")
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20251208 → Caveman_diagonal")
    n = process_flat_dir(s208 / "20251208_Caveman_diagonal", TRICKS_DEST, dry_run,
                         fixed_class="Caveman_diagonal")
    # Bewusst eine EIGENE Klasse: der schräge Kamerawinkel erzeugt völlig andere
    # Gelenkkoordinaten. In einen Topf geworfen würde das die Klasse verwässern.
    print(
        f"         {n} files (kept separate — diagonal camera angle variant)")
    total += n

    print(f"\n{label} 20251208 → Tricks (flat)")
    n = process_flat_dir(
        s208 / "20251208_other" / "20251208_other_tricks",
        TRICKS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20251208 → Basics (flat)")
    n = process_flat_dir(
        s208 / "20251208_other" / "20251208_other_basics",
        BASICS_DEST, dry_run)   # anderes Ziel: Basics statt Tricks
    print(f"         {n} files")
    total += n

    # ── 5. Session 20260119 — Basics + Tricks (als 'unclear' gekennzeichnet) ──
    s119 = BASE / "20260119_videos_LDF_Basics-Tricks_noboarddrop"

    print(f"\n{label} 20260119 → Tricks (flat)")
    n = process_flat_dir(
        s119 / "20260119_other" / "20260119_other_tricks",
        TRICKS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20260119 → Basics (flat)")
    n = process_flat_dir(
        s119 / "20260119_other" / "20260119_other_basics",
        BASICS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    # ── 6. Session 20260202 — nur Tricks ─────────────────────────────────────
    s202 = BASE / "20260202_videos_LDF_Tricks_noboarddrop"

    print(f"\n{label} 20260202 → Tricks (flat)")
    n = process_flat_dir(
        s202 / "20260202_other_tricks",
        TRICKS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    # ── 7. Session 20260309 — nur Steps (je 2 Clips) ─────────────────────────
    print(f"\n{label} 20260309 → Steps (flat, 2 clips/class)")
    n = process_flat_dir(
        BASE / "20260309_videos_LDF_Steps",
        TRICKS_DEST, dry_run)
    # Nur 2 Clips pro Klasse — extract_features.py wird diese Klassen wegen
    # --min-clips überspringen, außer sie werden mit anderen Sessions ergänzt.
    print(
        f"         {n} files (note: only 2 clips per class — too few to train alone)")
    total += n

    # ── 8. Session 20260316 — Basics Wendelin + Steps ────────────────────────
    print(f"\n{label} 20260316 → Basics Wendelin (flat)")
    n = process_flat_dir(
        BASE / "20260316_videos_LDF_Basics_Wendelin_Reha",
        BASICS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    print(f"\n{label} 20260316 → Steps (flat)")
    n = process_flat_dir(
        BASE / "20260316_videos_LDF_Steps",
        TRICKS_DEST, dry_run)
    print(f"         {n} files")
    total += n

    # ── 9. Session 20260323 — nur Caveman ────────────────────────────────────
    print(f"\n{label} 20260323 → Caveman")
    n = process_flat_dir(
        BASE / "20260323_videos_LDF_Caveman",
        TRICKS_DEST, dry_run, fixed_class="Caveman")
    print(f"         {n} files")
    total += n

    # ── Abschlussbericht ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    if dry_run:
        print(
            f"[DRY-RUN] Would copy {total} files total. Run without --dry-run to execute.")
    else:
        print(f"[DONE] Copied {total} files total.")
        print(f"\nDestinations:")
        print(f"  Tricks/Steps: {TRICKS_DEST}")
        print(f"  Basics:       {BASICS_DEST}")

        # Übersicht: wie viele Clips hat am Ende jede Klasse?
        # Klassen unter 20 Clips werden markiert, weil extract_features.py sie
        # standardmäßig aussortiert.
        print(f"\n{'─'*60}")
        print("Trick/Step class clip counts:")
        for cls_dir in sorted(TRICKS_DEST.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*.mp4")))
                flag = " ⚠ (< 20 clips — borderline)" if count < 20 else ""
                print(f"  {cls_dir.name:<45} {count:>4} clips{flag}")

        print(f"\nBasics class clip counts:")
        for cls_dir in sorted(BASICS_DEST.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*.mp4")))
                print(f"  {cls_dir.name:<45} {count:>4} clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organise external drive data into project structure")
    # action="store_true": Der Schalter braucht keinen Wert. Steht --dry-run in
    # der Befehlszeile, ist er True, sonst False.
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview which files would be copied without actually copying")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
