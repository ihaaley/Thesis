"""
main.py — HERZSTÜCK DES PROJEKTS: Live-Analyse eines Skateboard-/Longboard-Videos.

WAS DIESES SKRIPT MACHT (in dieser Reihenfolge, für jedes einzelne Videobild):
  1. YOLOv8-Pose sucht alle Personen und ihre 17 Körperpunkte
  2. YOLOv8 sucht zusätzlich das Skateboard (COCO-Klasse 36)
  3. Jedes Board wird der nächstgelegenen Person zugeordnet
  4. Aus den Punkten werden Messwerte berechnet: Kniewinkel, Tempo, Boardhöhe,
     Flugzeit, Bewegungsphase, Stance, Sturzerkennung
  5. Die letzten 30 Posen gehen in das trainierte BiLSTM → Trickvorhersage
  6. Alles wird ins Bild gezeichnet (Skelett, Karten, Bestenliste, Balken)
  7. Optional: Speichern als Video, CSV-Datei und SQLite-Datenbank
  8. Am Ende erscheint eine Ergebniskarte mit dem Gesamturteil

Steuerung während der Wiedergabe:  q = beenden, p = Pause, r = zurücksetzen

Diese Datei liefert außerdem alle Funktionen, die viewer.py importiert.
"""

import cv2  # type: ignore
import argparse
import csv
import time
import json
import os
import sqlite3
import math
from collections import deque
import numpy as np  # type: ignore
import torch  # type: ignore
from ultralytics import YOLO  # type: ignore
from trick_utils import normalize_keypoints, load_classifier, WINDOW_SIZE as TRICK_WINDOW

# Die 17 Körperpunkte des COCO-Standards, in genau dieser Reihenfolge.
# Der Index in dieser Liste ist überall im Code der Zugriffsschlüssel:
# z. B. ist 15 = linker Knöchel, 11 = linke Hüfte.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKATEBOARD_CLASS_ID = 36  # COCO-Klasse 'skateboard' – das Standardmodell kennt sie bereits
COLOR_SKATEBOARD = (0, 165, 255)   # Orange. ACHTUNG: OpenCV nutzt BGR, nicht RGB!

# ── Konstanten für Trick-Aufblitzen und Sicherheitsbalken ───────────────────
TRICK_FLASH_FRAMES = 60      # so viele Bilder bleibt die Einblendung stehen
TRICK_FLASH_CONF_MIN = 0.50    # Mindestsicherheit, damit überhaupt eingeblendet wird
PRED_VOTE_BUF_LEN = 60      # über so viele Bilder wird abgestimmt
# derselbe Trick muss so viele Bilder in Folge führen, bevor eingeblendet wird
STABLE_FRAMES_MIN = 20
# 19 unterscheidbare BGR-Farben – eine je Trickklasse (Index = Klassenindex)
TRICK_CLASS_COLORS = [
    (255,  80,  80), (80, 255,  80), (80,  80, 255),
    (255, 255,  80), (255,  80, 255), (80, 255, 255),
    (255, 160,  80), (80, 160, 255), (160,  80, 255),
    (255,  80, 160), (160, 255,  80), (80, 255, 160),
    (200, 200,  80), (80, 200, 200), (200,  80, 200),
    (255, 120, 120), (120, 255, 120), (120, 120, 255),
    (255, 200,  50),
]

# Unterscheidbare BGR-Farben für bis zu 10 gleichzeitig verfolgte Personen
PLAYER_COLORS = [
    (0, 255, 0),      # grün
    (255, 0, 0),      # blau
    (0, 0, 255),      # rot
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # gelb
    (128, 0, 255),    # violett
    (0, 128, 255),    # orange-blau
    (255, 128, 0),    # himmelblau
    (128, 255, 0),    # limette
]

# ── Einstellbare Schwellenwerte ───────────────────────────────────────────────
# Diese Zahlen sind empirisch an den Aufnahmen dieses Projekts eingestellt.
# Bei anderer Kameraentfernung oder Auflösung müssen sie angepasst werden.
KP_CONF_THRESH = 0.40   # Mindestsicherheit, um ein Gelenk zu zeichnen/zu nutzen
# Fensterbreite für den gleitenden Mittelwert der Boardhöhe (in Bildern)
HEIGHT_SMOOTH_N = 7
AIRBORNE_THRESH = 12     # ab so vielen Pixeln über dem Boden gilt: in der Luft

# ── Zeitbasis für alle Dauermessungen ────────────────────────────────────────
# Bei einer Videodatei zählt die Zeit IM VIDEO, nicht die Uhrzeit des Rechners.
# Wird ein Video langsamer als in Echtzeit verarbeitet — bei hoher Auflösung
# sind ein bis drei Bilder pro Sekunde normal — dann wären mit time.time()
# gemessene Dauern um genau diesen Faktor zu groß. Eine Flugphase von einer
# halben Sekunde erschiene dann als eine halbe Minute.
# Im Kamerabetrieb bleibt die Uhrzeit des Rechners richtig; dort ist die
# Videouhr None und jetzt() fällt automatisch darauf zurück.
_video_uhr = None


def setze_video_uhr(sekunden):
    """Setzt die Position im Video (in Sekunden). None = Kamerabetrieb."""
    global _video_uhr
    _video_uhr = sekunden


def jetzt():
    """Aktuelle Zeit: Videozeit bei Dateien, sonst Uhrzeit des Rechners."""
    return time.time() if _video_uhr is None else _video_uhr
# grober Umrechnungsfaktor Pixel/Bild → km/h (je Kamera neu einzustellen)
SPEED_PX_TO_KMH = 0.05
TRICK_OLLIE_MIN_H = 20  # Mindesthöhe in Pixeln, um von einem Ollie zu sprechen
# maximaler Kniewinkel (Grad) in der Luftphase für einen Kickflip
TRICK_KICKFLIP_KNEE = 60
# minimaler Kniewinkel (Grad) bei fast null Höhe für einen Grind
TRICK_GRIND_KNEE = 150

# ── Konstanten für Darstellung und Spielelemente ─────────────────────────────
TRAIL_LENGTH = 25   # Länge der Bewegungsspur in Bildern
GRAPH_WIDTH = 200  # Breite des kleinen Tempo-/Höhendiagramms in Pixeln
GRAPH_HEIGHT = 60   # Höhe desselben Diagramms
GRAPH_HISTORY = 90   # so viele Bilder Verlauf zeigt das Diagramm
SLO_MO_BUFFER_SECS = 2.0  # Sekunden, die für die Zeitlupen-Wiederholung gepuffert werden
HIGHLIGHT_CONF_THRESH = 0.70  # ab dieser Sicherheit wird ein Highlight-Clip gespeichert
LANDING_STABLE_FRAMES = 10    # Bilder nach der Landung zur Bewertung der Landequalität
TRICK_SCORE_BASE = 1000    # Grundpunktzahl je Trick
BADGE_DISPLAY_SECS = 3.0    # so lange bleiben Erfolgsmeldungen stehen

# Farben des Skeletts je Bewegungsphase (BGR)
PHASE_COLORS = {
    "ground":  (0, 200, 0),     # grün
    "rising":  (0, 200, 255),   # gelb
    "peak":    (0, 0, 255),     # rot
    "landing": (0, 255, 0),     # hellgrün
}

# Trainingstipps: (Bedingungsschlüssel, Meldung)
COACHING_HINTS = [
    ("low_landing_score", "Bend knees more on landing"),
    ("arms_wide",         "Keep arms tighter for rotation"),
    ("good_height",       "Great pop height!"),
    ("long_airtime",      "Excellent hangtime!"),
]

# Erfolge/Auszeichnungen: (Schlüssel, Anzeigetext, Prüffunktion)
#  Die Prüffunktion bekommt den Zustand einer Person (st) und gibt True/False
#  zurück. 'lambda' ist eine kurze, namenlose Funktion — so lässt sich die
#  Bedingung direkt in der Liste mitschreiben, statt sie separat zu definieren.
ACHIEVEMENTS = [
    ("first_trick",      "FIRST TRICK!",
     lambda st: (st.get("trick_count") or 0) == 1),
    ("pb_height",        "NEW HEIGHT PB!",
     lambda st: bool(st.get("new_height_pb", False))),
    ("pb_airtime",       "NEW AIRTIME PB!",
     lambda st: bool(st.get("new_airtime_pb", False))),
    ("trick_streak_5",   "5 TRICK STREAK!",
     lambda st: (st.get("trick_streak") or 0) == 5),
    ("clean_landing",    "CLEAN LANDING!", lambda st: (
        st.get("last_landing_score") or 0) >= 80),
]


# ── Hilfsfunktion: Kniewinkel aus drei Gelenkpunkten berechnen ────────────────
def compute_knee_angle(kps_xy, kps_conf, side="left"):
    """Berechnet den Kniewinkel in Grad, oder None bei unzuverlässigen Punkten.

    MATHEMATIK DAHINTER:
    Das Knie ist der Scheitelpunkt zwischen zwei Strecken — Knie→Hüfte und
    Knie→Knöchel. Der Winkel dazwischen ergibt sich aus dem Skalarprodukt:
        cos(Winkel) = (v1 · v2) / (|v1| · |v2|)
    180° = Bein gestreckt, 90° = tief in der Hocke.
    """
    # Indizes im COCO-Schema: (Hüfte, Knie, Knöchel) je Seite
    idx_map = {"left": (11, 13, 15), "right": (12, 14, 16)}
    hi, ki, ai = idx_map[side]
    if kps_xy is None or len(kps_xy) <= max(hi, ki, ai):
        return None
    # Ist auch nur einer der drei Punkte unsicher, wäre der Winkel wertlos
    if kps_conf is not None:
        if any(kps_conf[j] < KP_CONF_THRESH for j in [hi, ki, ai]):
            return None
    hip = kps_xy[hi][:2].astype(float)
    knee = kps_xy[ki][:2].astype(float)
    ankle = kps_xy[ai][:2].astype(float)
    if np.all(knee == 0):
        return None      # (0,0) bedeutet bei YOLO "nicht gefunden"
    v1 = hip - knee      # Vektor vom Knie zur Hüfte
    v2 = ankle - knee    # Vektor vom Knie zum Knöchel
    # +1e-6 verhindert Division durch null, falls ein Vektor die Länge 0 hat
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    # np.clip(..., -1, 1): Rundungsfehler können knapp über 1 liegen,
    # arccos würde dann einen Fehler (NaN) liefern.
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))))


# ── Hilfsfunktion: gefilterte Knöchelposition ─────────────────────────────────
def get_ankle_y(kps_xy, kps_conf):
    """Mittlere Y-Position beider Knöchel in Pixeln, oder None bei zu geringer
    Sicherheit. Dient als Bezugspunkt für Tempo- und Sprunghöhenschätzung.

    Ist nur ein Knöchel sicher erkannt, wird dessen Wert allein genommen —
    besser eine halbe Information als gar keine."""
    if kps_xy is None or len(kps_xy) < 17:
        return None
    vals = []
    for idx in (15, 16):     # linker und rechter Knöchel
        if kps_conf is None or kps_conf[idx] >= KP_CONF_THRESH:
            y = float(kps_xy[idx][1])   # Index 1 = Y-Koordinate
            if y > 0:                   # 0 hieße "nicht erkannt"
                vals.append(y)
    return sum(vals) / len(vals) if vals else None


# ── Regelbasierte Trickerkennung ──────────────────────────────────────────────
def detect_trick(h_px, knee_angle_left, knee_angle_right, prev_h_px):
    """
    Ordnet die Bewegung des aktuellen Bildes einer Trickbezeichnung zu.
    Rückgabe: eine Zeichenkette oder None.

    HINWEIS ZUM AKTUELLEN STAND:
    Ursprünglich sollten hier feste Regeln Ollie, Kickflip und Grind
    unterscheiden. Da inzwischen das trainierte BiLSTM diese Aufgabe übernimmt,
    liefert diese Funktion nur noch "AIR" (= Luftphase, ein interner Zustand,
    keine trainierte Klasse) oder None. Die Regeln bleiben dokumentiert, weil
    sie die Punktevergabe auslösen und als Vergleichsverfahren dienen.

    Ursprüngliche Regeln:
      - OLLIE       : in der Luft, Höhe ≥ TRICK_OLLIE_MIN_H, Knie nicht stark gebeugt
      - KICKFLIP    : in der Luft + mindestens ein Kniewinkel ≤ TRICK_KICKFLIP_KNEE
      - GRIND       : bodennah (h_px ≤ 8) nach einer Luftphase,
                      beide Knie gestreckt (Winkel ≥ TRICK_GRIND_KNEE)
      - LANDING     : Höhe fällt von ≥ TRICK_OLLIE_MIN_H unter TRICK_OLLIE_MIN_H
    """
    airborne = h_px > AIRBORNE_THRESH
    # Nur die tatsächlich messbaren Kniewinkel behalten
    ka = [a for a in [knee_angle_left, knee_angle_right] if a is not None]

    if airborne:
        if ka and min(ka) <= TRICK_KICKFLIP_KNEE:
            return "AIR"   # nur interne Phase – keine trainierte Klasse
        if h_px >= TRICK_OLLIE_MIN_H:
            return "AIR"   # nur interne Phase – keine trainierte Klasse
    else:
        if ka and all(a >= TRICK_GRIND_KNEE for a in ka) and h_px <= 8:
            return None
        if prev_h_px is not None and prev_h_px >= TRICK_OLLIE_MIN_H and h_px < TRICK_OLLIE_MIN_H:
            return None
    return None


# ── Stance-Erkennung (Fußstellung) ────────────────────────────────────────────
def detect_stance(kps_xy, kps_conf):
    """Gibt 'REGULAR' (linker Fuß vorne) oder 'GOOFY' (rechter Fuß vorne) zurück,
    oder None bei unsicheren Punkten.

    EINSCHRÄNKUNG: Verglichen wird nur die X-Position der Knöchel im Bild. Das
    stimmt nur, solange die Person quer zur Kamera fährt. Deshalb wird das
    Ergebnis nicht einzeln verwendet, sondern über 30 Bilder abgestimmt
    (siehe stance_votes im Personenzustand)."""
    if kps_xy is None or len(kps_xy) < 17:
        return None
    l_ankle = kps_xy[15]
    r_ankle = kps_xy[16]
    lc = float(kps_conf[15]) if kps_conf is not None else 1.0
    rc = float(kps_conf[16]) if kps_conf is not None else 1.0
    if lc < KP_CONF_THRESH or rc < KP_CONF_THRESH:
        return None
    # In Bildkoordinaten bedeutet ein kleineres X: weiter links im Bild.
    # Verglichen wird die X-Position; der führende Fuß zeigt in Fahrtrichtung.
    # Statt pro Bild zu entscheiden, wird im Personenzustand abgestimmt.
    return "REGULAR" if float(l_ankle[0]) < float(r_ankle[0]) else "GOOFY"


# ── Einordnung der Bewegungsphase ─────────────────────────────────────────────
def get_phase(h_px, prev_h_px, airborne):
    """Bestimmt die aktuelle Phase: ground / rising / peak / landing.

    Die Unterscheidung entsteht allein aus dem Vergleich mit der Höhe des
    vorherigen Bildes:
      am Boden, vorher in der Luft  → landing (genau dieses eine Bild)
      am Boden                      → ground
      in der Luft, Höhe steigt      → rising
      in der Luft, Höhe fällt       → peak (Scheitelpunkt überschritten)"""
    if not airborne:
        if prev_h_px is not None and prev_h_px > AIRBORNE_THRESH:
            return "landing"
        return "ground"
    if prev_h_px is not None and h_px >= prev_h_px:
        return "rising"
    return "peak"


# ── Punktevergabe für einen Trick ─────────────────────────────────────────────
def compute_trick_score(h_px, airtime_s, jump_angle, landing_score, vid_height):
    """Verrechnet die Messwerte zu einer Punktzahl von 0 bis etwa 9999.

    Vier Bestandteile, jeder mit einer Obergrenze (min(...)), damit ein
    einzelner Ausreißer — etwa eine kurz falsch erkannte Höhe — die Wertung
    nicht sprengen kann:
      Höhe    max. 3000 (relativ zur Bildhöhe, dadurch auflösungsunabhängig)
      Flugzeit max. 2000
      Winkel   max. 1000
      Landung  max. 3000 (Note 0–100 × 30)
    Dazu die Grundpunktzahl von 1000."""
    h_bonus = min(h_px / max(vid_height, 1) * 3000, 3000)
    air_bonus = min(airtime_s * 1000, 2000)
    ang_bonus = min(abs(jump_angle or 0) * 10, 1000)
    land_bon = (landing_score or 0) * 30          # 0–3000
    return int(TRICK_SCORE_BASE + h_bonus + air_bonus + ang_bonus + land_bon)


# ── Bewertung der Landequalität ───────────────────────────────────────────────
def compute_landing_score(knee_angles_after_landing):
    """
    Note von 0 bis 100, abgeleitet aus der Ruhe der Kniewinkel nach der Landung.

    IDEE: Bei einer sauberen Landung bleiben die Knie stabil, die Winkel ändern
    sich kaum → geringe Varianz. Bei einem Wackler oder Sturz schwanken sie
    stark → hohe Varianz.

    Umrechnung: Varianz 0 → 100 Punkte; Varianz ab 400 → 0 Punkte.
    Fehlen Daten, gibt es die neutrale Note 50 statt einer Falschbewertung."""
    if not knee_angles_after_landing:
        return 50
    vals = [a for a in knee_angles_after_landing if a is not None]
    if len(vals) < 2:
        return 50      # aus einem einzigen Wert lässt sich keine Varianz bilden
    variance = float(np.var(vals))
    # Varianz nahe 0 → Note 100; Varianz ≥ 400 → Note 0
    score = max(0, 100 - int(variance / 4))
    return score


# ── Trainingstipp ─────────────────────────────────────────────────────────────
def get_coaching_hint(st):
    """Wählt anhand der letzten Werte einen passenden Hinweis aus, oder None.

    Die Reihenfolge der Abfragen ist die Rangfolge: ein Landefehler wird immer
    zuerst gemeldet, Lob erst danach."""
    land = st.get("last_landing_score")
    airtime = st.get("last_airtime", 0)
    # Sprunghöhe als Prozentsatz der Bildhöhe → unabhängig von der Auflösung
    h_pct = (st.get("max_height_px", 0) /
             max(st.get("vid_height", 1), 1)) * 100

    if land is not None and land < 40:
        return "Bend knees more on landing"
    if h_pct > 8:
        return "Great pop height!"
    if airtime > 0.5:
        return "Excellent hangtime!"
    return None


# ── Kleines Verlaufsdiagramm zeichnen ─────────────────────────────────────────
def draw_mini_graph(frame, history, label, x_origin, y_origin,
                    graph_w=GRAPH_WIDTH, graph_h=GRAPH_HEIGHT,
                    line_color=(0, 220, 255), bg_alpha=0.55):
    """
    Zeichnet ein kleines Liniendiagramm der letzten N Werte an die angegebene
    obere linke Ecke.
    history: deque mit Fließkommawerten
    """
    if len(history) < 2:
        return      # eine Linie braucht mindestens zwei Punkte
    vals = list(history)
    max_v = max(vals) or 1.0
    min_v = min(vals)
    # Wertebereich; "or 1.0" fängt ab, dass alle Werte gleich sind (rng = 0)
    rng = max_v - min_v or 1.0

    # Hintergrund; auf die Bildgrenzen beschneiden, damit nichts überläuft
    x2, y2 = x_origin + graph_w, y_origin + graph_h
    fh, fw = frame.shape[:2]
    x2 = min(x2, fw - 1)
    y2 = min(y2, fh - 1)
    # Halbtransparenz: auf einer Kopie zeichnen und beide Bilder mischen
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_origin, y_origin), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.rectangle(frame, (x_origin, y_origin), (x2, y2), (80, 80, 80), 1)

    # Werte in Bildkoordinaten umrechnen
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        # X: gleichmäßig über die Breite verteilen
        px = x_origin + int(i / (n - 1) * (graph_w - 4)) + 2
        # Y: der Wertebereich wird auf die Höhe abgebildet. Subtrahiert wird von
        # y2, weil in Bildkoordinaten Y nach UNTEN wächst — hohe Werte sollen
        # aber oben liegen.
        py = y2 - int((v - min_v) / rng * (graph_h - 8)) - 4
        pts.append((px, py))
    # Die Punkte durch Linien verbinden
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], line_color, 1)

    # Beschriftung mit dem aktuellen Wert
    cv2.putText(frame, f"{label}: {vals[-1]:.1f}",
                (x_origin + 3, y_origin + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, line_color, 1)


# ── Bewegungsspuren zeichnen ──────────────────────────────────────────────────
def draw_trails(frame, trail_history, color):
    """
    Zeichnet verblassende Bewegungsspuren an Knöcheln und Handgelenken.
    trail_history: deque von (kps_xy, kps_conf) — ältester Eintrag zuerst.

    Der Verblasseffekt entsteht dadurch, dass Farbe und Punktgröße mit dem
    Alter multipliziert werden: alte Punkte werden dunkler und kleiner.
    """
    TRAIL_JOINTS = [9, 10, 15,
                    16]   # linkes/rechtes Handgelenk, linker/rechter Knöchel
    n = len(trail_history)
    for t_idx, (kxy, kconf) in enumerate(trail_history):
        if kxy is None:
            continue
        # (t_idx+1)/n läuft von fast 0 (alt) bis 1 (neu)
        alpha = int(200 * (t_idx + 1) / n)   # fade older frames
        fade_color = tuple(int(c * (t_idx + 1) / n) for c in color)
        radius = max(1, int(4 * (t_idx + 1) / n))
        for j in TRAIL_JOINTS:
            if j >= len(kxy):
                continue
            kc = float(kconf[j]) if kconf is not None else 1.0
            if kc < KP_CONF_THRESH:
                continue
            kx, ky = int(kxy[j][0]), int(kxy[j][1])
            if kx == 0 and ky == 0:
                continue      # nicht erkannter Punkt
            cv2.circle(frame, (kx, ky), radius, fade_color, -1)


# ── Zeitlupen-Wiederholung als Bild-im-Bild ───────────────────────────────────
def draw_slomo_pip(frame, slomo_frames, pip_scale=0.28):
    """
    Zeichnet den Rahmen für eine Zeitlupenwiederholung unten rechts.
    slomo_frames: Liste von BGR-Bildern (der aufgezeichnete Trickausschnitt).
    Rückgabe: (Breite, Höhe, X, Y) des Bereichs — das Einsetzen der Bilder
    übernimmt die aufrufende Stelle.
    """
    if not slomo_frames:
        return
    fh, fw = frame.shape[:2]
    pip_w = int(fw * pip_scale)
    pip_h = int(fh * pip_scale)
    pip_x = fw - pip_w - 10      # 10 px Abstand zum rechten Rand
    pip_y = fh - pip_h - 10      # 10 px Abstand zum unteren Rand
    # Beschriftung
    cv2.rectangle(frame, (pip_x - 2, pip_y - 20),
                  (pip_x + pip_w + 2, pip_y), (0, 0, 0), -1)
    cv2.putText(frame, "SLO-MO REPLAY", (pip_x, pip_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return pip_w, pip_h, pip_x, pip_y


# ── Punkte-Banner für einen Trick ─────────────────────────────────────────────
def draw_trick_banner(frame, track_id, trick_name, score, color):
    """Blendet Trickname und Punktzahl groß in der Bildmitte ein.

    Der Schlagschatten entsteht durch zweimaliges Schreiben: erst schwarz um
    2 Pixel versetzt, dann farbig darüber. Ohne ihn wäre heller Text auf
    hellem Hintergrund kaum lesbar."""
    fh, fw = frame.shape[:2]
    text1 = trick_name.upper()
    text2 = f"SCORE  {score:,}"
    # Beide Zeilen mit je eigenem Versatz, eigener Größe und Strichstärke
    for text, y_off, scale, thick in [(text1, -28, 1.1, 3), (text2, 18, 0.7, 2)]:
        # Textbreite messen, um exakt zu zentrieren
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        tx = (fw - tw) // 2
        ty = fh // 2 + y_off
        cv2.putText(frame, text, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 1)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)


# ── Banner für Erfolge ────────────────────────────────────────────────────────
def draw_achievement_banner(frame, label, color=(0, 215, 255)):
    """Zeichnet einen goldenen Erfolgsstreifen über den oberen Bildrand."""
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    (tw, _), _ = cv2.getTextSize(
        f"★  {label}  ★", cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
    tx = (fw - tw) // 2
    cv2.putText(frame, f"★  {label}  ★", (tx, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)


# ── Bestenliste ───────────────────────────────────────────────────────────────
def draw_leaderboard(frame, player_state):
    """Zeichnet eine kleine Rangliste unten links (nur bei mehreren Personen sinnvoll)."""
    if not player_state:
        return
    fh, fw = frame.shape[:2]
    # Nach Punktzahl absteigend sortieren
    entries = sorted(
        [(tid, st.get("session_score", 0))
         for tid, st in player_state.items()],
        key=lambda x: x[1], reverse=True
    )
    board_w, line_h = 200, 18
    # Höhe wächst mit der Anzahl der Einträge
    board_h = 22 + len(entries) * line_h
    bx, by = 10, fh - board_h - 40   # 40 px Abstand nach unten (Platz für die FPS-Leiste)

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + board_w,
                  by + board_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + board_w,
                  by + board_h), (200, 200, 0), 1)
    cv2.putText(frame, "LEADERBOARD", (bx + 5, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 0), 1)

    # Eine Zeile je Person, in der jeweiligen Personenfarbe
    for rank, (tid, score) in enumerate(entries):
        color = get_player_color(tid)
        y = by + 22 + rank * line_h
        cv2.putText(frame, f"#{rank+1}  P{tid:03d}  {score:,}",
                    (bx + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)


# ── Benotung der Körperhaltung ────────────────────────────────────────────────
def compute_form_grade(kps_xy, kps_conf, trick_label, reference_poses):
    """
    Vergleicht die aktuelle Haltung mit einer hinterlegten Musterhaltung.
    Rückgabe: Note A/B/C/D, oder None wenn kein Muster vorliegt.
    reference_poses: dict { trick_label: np.ndarray der Form (51,) }

    Weil beide Vektoren normalisiert sind (siehe trick_utils), lassen sie sich
    direkt vergleichen: der euklidische Abstand ist ein Maß für die Abweichung.
    Je kleiner, desto näher an der Musterhaltung.
    """
    if trick_label is None or trick_label not in reference_poses:
        return None
    ref = reference_poses[trick_label]
    vec = normalize_keypoints(kps_xy, kps_conf)
    if vec is None:
        return None
    dist = float(np.linalg.norm(vec - ref))
    if dist < 0.5:
        return "A"
    if dist < 1.0:
        return "B"
    if dist < 1.75:
        return "C"
    return "D"


# ── Sturzerkennung ────────────────────────────────────────────────────────────
def detect_bail(kps_xy, kps_conf, prev_kps_xy):
    """
    Einfache Sturzerkennung: Die Nase fällt deutlich unter die Hüftmitte.

    Im Stehen ist der Kopf immer OBERHALB der Hüfte, also kleineres Y.
    Liegt die Nase mehr als 100 Pixel darunter, ist die Person waagerecht —
    ein starkes Anzeichen für einen Sturz.
    Rückgabe: True, wenn ein Sturz wahrscheinlich ist.
    """
    if kps_xy is None or len(kps_xy) < 13:
        return False
    nose_conf = float(kps_conf[0]) if kps_conf is not None else 1.0
    l_hip_conf = float(kps_conf[11]) if kps_conf is not None else 1.0
    r_hip_conf = float(kps_conf[12]) if kps_conf is not None else 1.0
    if nose_conf < KP_CONF_THRESH or l_hip_conf < KP_CONF_THRESH:
        return False
    nose_y = float(kps_xy[0][1])
    hip_mid_y = (float(kps_xy[11][1]) + float(kps_xy[12][1])) / 2
    # In Bildkoordinaten heißt ein Sturz: Nase näher am Boden (größeres Y) als
    # die Hüfte, und zwar mit deutlichem Abstand — die Person liegt waagerecht.
    return (nose_y - hip_mid_y) > 100   # Schwelle in Pixeln


def get_player_color(track_id: int):
    """Ordnet jeder Tracking-ID dauerhaft dieselbe Farbe zu.
    Der Modulo-Operator (%) lässt die Farben nach 10 Personen wieder von vorn
    beginnen, sodass es nie einen Indexfehler gibt."""
    return PLAYER_COLORS[(track_id - 1) % len(PLAYER_COLORS)]


def draw_player_panel(frame, track_id, color, box, keypoints_xy, kps_conf, conf):
    """Zeichnet ein kleines, halbtransparentes Infofeld direkt über der Box
    einer Person (Vorgängerversion der Karten oben rechts)."""
    x1, y1, x2, y2 = box
    panel_w = max(x2 - x1, 200)    # mindestens 200 px breit
    panel_h = 80
    px1 = x1
    # Das Feld sitzt über der Box; max() verhindert, dass es aus dem Bild rutscht
    py2 = max(y1 - 5, panel_h + 5)
    py1 = py2 - panel_h
    px2 = px1 + panel_w

    # Auf die Bildgrenzen beschneiden
    fh, fw = frame.shape[:2]
    px1 = max(0, px1)
    px2 = min(fw, px2)
    py1 = max(0, py1)
    py2 = min(fh, py2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)

    ankle_y = get_ankle_y(keypoints_xy, kps_conf)

    unique_code = f"P{track_id:03d}"
    # Die anzuzeigenden Zeilen: (Text, Farbe, Schriftgröße, Strichstärke)
    lines = [
        (f"ID: {unique_code}",                    color,           0.55, 2),
        (f"Conf : {conf:.2f}",                    (200, 200, 200), 0.45, 1),
        (f"Box  : ({x1},{y1})-({x2},{y2})",       (180, 180, 180), 0.40, 1),
        (f"Ankle Y : {ankle_y:.1f}px" if ankle_y else "Ankle Y : --",
         (0, 220, 255),   0.45, 1),
    ]
    for i, (text, tcol, scale, thick) in enumerate(lines):
        ty = py1 + 16 + i * 17     # jede Zeile 17 px tiefer
        if ty < py2:               # nur zeichnen, wenn noch Platz im Feld ist
            cv2.putText(frame, text, (px1 + 5, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, tcol, thick)


def assign_skateboards_to_players(player_data_list, skate_boxes):
    """
    Ordnet jedes erkannte Skateboard der nächstgelegenen Person zu (gemessen am
    Abstand der Boxmittelpunkte).

    WARUM NÖTIG? Die beiden YOLO-Modelle arbeiten unabhängig: eines findet
    Personen, das andere Boards. Welches Board zu wem gehört, weiß keines von
    beiden — diese Funktion stellt die Verbindung her.

    Rückgabe: dict { track_id -> skate_box (sx1,sy1,sx2,sy2) oder None }
    """
    # Erst einmal alle Personen auf "kein Board" setzen
    assignment = {pdata["track_id"]: None for pdata in player_data_list}
    if not skate_boxes or not player_data_list:
        return assignment

    for sbox in skate_boxes:
        sx1, sy1, sx2, sy2 = sbox
        scx = (sx1 + sx2) / 2     # Mittelpunkt des Boards
        scy = (sy1 + sy2) / 2
        best_tid = None
        best_dist = float("inf")   # unendlich: der erste Vergleich gewinnt immer
        for pdata in player_data_list:
            bx1, by1, bx2, by2 = pdata["box"]
            pcx = (bx1 + bx2) / 2  # Mittelpunkt der Person
            pcy = (by1 + by2) / 2
            # Satz des Pythagoras: gerader Abstand zwischen beiden Mittelpunkten
            dist = ((scx - pcx) ** 2 + (scy - pcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid = pdata["track_id"]
        if best_tid is not None:
            assignment[best_tid] = sbox
    return assignment


# ── Maße der Infokarten oben rechts ───────────────────────────────────────────
_HUD_CARD_W = 220   # Kartenbreite
_HUD_MARGIN = 10    # Abstand zum Bildrand
_HUD_GAP = 6     # Abstand zwischen gestapelten Karten
_HUD_HDR_H = 36    # Höhe der Kopfzeile (Personen-ID)
_HUD_ROW_H = 26    # Höhe einer Datenzeile
_HUD_PAD = 10    # innerer Abstand links/rechts

# Anzeigetext und Farbe je Bewegungsphase.
# Die \xe2\x86\x91-Folgen sind UTF-8-Bytes für Pfeilsymbole (↑ bzw. ▲).
_PHASE_META = {
    "ground":  ("GROUND",    (100, 200, 100)),
    "rising":  ("RISING  \xe2\x86\x91", (0, 230, 255)),
    "peak":    ("PEAK  \xe2\x96\xb2",   (60,  80, 255)),
    "landing": ("LANDING",   (0, 200, 220)),
}


def _draw_card_bg(frame, x1, y1, x2, y2, border_color, alpha=0.80):
    """Dunkle, halbtransparente Karte mit farbigem Rahmen."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (14, 14, 14), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)


def draw_top_right_player_panels(frame, player_data_list, skate_assignment,
                                 vid_height, ground_y, player_state):
    """
    DIE WICHTIGSTE FUNKTION DER ANZEIGE — und zugleich der Ort, an dem die
    gesamte Physikberechnung stattfindet.

    Sie zeichnet je erkannter Person eine Karte oben rechts UND aktualisiert
    dabei player_state[track_id] mit allen laufenden Werten (Tempo, Höhe,
    Flugzeit, Phase, Punkte, Erfolge).

    HINWEIS ZUR STRUKTUR: Berechnen und Zeichnen in einer Funktion zu
    vermischen ist untypisch. Es ist hier historisch gewachsen — deshalb
    trennt viewer.py beides und ruft nur die Berechnungen nach.
    """
    panel_w = _HUD_CARD_W
    margin = _HUD_MARGIN
    gap = _HUD_GAP
    fw = frame.shape[1]
    line_h = 17  # kept for stacking compat below

    for idx, pdata in enumerate(player_data_list):
        track_id = pdata["track_id"]
        color = pdata["color"]
        conf = pdata["conf"]
        bx1, by1, bx2, by2 = pdata["box"]
        kps_xy = pdata["kps_xy"]
        kps_conf = pdata["kps_conf"]
        clf_label = pdata.get("clf_label")   # Ausgabe des KI-Modells, oder None
        sbox = skate_assignment.get(track_id)

        # ── Zustand dieser Person holen oder erstmalig anlegen ──
        # setdefault legt den kompletten Startzustand nur beim ERSTEN Auftreten
        # an. In allen weiteren Bildern kommt der bestehende Eintrag zurück —
        # so bleiben Bestwerte und Punkte über das Video hinweg erhalten.
        st = player_state.setdefault(track_id, {
            "max_height_px":      0,      # höchste je erreichte Boardhöhe
            "airborne_start":     None,   # Zeitpunkt des Absprungs
            "last_airtime":       0.0,    # Dauer des letzten Sprungs
            "max_airtime":        0.0,    # längster Sprung
            "ankle_history":      deque(maxlen=10),   # für die Temposchätzung
            "speed_kmh":          0.0,
            "jump_start_ankle":   None,   # Knöchelhöhe beim Absprung
            "jump_peak_ankle":    None,   # höchster Punkt während des Sprungs
            "jump_angle":         None,
            "height_buf":         deque(maxlen=HEIGHT_SMOOTH_N),  # Glättung
            "trick_label":        None,
            "prev_h_px":          0,      # Höhe im vorherigen Bild
            # new fields
            "trail_history":      deque(maxlen=TRAIL_LENGTH),
            "speed_history":      deque(maxlen=GRAPH_HISTORY),
            "height_history":     deque(maxlen=GRAPH_HISTORY),
            "session_score":      0,
            "trick_count":        0,
            "trick_streak":       0,      # Tricks in Folge ohne Unterbrechung
            "last_landing_score": None,
            "landing_knee_buf":   [],     # Kniewinkel nach der Landung
            "landing_frames_left": 0,     # Restbilder des Beobachtungsfensters
            "new_height_pb":      False,  # persönliche Bestleistung Höhe
            "new_airtime_pb":     False,  # persönliche Bestleistung Flugzeit
            "stance_votes":       deque(maxlen=30),
            "stance":             None,
            "phase":              "ground",
            "pending_badge":      None,
            "badge_until":        0.0,
            "coaching_hint":      None,
            "bail_detected":      False,
            "vid_height":         vid_height,
        })

        # ── Aus den Gelenkpunkten abgeleitete Werte ──
        ankle_y = get_ankle_y(kps_xy, kps_conf)
        knee_left = compute_knee_angle(kps_xy, kps_conf, "left")
        knee_right = compute_knee_angle(kps_xy, kps_conf, "right")

        # Stance über eine Mehrheitsentscheidung der letzten 30 Bilder
        sv = detect_stance(kps_xy, kps_conf)
        if sv:
            st["stance_votes"].append(sv)
        if st["stance_votes"]:
            from collections import Counter
            st["stance"] = Counter(st["stance_votes"]).most_common(1)[0][0]

        # Sturzerkennung
        st["bail_detected"] = detect_bail(kps_xy, kps_conf, None)

        # Pose für die Bewegungsspur merken
        st["trail_history"].append((kps_xy, kps_conf))

        # Tempo aus der Knöchelverschiebung (Pixel/Bild → Näherung in km/h).
        # Verglichen wird der neueste mit dem ältesten Wert im Puffer, also die
        # Bewegung über die letzten 10 Bilder — das glättet Einzelaussetzer.
        if ankle_y is not None:
            st["ankle_history"].append(ankle_y)
        if len(st["ankle_history"]) >= 2:
            disp = abs(st["ankle_history"][-1] - st["ankle_history"][0])
            st["speed_kmh"] = disp * SPEED_PX_TO_KMH
        st["speed_history"].append(st["speed_kmh"])

        # ── Boardhöhe (geglättet) ──
        h_px = h_pct = max_h_pct = 0
        airborne = False
        skate_status = None
        now = jetzt()
        airtime_s = 0.0

        # Der gesamte folgende Block läuft nur, wenn dieser Person ein Board
        # zugeordnet werden konnte — ohne Board gibt es keine Höhenreferenz.
        if sbox is not None:
            _, _, _, sb_bottom = sbox
            # Höhe = Boden minus Unterkante des Boards, nie negativ
            raw_h = max(0, (ground_y or vid_height) - sb_bottom)
            st["height_buf"].append(raw_h)
            h_px = int(sum(st["height_buf"]) /
                       len(st["height_buf"]))  # geglättet
            h_pct = h_px / vid_height * 100 if vid_height > 0 else 0

            # Neue Bestleistung in der Höhe?
            prev_max = st["max_height_px"]
            if h_px > st["max_height_px"]:
                st["max_height_px"] = h_px
                st["new_height_pb"] = True
            else:
                st["new_height_pb"] = False
            max_h_pct = st["max_height_px"] / \
                vid_height * 100 if vid_height > 0 else 0

            airborne = h_px > AIRBORNE_THRESH

            # Bewegungsphase bestimmen
            st["phase"] = get_phase(h_px, st["prev_h_px"], airborne)

            # ── Flugzeitmessung ──
            if airborne:
                if st["airborne_start"] is None:
                    # Erstes Bild in der Luft = Absprung
                    st["airborne_start"] = now
                    st["jump_start_ankle"] = ankle_y
                    st["jump_peak_ankle"] = ankle_y
                    st["landing_knee_buf"] = []
                else:
                    # Höchsten Punkt verfolgen (kleineres Y = höher im Bild)
                    if ankle_y is not None and st["jump_peak_ankle"] is not None:
                        if ankle_y < st["jump_peak_ankle"]:
                            st["jump_peak_ankle"] = ankle_y
                airtime_s = now - st["airborne_start"]
            else:
                if st["airborne_start"] is not None:
                    # Erstes Bild wieder am Boden = Landung
                    finished_airtime = now - st["airborne_start"]
                    st["last_airtime"] = finished_airtime
                    prev_max_air = st["max_airtime"]
                    st["max_airtime"] = max(
                        st["max_airtime"], finished_airtime)
                    st["new_airtime_pb"] = st["max_airtime"] > prev_max_air

                    # Sprungwinkel aus senkrechtem Hub und Boardhöhe
                    if st["jump_start_ankle"] and st["jump_peak_ankle"]:
                        dy = st["jump_start_ankle"] - st["jump_peak_ankle"]
                        st["jump_angle"] = float(
                            np.degrees(np.arctan2(dy, max(1, h_px))))

                    # Beobachtungsfenster für die Landequalität öffnen
                    st["landing_frames_left"] = LANDING_STABLE_FRAMES

                    # Sprungzustand zurücksetzen
                    st["airborne_start"] = None
                    st["jump_start_ankle"] = None
                    st["jump_peak_ankle"] = None

                # Bewertung der Landung: einige Bilder lang Kniewinkel sammeln
                if st["landing_frames_left"] > 0:
                    ka_avg = None
                    if knee_left is not None and knee_right is not None:
                        ka_avg = (knee_left + knee_right) / 2
                    elif knee_left is not None:
                        ka_avg = knee_left        # nur ein Bein sichtbar
                    elif knee_right is not None:
                        ka_avg = knee_right
                    if ka_avg is not None:
                        st["landing_knee_buf"].append(ka_avg)
                    st["landing_frames_left"] -= 1
                    if st["landing_frames_left"] == 0:
                        # Fenster abgelaufen → Note berechnen
                        score = compute_landing_score(st["landing_knee_buf"])
                        st["last_landing_score"] = score

            skate_status = "IN AIR" if airborne else "ON GROUND"

            # ── Regelbasierte Trickerkennung + Punktevergabe ──
            trick = detect_trick(h_px, knee_left, knee_right, st["prev_h_px"])
            if trick:
                # Punkte nur bei einem NEUEN Trick, sonst würde jedes Bild eines
                # langen Tricks erneut zählen.
                if st["trick_label"] != trick:
                    # new trick detected — score it
                    land_score = st.get("last_landing_score") or 50
                    tscore = compute_trick_score(h_px, st.get("last_airtime", 0),
                                                 st.get("jump_angle"), land_score, vid_height)
                    st["session_score"] += tscore
                    st["trick_count"] += 1
                    st["trick_streak"] = st.get("trick_streak", 0) + 1
                    st["last_trick_score"] = tscore
                    st["show_score_until"] = now + 2.5   # Anzeige für 2,5 Sekunden
                st["trick_label"] = trick
            elif not airborne:
                # Wieder am Boden ohne Trick → Serie endet
                st["trick_label"] = None
                st["trick_streak"] = 0
            st["prev_h_px"] = h_px
        st["height_history"].append(h_px)

        # ── Erfolge prüfen ──
        st["coaching_hint"] = get_coaching_hint(st)
        for ach_key, ach_label, ach_fn in ACHIEVEMENTS:
            # 'last_ach' verhindert, dass derselbe Erfolg dauernd erneut auslöst.
            # 'break' sorgt dafür, dass pro Bild höchstens ein Banner erscheint.
            if ach_fn(st) and st.get("last_ach") != ach_key:
                st["pending_badge"] = ach_label
                st["badge_until"] = now + BADGE_DISPLAY_SECS
                st["last_ach"] = ach_key
                break

        # ── Karteninhalt zusammenstellen ───────────────────────────────────
        phase_label, phase_color = _PHASE_META.get(
            st["phase"], ("--", (150, 150, 150)))
        stance_text = st["stance"] if st["stance"] else "--"
        speed_text = f"{st['speed_kmh']:.1f} km/h"

        # Flugzeit / Höhe (nur mit zugeordnetem Board)
        now = jetzt()
        airtime_s = 0.0
        h_pct = 0.0
        max_air = st.get("max_airtime", 0.0)
        if sbox is not None:
            if st.get("airborne_start"):
                airtime_s = now - st["airborne_start"]
            vid_h = st.get("vid_height", 1) or 1
            h_buf = st.get("height_buf", [])
            if h_buf:
                h_px = int(sum(h_buf) / len(h_buf))
                h_pct = h_px / vid_h * 100

        # Die Zeilen der Karte: (Beschriftung, Wert, Wertfarbe)
        rows = [
            ("PHASE",   phase_label,                    phase_color),
            ("SPEED",   speed_text,                     (0, 255, 200)),
        ]
        # Höhe und Flugzeit nur anzeigen, wenn sie überhaupt messbar sind
        if sbox is not None:
            rows.append(("HEIGHT",  f"{h_pct:.1f}%",    (0, 210, 255)))
            rows.append(("AIRTIME", f"{airtime_s:.2f}s  best {max_air:.2f}s",
                         (255, 200, 80)))
        rows.append(("STANCE",  stance_text,         (180, 180, 200)))

        card_h = _HUD_HDR_H + len(rows) * _HUD_ROW_H + 6

        # ── Karten untereinander stapeln (eine je Person) ──────────────────
        # Die Y-Position ergibt sich aus der Gesamthöhe aller vorherigen Karten.
        # Diese können unterschiedlich hoch sein, je nachdem ob ein Board
        # erkannt wurde — deshalb wird hier nachgerechnet statt fest gerastert.
        cy1 = margin
        for j in range(idx):
            prev_tid = player_data_list[j]["track_id"]
            prev_sbox = skate_assignment.get(prev_tid)
            prev_rows = 4 if prev_sbox else 3          # PHASE+SPEED+STANCE±BOARD
            cy1 += _HUD_HDR_H + prev_rows * _HUD_ROW_H + 6 + gap

        cx1 = fw - panel_w - margin    # rechtsbündig
        cx2 = cx1 + panel_w
        cy2 = cy1 + card_h

        # Passt keine weitere Karte mehr ins Bild, abbrechen
        if cy2 > frame.shape[0]:
            break

        # ── Karte zeichnen ─────────────────────────────────────────────────
        _draw_card_bg(frame, cx1, cy1, cx2, cy2, color)

        # Kopfzeile: farbiger Streifen links + Personen-ID
        cv2.rectangle(frame, (cx1, cy1), (cx1 + 3, cy2), color, -1)
        player_label = f"P{track_id:03d}"
        cv2.putText(frame, player_label,
                    (cx1 + _HUD_PAD + 4, cy1 + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, color, 2)

        # Dünne Trennlinie unter der Kopfzeile
        sep_y = cy1 + _HUD_HDR_H
        cv2.line(frame, (cx1 + 4, sep_y), (cx2 - 2, sep_y), (50, 50, 50), 1)

        # Zeilen: Beschriftung in gedämpftem Grau, Wert in der Signalfarbe
        for ri, (lbl, val, vcol) in enumerate(rows):
            ry = sep_y + 6 + ri * _HUD_ROW_H + 16
            if ry > cy2:
                break      # kein Platz mehr in dieser Karte
            cv2.putText(frame, lbl,
                        (cx1 + _HUD_PAD + 4, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)
            cv2.putText(frame, val,
                        (cx1 + _HUD_PAD + 4, ry + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, vcol, 1)


def draw_fps(frame, fps_val):
    """Schmale Statusleiste am unteren Bildrand mit der Bildrate."""
    fh, fw = frame.shape[:2]
    bar_h = 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, fh - bar_h), (fw, fh), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, f"FPS  {fps_val:.0f}",
                (10, fh - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 100), 1)


# ── Einblendanimation für erkannte Tricks ─────────────────────────────────────
def draw_trick_pop(frame, trick_name, frames_left, trick_color,
                   total=TRICK_FLASH_FRAMES):
    """Zeigt den Tricknamen groß in der Bildmitte, mit Ausblendeffekt.

    frames_left zählt von 60 auf 0 herunter. t = frames_left/total läuft also
    von 1.0 (frisch) auf 0.0 (gleich weg) und steuert sowohl Größe als auch
    Helligkeit — daraus entsteht das Aufpoppen und Verblassen."""
    fh, fw = frame.shape[:2]
    t = frames_left / max(total, 1)
    # Schrift wächst in der ersten Hälfte, bleibt danach auf voller Größe
    scale = 1.1 + 0.4 * min(t * 2, 1.0)
    thick = 3
    display = trick_name.replace("_", " ").upper()
    if len(display) > 24:
        display = display[:24]     # sehr lange Namen abschneiden
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    tx = (fw - tw) // 2
    ty = int(fh * 0.42) + th // 2   # etwas oberhalb der Bildmitte
    pad = 16
    # Halbtransparenter Hintergrund, der mit dem Text mitverblasst
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (tx - pad, ty - th - pad),
                  (tx + tw + pad, ty + pad),
                  (10, 10, 10), -1)
    bg_alpha = 0.70 * t
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    # Schlagschatten + farbiger Text; die Farbe wird mit t abgedunkelt
    col = tuple(int(c * t) for c in trick_color)
    cv2.putText(frame, display, (tx + 3, ty + 3),
                cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, display, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, scale, col, thick)
    # Punktreihe darunter als Fortschrittsanzeige der Einblendung
    dot_y = ty + pad + 8
    total_dots = TRICK_FLASH_FRAMES
    filled = frames_left
    for d in range(10):
        cx = fw // 2 - 45 + d * 10
        # gefüllte Punkte in Trickfarbe, der Rest dunkelgrau
        dc = col if d < int(filled / max(total_dots, 1) * 10) else (60, 60, 60)
        cv2.circle(frame, (cx, dot_y), 3, dc, -1)


# ── Balkenanzeige der drei wahrscheinlichsten Tricks ──────────────────────────
def draw_confidence_bars(frame, top3_names, top3_probs, trick_color):
    """Feld unten links: die drei wahrscheinlichsten Tricks mit Balken.

    Sehr nützlich für die Fehleranalyse: man sieht sofort, ob das Modell
    schwankt oder ob zwei Tricks dicht beieinanderliegen."""
    if not top3_names:
        return
    fh, fw = frame.shape[:2]
    panel_w = 300
    row_h = 28
    pad = 8
    panel_h = pad * 2 + 18 + len(top3_names) * row_h
    bx = 10
    by = fh - panel_h - 55   # oberhalb der Bestenliste platzieren
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + panel_w,
                  by + panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + panel_w,
                  by + panel_h), trick_color, 1)
    cv2.putText(frame, "TRICK PREDICTION", (bx + pad, by + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, trick_color, 1)
    for i, (name, prob) in enumerate(zip(top3_names, top3_probs)):
        row_y = by + 20 + i * row_h
        label = name.replace("_", " ")
        if len(label) > 22:
            label = label[:22]
        # Ampelfarben: grün = sicher, türkis = mittel, rötlich = unsicher
        if prob >= 0.70:
            bar_col = (60, 220, 60)
        elif prob >= 0.40:
            bar_col = (50, 200, 220)
        else:
            bar_col = (90, 90, 210)
        cv2.putText(frame, label, (bx + pad, row_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)
        pct_str = f"{prob:.0%}"
        cv2.putText(frame, pct_str,
                    (bx + panel_w - 42, row_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_col, 1)
        # Balken: erst die graue Spur, dann der farbige Füllstand darüber
        bar_x = bx + pad
        bar_y = row_y + 17
        bar_w = panel_w - pad * 2 - 46
        bar_h2 = 6
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h2), (50, 50, 50), -1)
        fill_w = max(0, int(bar_w * prob))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h2), bar_col, -1)


# ── Zähler der Tricks dieser Session ──────────────────────────────────────────
def draw_rep_counter(frame, session_counts):
    """Feld oben links: wie oft welcher Trick in dieser Session erkannt wurde.
    Es werden nur die sechs häufigsten angezeigt, sonst würde das Feld das
    halbe Bild verdecken."""
    if not session_counts:
        return
    sorted_tricks = sorted(session_counts.items(), key=lambda x: x[1],
                           reverse=True)[:6]
    panel_w = 230
    row_h = 18
    panel_h = 20 + len(sorted_tricks) * row_h + 6
    bx, by = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + panel_w,
                  by + panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + panel_w,
                  by + panel_h), (180, 180, 0), 1)
    cv2.putText(frame, "SESSION TRICKS", (bx + 5, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 0), 1)
    for i, (trick, count) in enumerate(sorted_tricks):
        label = trick.replace("_", " ")
        if len(label) > 20:
            label = label[:20]
        y = by + 20 + i * row_h
        # Balkenlänge relativ zum häufigsten Trick (der bekommt volle Breite)
        bar_w = int((panel_w - 12) * min(count /
                    max(max(v for _, v in sorted_tricks), 1), 1.0))
        cv2.rectangle(frame, (bx + 4, y + 1), (bx + 4 + bar_w, y + row_h - 2),
                      (40, 80, 40), -1)
        cv2.putText(frame, f"{label}  x{count}", (bx + 6, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)


# ── Ergebniskarte am Videoende ───────────────────────────────────────────────
def draw_end_card(frame, trick_name, confidence,
                  total_airtime, max_airtime, trick_color):
    """Bildschirmfüllende Abschlusskarte nach dem Ende des Videos."""
    fh, fw = frame.shape[:2]
    frame[:] = (12, 12, 12)    # gesamtes Bild einfärben (alles überschreiben)

    # Farbige Streifen oben und unten
    cv2.rectangle(frame, (0, 0), (fw, 6), trick_color, -1)
    cv2.rectangle(frame, (0, fh - 6), (fw, fh), trick_color, -1)

    # Kleine Überschrift
    header = "TRICK DETECTED"
    (hw, _), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(frame, header, ((fw - hw) // 2, fh // 2 - 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)

    # Großer Trickname
    display = trick_name.replace("_", " ").upper()
    scale = 1.6
    thick = 3
    # Schrift so lange verkleinern, bis sie ins Bild passt (Mindestgröße 0.6),
    # damit auch sehr lange Klassennamen nicht abgeschnitten werden.
    while True:
        (tw, th), _ = cv2.getTextSize(
            display, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        if tw < fw - 60 or scale < 0.6:
            break
        scale -= 0.05
    tx = (fw - tw) // 2
    ty = fh // 2 - 60 + th // 2
    # Schatten
    cv2.putText(frame, display, (tx + 4, ty + 4),
                cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, display, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, scale, trick_color, thick)

    # Waagerechte Trennlinie in abgedunkelter Trickfarbe
    div_y = ty + 24
    cv2.line(frame, (fw // 6, div_y), (fw * 5 // 6, div_y),
             tuple(max(0, c - 80) for c in trick_color), 1)

    # Kennzahlenreihe: vier gleich breite Spalten
    stats = [
        ("CONFIDENCE",  f"{confidence:.0%}"),
        ("TOTAL AIR",   f"{total_airtime:.2f}s"),
        ("BEST AIR",    f"{max_airtime:.2f}s"),
        ("SCORE",       f"{int(confidence * 1000 + total_airtime * 200)}"),
    ]
    n_stats = len(stats)
    col_w = fw // n_stats
    for i, (label, value) in enumerate(stats):
        cx = i * col_w + col_w // 2      # Mitte der jeweiligen Spalte
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        (vw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)
        cv2.putText(frame, label, (cx - lw // 2, div_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)
        cv2.putText(frame, value, (cx - vw // 2, div_y + 72),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, trick_color, 2)

    # Fußzeile
    hint = "Press any key to close"
    (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, hint, ((fw - hint_w) // 2, fh - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1)


# ── Startbildschirm ───────────────────────────────────────────────────────────
def draw_splash_screen(frame, model_name, class_count, device_str,
                       label_names=None):
    """Zeichnet einen Informationsbildschirm direkt in `frame` (verändert das
    übergebene Bild, gibt nichts zurück).

    Zeigt beim Start, welches Modell geladen wurde, wie viele Klassen es kennt
    und auf welcher Recheneinheit es läuft — praktisch zur Kontrolle vor einer
    Vorführung."""
    fh, fw = frame.shape[:2]
    frame[:] = (15, 15, 15)
    title = "SKATEBOARD TRICK DETECTOR"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.95, 2)
    cv2.putText(frame, title, ((fw - tw) // 2, fh // 2 - 90),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 220, 255), 2)
    # Waagerechte Trennlinie
    cv2.line(frame, (fw // 4, fh // 2 - 72), (fw * 3 // 4, fh // 2 - 72),
             (60, 60, 60), 1)
    info_lines = [
        f"Model  : {os.path.basename(model_name)}",
        f"Classes: {class_count}",
        f"Device : {device_str}",
        "",
        "Press  q = quit   p = pause   r = reset",
    ]
    # Jede Zeile einzeln messen und mittig setzen
    for i, line in enumerate(info_lines):
        (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(frame, line, ((fw - lw) // 2, fh // 2 - 48 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1)
    # Liste aller Trickklassen, auf drei Spalten verteilt
    if label_names:
        cv2.putText(frame, "Detected trick classes:",
                    (fw // 2 - 100, fh // 2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 120), 1)
        col_n = 3
        # Aufrunden per Ganzzahlrechnung: (a + b - 1) // b
        items_per_col = (len(label_names) + col_n - 1) // col_n
        col_w = fw // col_n
        for idx_l, name in enumerate(label_names):
            col = idx_l // items_per_col     # Spaltennummer
            row = idx_l % items_per_col      # Zeile in dieser Spalte
            x = col * col_w + 20
            y = fh // 2 + 110 + row * 18
            if y < fh - 10:                  # nur zeichnen, wenn im Bild
                cv2.putText(frame, f"• {name.replace('_', ' ')}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 210, 130), 1)


def draw_keypoints_filtered(frame, kps_xy, kps_conf, color, phase="ground"):
    """Zeichnet das Skelett — aber nur Punkte oberhalb von KP_CONF_THRESH.
    Die Farbe von Knochen und Gelenken richtet sich nach der Bewegungsphase.

    Die Filterung ist wichtig: YOLO rät verdeckte Gelenke und legt sie oft an
    unsinnige Stellen. Ungefiltert würden wilde Linien quer durchs Bild laufen."""
    if kps_xy is None:
        return
    phase_color = PHASE_COLORS.get(phase, color)
    # Welche Punkte durch eine Linie ("Knochen") verbunden werden.
    # Die Zahlen sind Indizes in der COCO-Liste ganz oben.
    SKELETON_PAIRS = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Schultern/Arme
        (5, 11), (6, 12), (11, 12),             # Rumpf
        (11, 13), (13, 15), (12, 14), (14, 16),   # Beine
        (0, 1), (0, 2), (1, 3), (2, 4),           # Gesicht
    ]
    n = len(kps_xy)
    # Erst die Knochen …
    for a, b in SKELETON_PAIRS:
        if a >= n or b >= n:
            continue
        ca = float(kps_conf[a]) if kps_conf is not None else 1.0
        cb = float(kps_conf[b]) if kps_conf is not None else 1.0
        # BEIDE Endpunkte müssen sicher sein, sonst keine Linie
        if ca < KP_CONF_THRESH or cb < KP_CONF_THRESH:
            continue
        x1, y1 = int(kps_xy[a][0]), int(kps_xy[a][1])
        x2, y2 = int(kps_xy[b][0]), int(kps_xy[b][1])
        if x1 == 0 and y1 == 0:
            continue
        cv2.line(frame, (x1, y1), (x2, y2), phase_color, 2)
    # … dann die Gelenkpunkte darüber
    for i, (kx, ky) in enumerate(kps_xy):
        kc = float(kps_conf[i]) if kps_conf is not None else 1.0
        if kc < KP_CONF_THRESH or (kx == 0 and ky == 0):
            continue
        cv2.circle(frame, (int(kx), int(ky)), 4, phase_color, -1)


def process_video(source, model_size="s", conf_thresh=0.25, iou_thresh=0.45,
                  save_output=False, output_path="output.mp4", log_csv=False,
                  trick_model_path=None, label_map_path=None,
                  highlights_dir=None, db_path=None, ref_poses_path=None):
    """
    DIE HAUPTFUNKTION: Verarbeitet ein komplettes Video von Anfang bis Ende.

    Mehrpersonen-Posenverfolgung mit:
      - konfidenzgefiltertem, phasengefärbtem Skelett
      - je Person: Tempo, Kniewinkel, Sprungwinkel, Flugzeit, Trickerkennung
      - trainiertem BiLSTM-Trickklassifikator (optional, via --trick-model)
      - geglätteter Boardhöhe, Landequalität, Punktevergabe
      - Bewegungsspuren, Echtzeitdiagrammen für Tempo/Höhe
      - Zeitlupenausschnitten, SQLite-Datenbank, Bestenliste, Erfolgen
      - CSV-Datenprotokoll
    Steuerung: q = beenden  p = Pause  r = Bodenlinie zurücksetzen
    """
    # ── Zwei getrennte YOLO-Modelle laden ──
    pose_model = YOLO(f"yolov8{model_size}-pose.pt")   # Körperpunkte
    det_model = YOLO(f"yolov8{model_size}.pt")         # Objekte (Skateboard)

    # ── Optionaler trainierter Trickklassifikator ──
    trick_clf = None
    idx_to_label = {}
    # Schnellste verfügbare Recheneinheit wählen
    clf_device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    if trick_model_path and label_map_path:
        if os.path.isfile(trick_model_path) and os.path.isfile(label_map_path):
            trick_clf, idx_to_label = load_classifier(
                trick_model_path, label_map_path, clf_device)
            print(
                f"[INFO] Loaded trick classifier: {list(idx_to_label.values())}")
        else:
            # Ohne Modell läuft alles Übrige weiter — nur ohne KI-Trickerkennung
            print(
                "[WARN] --trick-model or --label-map file not found; using rule-based detection.")

    # ── Videoquelle öffnen ──
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30    # "or 30": Webcams melden oft 0

    # ── Optional: annotiertes Video mitschreiben ──
    writer = None
    if save_output:
        # fourcc = Codec-Kennung; "mp4v" passt zur .mp4-Endung
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {output_path}")

    # ── CSV-Protokoll einrichten ──
    csv_file = csv_writer = None
    if log_csv:
        log_path = os.path.splitext(output_path)[0] + "_data.csv"
        csv_file = open(log_path, "w", newline="")   # newline="": keine Leerzeilen unter Windows
        csv_writer = csv.writer(csv_file)
        # Kopfzeile — diese Datei lässt sich direkt in Excel oder pandas öffnen
        csv_writer.writerow([
            "frame", "player_id", "conf",
            "ankle_y_px", "knee_angle_left_deg", "knee_angle_right_deg",
            "speed_kmh", "skate_height_px", "skate_height_pct",
            "skate_max_height_px", "airtime_s", "jump_angle_deg", "trick",
        ])
        print(f"[INFO] Logging data to: {log_path}")

    # ── SQLite-Session-Datenbank ──
    db_conn = None
    if db_path:
        import session_db
        db_conn = session_db.init_db(db_path)
        print(f"[INFO] Session DB: {db_path}")

    # ── Ordner für Highlight-Clips ──
    if highlights_dir:
        os.makedirs(highlights_dir, exist_ok=True)

    # ── Musterhaltungen (für die Haltungsnote) ──
    reference_poses = {}
    if ref_poses_path and os.path.isfile(ref_poses_path):
        with open(ref_poses_path) as f:
            reference_poses = json.load(f)

    # Ringpuffer für Zeitlupen: Tracking-ID → deque der letzten Rohbilder
    slomo_buffers: dict = {}
    slomo_fps_cap = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    slomo_buf_len = int(slomo_fps_cap * SLO_MO_BUFFER_SECS)

    # ── Startbildschirm (3 Sekunden) ─────────────────────────────────────────
    if idx_to_label:
        splash = np.zeros((height, width, 3), dtype=np.uint8)   # schwarzes Bild
        label_names_list = [idx_to_label[i] for i in range(len(idx_to_label))]
        draw_splash_screen(splash, trick_model_path or "(none)",
                           len(idx_to_label), str(clf_device).upper(),
                           label_names_list)
        cv2.imshow("YOLOv8 Multi-Player Pose", splash)
        cv2.waitKey(3000)     # 3000 ms warten (0 würde auf Tastendruck warten)

    print("[INFO] Press 'q' quit | 'p' pause/resume | 'r' reset ground baseline.")
    paused = False
    frame_idx = 0
    ground_y = None       # Bodenlinie; wird aus den Board-Erkennungen gelernt
    # Tracking-ID → laufende Werte (wird in draw_top_right_player_panels gepflegt)
    player_state = {}
    session_counts: dict = {}   # Trickname → Anzahl Erkennungen in dieser Session
    # Sammler für das Gesamturteil auf der Endkarte
    global_vote_buf: list = []
    global_prob_buf: list = []
    global_stats: dict = {"total_airtime": 0.0, "max_airtime": 0.0}

    # Messung der Bildrate
    fps_timer = time.time()
    fps_display = 0.0
    fps_counter = 0

    # ══ HAUPTSCHLEIFE: ein Durchlauf pro Videobild ═══════════════════════════
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video.")
                break

            frame_idx += 1
            fps_counter += 1

            # Zeitbasis für Flugzeitmessung: bei Dateien die Videozeit
            setze_video_uhr(None if isinstance(source, int)
                            else frame_idx / max(fps, 1))

            # ── Posenverfolgung ──
            # .track() vergibt stabile IDs über die Bilder hinweg;
            # persist=True behält die Zuordnung von Bild zu Bild bei.
            pose_results = pose_model.track(
                frame, persist=True, verbose=False,
                conf=conf_thresh, iou=iou_thresh
            )

            # Grundbild ohne YOLOs eigene Beschriftungen — wir zeichnen selbst
            annotated = pose_results[0].plot(
                boxes=False, labels=False, kpt_line=False, kpt_radius=0)

            boxes = pose_results[0].boxes
            keypoints = pose_results[0].keypoints

            player_data_list = []

            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Tracking-ID; fehlt sie, Ersatznummer i+1 verwenden
                    track_id = int(box.id[0].item()
                                   ) if box.id is not None else (i + 1)
                    color = get_player_color(track_id)
                    conf = box.conf[0].item()
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                    # Gelenkpunkte dieser Person
                    kps_xy = None
                    kps_conf = None
                    if keypoints is not None and keypoints.xy is not None and i < len(keypoints.xy):
                        kps_xy = keypoints.xy[i].cpu().numpy()
                        kps_conf = keypoints.conf[i].cpu().numpy(
                        ) if keypoints.conf is not None else None

                    # Dünner farbiger Rahmen
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)
                    # Kleines ID-Schildchen oben links an der Box
                    _pill_lbl = f"P{track_id:03d}"
                    (_pw, _ph), _ = cv2.getTextSize(
                        _pill_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                    cv2.rectangle(annotated, (bx1, by1 - _ph - 8),
                                  (bx1 + _pw + 10, by1), color, -1)
                    cv2.putText(annotated, _pill_lbl,
                                (bx1 + 5, by1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1)

                    # Skelett — während eines Trick-Aufblitzens in Trickfarbe
                    _pst = player_state.get(track_id, {})
                    _phase = _pst.get("phase", "ground")
                    _skel_color = (_pst.get("trick_flash_color", color)
                                   if _pst.get("trick_flash_frames", 0) > 0
                                   else color)
                    draw_keypoints_filtered(
                        annotated, kps_xy, kps_conf, _skel_color, _phase)

                    # ── Trainierter Klassifikator: Zeitfenster je Person pflegen ──
                    clf_label = None
                    if trick_clf is not None:
                        # Hier wird der Zustand ggf. bereits angelegt, damit der
                        # Zeitfenster-Puffer existiert, bevor die Kartenfunktion
                        # später denselben Eintrag weiterverwendet.
                        pst = player_state.setdefault(track_id, {
                            "max_height_px":      0,
                            "airborne_start":     None,
                            "last_airtime":       0.0,
                            "max_airtime":        0.0,
                            "ankle_history":      deque(maxlen=10),
                            "speed_kmh":          0.0,
                            "jump_start_ankle":   None,
                            "jump_peak_ankle":    None,
                            "jump_angle":         None,
                            "height_buf":         deque(maxlen=HEIGHT_SMOOTH_N),
                            "trick_label":        None,
                            "prev_h_px":          0,
                            "trail_history":      deque(maxlen=TRAIL_LENGTH),
                            "speed_history":      deque(maxlen=GRAPH_HISTORY),
                            "height_history":     deque(maxlen=GRAPH_HISTORY),
                            "session_score":      0,
                            "trick_count":        0,
                            "trick_streak":       0,
                            "last_landing_score": None,
                            "landing_knee_buf":   [],
                            "landing_frames_left": 0,
                            "new_height_pb":      False,
                            "new_airtime_pb":     False,
                            "stance_votes":       deque(maxlen=30),
                            "stance":             None,
                            "phase":              "ground",
                            "pending_badge":      None,
                            "badge_until":        0.0,
                            "coaching_hint":      None,
                            "bail_detected":      False,
                            "vid_height":         frame.shape[0],
                        })
                        # Ringpuffer der letzten 30 normalisierten Posen
                        buf = pst.setdefault(
                            "trick_window", deque(maxlen=TRICK_WINDOW))
                        vec = normalize_keypoints(kps_xy, kps_conf)
                        buf.append(vec if vec is not None else np.zeros(
                            51, dtype=np.float32))

                        # Erst mit 30 Posen ist eine Vorhersage möglich
                        if len(buf) == TRICK_WINDOW:
                            seq = torch.from_numpy(
                                np.stack(list(buf)).astype(np.float32)
                            ).unsqueeze(0).to(clf_device)   # (1, 30, 51)
                            with torch.no_grad():   # kein Lernen → schneller
                                logits = trick_clf(seq)
                                probs = torch.softmax(logits, dim=1)
                                best = int(probs.argmax(1).item())
                                conf_s = float(probs[0, best].item())

                            # ── Glättung durch Abstimmung ──────────────────
                            # Einzelbildvorhersagen springen stark. Über 60
                            # Bilder abzustimmen ergibt eine ruhige Anzeige.
                            vote_buf = pst.setdefault(
                                "pred_vote_buf",
                                deque(maxlen=PRED_VOTE_BUF_LEN))
                            prob_acc = pst.setdefault(
                                "prob_acc_buf",
                                deque(maxlen=PRED_VOTE_BUF_LEN))
                            vote_buf.append(best)
                            prob_acc.append(conf_s)

                            # Häufigste Klasse + mittlere Sicherheit dieser Klasse
                            from collections import Counter
                            vote_counts = Counter(vote_buf)
                            voted_best, voted_n = vote_counts.most_common(1)[0]
                            voted_name = idx_to_label.get(voted_best, '?')
                            voted_conf = float(np.mean(
                                [p for idx_v, p in zip(vote_buf, prob_acc)
                                 if idx_v == voted_best]))

                            # Das abgestimmte Ergebnis anzeigen
                            clf_label_name = voted_name
                            clf_label = f"{voted_name} {voted_conf:.0%}"

                            # Die drei besten des AKTUELLEN Bildes (für die Balken)
                            probs_np = probs[0].cpu().numpy()
                            top3_idx = probs_np.argsort()[::-1][:3]
                            pst["top3_names"] = [idx_to_label.get(
                                int(j), '?') for j in top3_idx]
                            pst["top3_probs"] = [
                                float(probs_np[j]) for j in top3_idx]
                            pst["clf_best_idx"] = voted_best

                            # Zähler für aufeinanderfolgende gleiche Ergebnisse
                            if voted_name == pst.get("last_voted_name"):
                                pst["stable_frames"] = pst.get(
                                    "stable_frames", 0) + 1
                            else:
                                pst["stable_frames"] = 1
                                pst["last_voted_name"] = voted_name

                            # Einblenden erst, wenn der Trick N Bilder stabil ist
                            # UND die Sicherheit hoch genug — verhindert Flackern
                            if (pst["stable_frames"] >= STABLE_FRAMES_MIN
                                    and voted_conf >= TRICK_FLASH_CONF_MIN):
                                if voted_name != pst.get("trick_flash_name"):
                                    pst["trick_flash_name"] = voted_name
                                    pst["trick_flash_frames"] = TRICK_FLASH_FRAMES
                                    pst["trick_flash_color"] = TRICK_CLASS_COLORS[
                                        voted_best % len(TRICK_CLASS_COLORS)]
                                    # Sessionzähler für diesen Trick erhöhen
                                    session_counts[voted_name] = \
                                        session_counts.get(voted_name, 0) + 1
                        pst["clf_label"] = clf_label

                    # Alle Daten dieser Person gebündelt ablegen
                    player_data_list.append({
                        "track_id":  track_id,
                        "color":     color,
                        "conf":      conf,
                        "box":       (bx1, by1, bx2, by2),
                        "kps_xy":    kps_xy,
                        "kps_conf":  kps_conf,
                        "clf_label": clf_label,
                    })

            # ── Zähler des Trick-Aufblitzens herunterzählen ──
            for pdata in player_data_list:
                tid = pdata["track_id"]
                st = player_state.get(tid, {})
                if st.get("trick_flash_frames", 0) > 0:
                    st["trick_flash_frames"] -= 1

            # ── Skateboard-Erkennung ──
            skate_boxes = []
            det_results = det_model(
                frame, classes=[SKATEBOARD_CLASS_ID], verbose=False,
                conf=conf_thresh, iou=iou_thresh
            )
            if det_results[0].boxes is not None and len(det_results[0].boxes) > 0:
                for sdet in det_results[0].boxes:
                    sx1, sy1, sx2, sy2 = map(int, sdet.xyxy[0])
                    sconf = sdet.conf[0].item()
                    skate_boxes.append((sx1, sy1, sx2, sy2))
                    cv2.rectangle(annotated, (sx1, sy1),
                                  (sx2, sy2), COLOR_SKATEBOARD, 2)
                    cv2.putText(annotated, f"Board {sconf:.2f}", (sx1, sy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_SKATEBOARD, 2)
                    # BODENLINIE LERNEN: der tiefste je gesehene Punkt des Boards.
                    # Y wächst nach unten, deshalb max() statt min().
                    ground_y = sy2 if ground_y is None else max(ground_y, sy2)

            # Jedes Board der nächstgelegenen Person zuordnen
            skate_assignment = assign_skateboards_to_players(
                player_data_list, skate_boxes)

            # ── Karten oben rechts (aktualisiert player_state direkt) ──
            draw_top_right_player_panels(
                annotated, player_data_list, skate_assignment,
                height, ground_y, player_state
            )

            # ── Gesamtwerte für die Endkarte sammeln ──────────────────────
            for pdata in player_data_list:
                tid = pdata["track_id"]
                st = player_state.get(tid, {})
                # Flugzeit aufsummieren
                if st.get("phase") == "air" and st.get("airborne_start"):
                    _air_now = jetzt() - st["airborne_start"]
                    global_stats["total_airtime"] = (
                        global_stats.get("total_airtime", 0.0) + 1.0 / max(fps_display, 1))
                    global_stats["max_airtime"] = max(
                        global_stats.get("max_airtime", 0.0), _air_now)
                # Stimmen für das endgültige Trickurteil sammeln
                vb = st.get("pred_vote_buf")
                pb = st.get("prob_acc_buf")
                if vb:
                    for _idx_v, _p in zip(vb, pb if pb else []):
                        global_vote_buf.append(_idx_v)
                        global_prob_buf.append(float(_p))

            # ── Bildrate ──
            # Nur einmal pro Sekunde neu berechnen, sonst zappelt die Anzeige
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()
            draw_fps(annotated, fps_display)

            # ── Zeitlupen-Ringpuffer: Rohbilder je Person vorhalten ──
            for pdata in player_data_list:
                tid = pdata["track_id"]
                if tid not in slomo_buffers:
                    slomo_buffers[tid] = deque(maxlen=slomo_buf_len)
                # .copy() ist nötig: sonst würden alle Einträge auf dasselbe,
                # ständig überschriebene Bild zeigen.
                slomo_buffers[tid].append(frame.copy())

            # ── Bewegungsspuren ──
            for pdata in player_data_list:
                tid = pdata["track_id"]
                st = player_state.get(tid, {})
                draw_trails(annotated, st.get(
                    "trail_history", deque()), pdata["color"])

            # ── Punkte-Banner – während der Wiedergabe unterdrückt (erscheint auf der Endkarte) ──
            now_t = time.time()

            # ── In die Datenbank schreiben ──
            if db_conn:
                import session_db
                for pdata in player_data_list:
                    tid = pdata["track_id"]
                    st = player_state.get(tid, {})
                    session_db.log_frame(db_conn, frame_idx, tid, {
                        "speed_kmh":   st.get("speed_kmh", 0),
                        "height_px":   st.get("max_height_px", 0),
                        "airtime_s":   st.get("last_airtime", 0),
                        "trick_label": st.get("trick_label") or "",
                        "session_score": st.get("session_score", 0),
                        "phase":       st.get("phase", ""),
                        "stance":      st.get("stance") or "",
                        "bail":        int(st.get("bail_detected", False)),
                    })

            # ── CSV-Protokoll ──
            if csv_writer:
                for pdata in player_data_list:
                    tid = pdata["track_id"]
                    st = player_state.get(tid, {})
                    kxy = pdata["kps_xy"]
                    kc = pdata["kps_conf"]
                    ank = get_ankle_y(kxy, kc)
                    kl = compute_knee_angle(kxy, kc, "left")
                    kr = compute_knee_angle(kxy, kc, "right")
                    sbox = skate_assignment.get(tid)
                    # Mittelwert des Höhenpuffers; die and/or-Kette liefert 0,
                    # wenn der Puffer leer ist.
                    h_px = st.get("height_buf") and int(
                        sum(st["height_buf"]) / max(1, len(st["height_buf"]))) or 0
                    h_pct = (h_px / height * 100) if height > 0 else 0
                    airt = 0.0
                    if st.get("airborne_start"):
                        airt = jetzt() - st["airborne_start"]
                    # Felder ohne gültigen Wert bleiben leer ("") statt 0 —
                    # so lässt sich später "nicht gemessen" von "null" trennen.
                    csv_writer.writerow([
                        frame_idx, f"P{tid:03d}", f"{pdata['conf']:.3f}",
                        f"{ank:.1f}" if ank else "",
                        f"{kl:.1f}" if kl else "",
                        f"{kr:.1f}" if kr else "",
                        f"{st.get('speed_kmh', 0):.2f}",
                        h_px if sbox else "",
                        f"{h_pct:.1f}" if sbox else "",
                        st.get("max_height_px", "") if sbox else "",
                        f"{airt:.3f}" if sbox else "",
                        f"{st.get('jump_angle'):.1f}" if st.get(
                            "jump_angle") else "",
                        st.get("trick_label") or "",
                    ])

            if save_output and writer:
                writer.write(annotated)

            cv2.imshow("YOLOv8 Multi-Player Pose", annotated)

        # ── Tastatursteuerung ──
        # waitKey(1) wartet 1 ms auf eine Taste — nötig, damit OpenCV das Fenster
        # überhaupt zeichnet. & 0xFF blendet höherwertige Bits aus (Plattformfrage).
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit.")
            break
        elif key == ord("p"):
            paused = not paused      # umschalten
            print("[INFO] Paused." if paused else "[INFO] Resumed.")
        elif key == ord("r"):
            # Alles zurücksetzen: nützlich, wenn eine Fehlerkennung die
            # Bodenlinie verdorben hat.
            ground_y = None
            player_state = {}
            session_counts = {}
            print("[INFO] Ground baseline and all player stats reset.")

    # ── Aufräumen: alle geöffneten Ressourcen schließen ──
    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
        print("[INFO] CSV log saved.")
    if db_conn:
        db_conn.close()
        print(f"[INFO] Session DB closed: {db_path}")

    # ── Ergebniskarte am Videoende ───────────────────────────────────────────
    if trick_clf is not None and global_vote_buf:
        # Lokale Kurznamen, um Namenskonflikte mit dem übrigen Code zu vermeiden
        from collections import Counter as _Counter
        import numpy as _np_ec
        # Welcher Trick bekam über das GANZE Video die meisten Stimmen?
        vc = _Counter(global_vote_buf)
        final_best, _ = vc.most_common(1)[0]
        final_name = idx_to_label.get(final_best, "Not Recognized")
        # Mittlere Sicherheit, aber nur über die Stimmen für den Sieger
        final_conf = float(_np_ec.mean(
            [p for iv, p in zip(global_vote_buf, global_prob_buf)
             if iv == final_best])) if global_prob_buf else 0.0
        # Unter 30 % lieber ehrlich "nicht erkannt" melden als falsch behaupten
        if final_conf < 0.30:
            final_name = "Not Recognized"
            fin_color = (100, 100, 100)
        else:
            fin_color = TRICK_CLASS_COLORS[final_best %
                                           len(TRICK_CLASS_COLORS)]
        ec_h, ec_w = height, width
        end_card = np.zeros((ec_h, ec_w, 3), dtype=np.uint8)
        draw_end_card(end_card, final_name, final_conf,
                      global_stats["total_airtime"],
                      global_stats["max_airtime"],
                      fin_color)
        cv2.imshow("YOLOv8 Multi-Player Pose", end_card)
        print(f"[RESULT] Trick: {final_name}  conf={final_conf:.0%}  "
              f"air={global_stats['total_airtime']:.2f}s")
        cv2.waitKey(0)   # 0 = unbegrenzt warten, bis eine Taste gedrückt wird

    cv2.destroyAllWindows()


# ── Kommandozeilen-Schnittstelle ──────────────────────────────────────────────
# Dieser Block läuft NUR beim direkten Start (python main.py). Beim Import
# durch viewer.py wird er übersprungen — deshalb ist der Import gefahrlos.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 Multi-Player Skateboard Pose Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",  type=str,   default="0",
                        help="Video file path or '0' for webcam")
    parser.add_argument("--model",   type=str,   default="s",
                        choices=["n", "s", "m", "l"],
                        help="YOLOv8 model size")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Detection confidence threshold (0–1)")
    # IoU = Überlappungsschwelle, ab der zwei Erkennungen als dieselbe gelten
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold (0–1)")
    parser.add_argument("--save",    action="store_true",
                        help="Save annotated output video")
    parser.add_argument("--output",  type=str,   default="output.mp4",
                        help="Output video file path")
    parser.add_argument("--log-csv", action="store_true",
                        help="Export per-frame data log as CSV alongside output")
    parser.add_argument("--trick-model", type=str, default="features/tricks/trick_classifier.pt",
                        help="Path to trained trick_classifier.pt  (from train.py)")
    parser.add_argument("--label-map",   type=str, default="features/tricks/label_map.json",
                        help="Path to label_map.json  (from extract_features.py)")
    parser.add_argument("--highlights-dir", type=str, default=None,
                        help="Directory to save slo-mo highlight clips")
    parser.add_argument("--db",          type=str, default=None,
                        help="Path for SQLite session database (e.g. session.db)")
    parser.add_argument("--ref-poses",   type=str, default=None,
                        help="Path to reference poses JSON (for form grading)")
    args = parser.parse_args()

    # "0" → Zahl 0 (Webcam); alles andere bleibt ein Dateipfad
    source = int(args.source) if args.source.isdigit() else args.source
    process_video(
        source=source,
        model_size=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        save_output=args.save,
        output_path=args.output,
        log_csv=args.log_csv,
        trick_model_path=args.trick_model,
        label_map_path=args.label_map,
        highlights_dir=args.highlights_dir,
        db_path=args.db,
        ref_poses_path=args.ref_poses,
    )
