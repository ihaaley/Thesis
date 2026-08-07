"""
viewer.py — Desktop-Oberfläche (PyQt5) für die Skateboard-Trickerkennung
Starten mit:  python3 viewer.py

WAS IST DER UNTERSCHIED ZU main.py?
main.py zeichnet alle Informationen direkt ins Videobild (OpenCV-Fenster).
viewer.py trennt beides sauber: links läuft das Video, rechts stehen die
Messwerte in echten Bedienelementen (Karten, Knöpfe, Statusleiste). Das ist
übersichtlicher für Vorführungen und leichter zu bedienen.

WICHTIGES KONSTRUKTIONSPRINZIP — ZWEI THREADS:
Eine grafische Oberfläche muss ständig auf Klicks reagieren. Die KI-Analyse
braucht aber pro Bild viel Rechenzeit. Liefe beides im selben Thread, würde
das Fenster einfrieren ("Programm reagiert nicht").
Deshalb:
  • InferenceWorker (QThread)  → rechnet im Hintergrund
  • MainWindow                 → zeichnet nur und reagiert auf Klicks
Die Verständigung läuft über Qt-Signale (pyqtSignal). Das ist die einzige
threadsichere Art, Daten an die Oberfläche zu übergeben.

Aufbau des Fensters:
  ┌───────────────────────────────┬──────────────────────────┐
  │                               │  [PHASE]  [SPEED]        │
  │   VIDEO  (groß, linke Spalte) │  [AIRTIME]               │
  │                               ├──────────────────────────┤
  │                               │ HEIGHT │ STANCE │ FPS    │
  │                               │ BEST AIR│ SCORE │ FRAME  │
  └───────────────────────────────┴──────────────────────────┘
  │  [Open Video]  [Camera]  [▶ Start]  [⏸ Pause]  [⏹ Stop]  │
  └──────────────────────────────────────────────────────────┘
"""

import sys
import os
import time
import json
from collections import deque, Counter

import cv2
import numpy as np
import torch

# PyQt5: das Rahmenwerk für die Oberfläche
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QFileDialog,
    QSizePolicy, QStatusBar, QSlider, QSpacerItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# ── Analysefunktionen aus main.py übernehmen ─────────────────────────────────
# Die gesamte Physik- und Zeichenlogik wird wiederverwendet, statt sie ein
# zweites Mal zu schreiben. Der Import ist gefahrlos, weil main.py seinen
# Startcode hinter 'if __name__ == "__main__"' versteckt — beim Import läuft
# also keine Videoanalyse an.
from main import (
    PLAYER_COLORS, TRICK_CLASS_COLORS, TRICK_FLASH_FRAMES,
    TRICK_FLASH_CONF_MIN, PRED_VOTE_BUF_LEN, STABLE_FRAMES_MIN,
    TRICK_WINDOW, AIRBORNE_THRESH, TRAIL_LENGTH, HEIGHT_SMOOTH_N,
    SPEED_PX_TO_KMH, LANDING_STABLE_FRAMES, PHASE_COLORS,
    get_player_color, get_ankle_y, compute_knee_angle, detect_bail,
    detect_stance, detect_trick, get_phase, compute_landing_score,
    compute_trick_score, get_coaching_hint, draw_keypoints_filtered,
    draw_trails, draw_end_card,
    assign_skateboards_to_players,
    SKATEBOARD_CLASS_ID, COLOR_SKATEBOARD,
    KP_CONF_THRESH,
)
from trick_utils import normalize_keypoints, load_classifier, WINDOW_SIZE as TRICK_W

from ultralytics import YOLO  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
#  Farbpalette / Gestaltungskonstanten
#  Alle Farben zentral als Konstanten — so lässt sich das Design an einer
#  Stelle ändern, statt an 50 Stellen im Code.
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK = "#0d0d0d"        # Fensterhintergrund
BG_CARD = "#151515"        # Hintergrund der großen Karten
BG_CARD2 = "#1a1a1a"       # Hintergrund der kleinen Karten
BORDER_DIM = "#2a2a2a"     # unauffällige Rahmenfarbe
TEXT_DIM = "#555555"       # sehr blasse Schrift (Beschriftungen)
TEXT_GRAY = "#888888"      # graue Schrift (Zusatzinfos)
TEXT_WHITE = "#e8e8e8"     # normale Schrift
ACCENT_CYAN = "#00d4ff"    # Akzent Türkis
ACCENT_GRN = "#00e676"     # Akzent Grün
ACCENT_ORG = "#ff9800"     # Akzent Orange
ACCENT_YEL = "#ffd600"     # Akzent Gelb (Ergebnisse)
BTN_NORMAL = "#1e1e1e"     # Knopf normal
BTN_HOVER = "#2a2a2a"      # Knopf unter dem Mauszeiger
BTN_ACTIVE = "#0d47a1"     # hervorgehobener Knopf (Start)

PHASE_QC = {          # Bewegungsphase → Farbe (als CSS-Text für Qt)
    "ground":  "#4caf50",   # am Boden – grün
    "rising":  "#00e5ff",   # steigend – hellblau
    "peak":    "#7c4dff",   # Scheitelpunkt – violett
    "landing": "#26c6da",   # Landung – türkis
}


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: große Statistikkarte
# ─────────────────────────────────────────────────────────────────────────────
class StatCard(QFrame):
    """Eine dunkle Karte mit kleiner Beschriftung oben und großem Wert darunter.

    Wird für die drei wichtigsten Werte verwendet: Phase, Tempo, Flugzeit.
    Die farbige Linie am linken Rand kann sich zur Laufzeit ändern (z. B.
    wechselt die Phasenkarte die Farbe je nach Bewegungsphase)."""

    def __init__(self, title: str, initial: str = "--",
                 accent: str = ACCENT_CYAN, parent=None):
        super().__init__(parent)
        self._accent = accent
        # objectName erlaubt es, im Stylesheet gezielt NUR dieses Widget
        # anzusprechen (QFrame#StatCard) und nicht alle QFrames im Fenster.
        self.setObjectName("StatCard")
        self.setStyleSheet(f"""
            QFrame#StatCard {{
                background: {BG_CARD};
                border: 1px solid {BORDER_DIM};
                border-left: 3px solid {accent};
                border-radius: 4px;
            }}
        """)
        # Expanding: die Karte darf mitwachsen, wenn das Fenster größer wird
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Senkrechte Anordnung: Titel / Wert / Zusatzzeile
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self._title_lbl = QLabel(title)
        self._title_lbl.setFont(QFont("Helvetica", 9))
        self._title_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; background: transparent;")

        self._val_lbl = QLabel(initial)
        self._val_lbl.setFont(QFont("Helvetica", 22, QFont.Bold))
        self._val_lbl.setStyleSheet(
            f"color: {accent}; background: transparent;")
        self._val_lbl.setWordWrap(True)   # lange Tricknamen umbrechen

        self._sub_lbl = QLabel("")
        self._sub_lbl.setFont(QFont("Helvetica", 9))
        self._sub_lbl.setStyleSheet(
            f"color: {TEXT_GRAY}; background: transparent;")

        layout.addWidget(self._title_lbl)
        layout.addWidget(self._val_lbl)
        layout.addWidget(self._sub_lbl)

    def set_value(self, val: str, sub: str = ""):
        """Aktualisiert Hauptwert und Zusatzzeile."""
        self._val_lbl.setText(val)
        self._sub_lbl.setText(sub)

    def set_accent(self, color: str):
        """Ändert die Akzentfarbe (z. B. bei Phasenwechsel).

        Der Vergleich 'if color != self._accent' ist wichtig für die Leistung:
        Stylesheets neu zu setzen ist teuer und würde bei 30 Bildern pro
        Sekunde unnötig Rechenzeit verbrauchen."""
        if color != self._accent:
            self._accent = color
            self.setStyleSheet(f"""
                QFrame#StatCard {{
                    background: {BG_CARD};
                    border: 1px solid {BORDER_DIM};
                    border-left: 3px solid {color};
                    border-radius: 4px;
                }}
            """)
            self._val_lbl.setStyleSheet(
                f"color: {color}; background: transparent;")


class MiniCard(QFrame):
    """Kompakte zweizeilige Karte für Nebenwerte im unteren Raster
    (Höhe, beste Flugzeit, Stance, Punkte, FPS, Bildnummer)."""

    def __init__(self, title: str, initial: str = "--",
                 color: str = TEXT_GRAY, parent=None):
        super().__init__(parent)
        self.setObjectName("MiniCard")
        self.setStyleSheet(f"""
            QFrame#MiniCard {{
                background: {BG_CARD2};
                border: 1px solid {BORDER_DIM};
                border-radius: 4px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(1)

        self._t = QLabel(title)     # Beschriftung
        self._t.setFont(QFont("Helvetica", 8))
        self._t.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")

        self._v = QLabel(initial)   # Wert
        self._v.setFont(QFont("Helvetica", 14, QFont.Bold))
        self._v.setStyleSheet(f"color: {color}; background: transparent;")

        layout.addWidget(self._t)
        layout.addWidget(self._v)
        self._color = color

    def set_value(self, val: str, color: str = None):
        """Setzt den Wert; die Farbe wird nur bei echter Änderung neu gesetzt."""
        self._v.setText(val)
        if color and color != self._color:
            self._color = color
            self._v.setStyleSheet(f"color: {color}; background: transparent;")


# ─────────────────────────────────────────────────────────────────────────────
#  Hintergrund-Thread für die Analyse
# ─────────────────────────────────────────────────────────────────────────────
class InferenceWorker(QThread):
    """Führt die komplette Videoanalyse im Hintergrund aus.

    Die vier Signale sind die einzige Verbindung zur Oberfläche. Qt stellt
    sicher, dass die empfangenden Funktionen im Haupt-Thread ausgeführt werden
    — direkter Zugriff auf Widgets aus einem Nebenthread würde abstürzen."""

    frame_ready = pyqtSignal(np.ndarray, dict)   # fertiges Bild + Messwerte
    status_msg = pyqtSignal(str)                  # Text für die Statusleiste
    finished_run = pyqtSignal(dict)               # Daten für die Ergebniskarte
    error_occurred = pyqtSignal(str)              # unbehandelter Fehler

    def __init__(self, source, trick_model_path, label_map_path,
                 model_size="s", conf_thresh=0.25, iou_thresh=0.45):
        super().__init__()
        self.source = source                      # Dateipfad oder Kameranummer
        self.trick_model_path = trick_model_path
        self.label_map_path = label_map_path
        self.model_size = model_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # Steuerflaggen: werden vom Haupt-Thread gesetzt, hier laufend geprüft.
        # Einfache bool-Werte brauchen kein Lock — ein Zuweisen ist atomar.
        self._paused = False
        self._stop = False

    def pause(self):
        """Analyse anhalten (die Schleife wartet dann in einer kurzen Pause)."""
        self._paused = True

    def resume(self):
        """Analyse fortsetzen."""
        self._paused = False

    def stop(self):
        """Analyse beenden. Auch _paused zurücksetzen, sonst bliebe der Thread
        in der Pausenschleife hängen und würde nie zum Abbruch kommen."""
        self._stop = True
        self._paused = False

    def run(self):
        """Einstiegspunkt des Threads (wird von Qt beim .start() aufgerufen).

        Der try/except-Block ist entscheidend: Ein Absturz in einem QThread
        würde das ganze Programm ohne Meldung beenden. Stattdessen wird der
        Fehlertext per Signal an die Oberfläche geschickt."""
        try:
            self._run_inner()
        except Exception as exc:
            import traceback
            self.error_occurred.emit(traceback.format_exc())
            self.status_msg.emit(f"Error: {exc}")

    def _run_inner(self):
        """Die eigentliche Analyseschleife."""
        # ── Modelle laden ────────────────────────────────────────────────────
        self.status_msg.emit("Loading models…")
        pose_model = YOLO(f"yolov8{self.model_size}-pose.pt")   # findet Körperpunkte
        det_model = YOLO(f"yolov8{self.model_size}.pt")         # findet das Skateboard

        trick_clf = None
        idx_to_label = {}
        clf_device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")

        # Der Trickklassifikator ist optional: fehlt die Datei, läuft alles
        # Übrige (Tempo, Höhe, Phase) trotzdem weiter.
        if (self.trick_model_path and os.path.isfile(self.trick_model_path)
                and self.label_map_path and os.path.isfile(self.label_map_path)):
            trick_clf, idx_to_label = load_classifier(
                self.trick_model_path, self.label_map_path, device=clf_device)
            trick_clf.eval()
            self.status_msg.emit(
                f"Classifier loaded — {len(idx_to_label)} classes")
        else:
            self.status_msg.emit("Classifier not found — rule-based only")

        # ── Videoquelle öffnen ────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_msg.emit(f"Cannot open source: {self.source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30    # "or 30": Kameras melden oft 0

        self.status_msg.emit(
            f"Running  {width}×{height}  {fps:.0f} fps")

        # ── Zustand für diesen Durchlauf ──────────────────────────────────────
        player_state: dict = {}      # Tracking-ID → laufende Werte dieser Person
        ground_y = None              # Bodenhöhe in Pixeln (wird selbst gelernt)
        global_vote_buf: list = []   # alle Trick-Stimmen des ganzen Videos
        global_prob_buf: list = []   # zugehörige Sicherheiten
        global_stats = {"total_airtime": 0.0, "max_airtime": 0.0}
        frame_idx = 0
        fps_timer = time.time()
        fps_counter = 0
        fps_display = 0.0

        # ══ HAUPTSCHLEIFE: ein Durchlauf pro Videobild ═══════════════════════
        while cap.isOpened() and not self._stop:
            # ── Pause ─────────────────────────────────────────────────────────
            # Aktive Warteschleife mit kurzem Schlaf: verbraucht praktisch keine
            # Rechenzeit und reagiert innerhalb von 50 ms auf Fortsetzen/Stopp.
            while self._paused and not self._stop:
                time.sleep(0.05)
            if self._stop:
                break

            ret, frame = cap.read()
            if not ret:
                break            # Video zu Ende
            frame_idx += 1
            fps_counter += 1

            # ── Posenerkennung ────────────────────────────────────────────────
            # .track() statt .predict(): YOLO vergibt zusätzlich stabile IDs, so
            # bleibt "Person 1" über alle Bilder hinweg dieselbe Person.
            # persist=True heißt: die Verfolgung über Bilder hinweg beibehalten.
            pose_results = pose_model.track(
                frame, persist=True, verbose=False,
                conf=self.conf_thresh, iou=self.iou_thresh,
            )
            # Grundbild erzeugen, aber YOLOs eigene Beschriftungen abschalten —
            # Boxen und Skelett zeichnen wir selbst, farblich passend.
            annotated = pose_results[0].plot(
                boxes=False, labels=False, kpt_line=False, kpt_radius=0)

            boxes = pose_results[0].boxes
            keypoints = pose_results[0].keypoints

            player_data_list = []   # sammelt die Daten aller Personen dieses Bildes

            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Tracking-ID; fehlt sie (erstes Bild), Notnummer i+1 nehmen
                    track_id = int(box.id[0].item()
                                   ) if box.id is not None else (i + 1)
                    color = get_player_color(track_id)
                    conf = float(box.conf[0].item())
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])   # Eckpunkte der Box

                    # Gelenkpunkte dieser Person herausziehen
                    kps_xy = kps_conf_arr = None
                    if (keypoints is not None
                            and keypoints.xy is not None
                            and i < len(keypoints.xy)):
                        kps_xy = keypoints.xy[i].cpu().numpy()
                        kps_conf_arr = (keypoints.conf[i].cpu().numpy()
                                        if keypoints.conf is not None else None)

                    # ── Rahmen + ID-Schildchen zeichnen ──
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)
                    _pill = f"P{track_id:03d}"     # z. B. "P001"
                    # Textgröße messen, damit das farbige Feld genau passt
                    (_pw, _ph), _ = cv2.getTextSize(
                        _pill, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                    cv2.rectangle(annotated,
                                  (bx1, by1 - _ph - 8),
                                  (bx1 + _pw + 10, by1), color, -1)   # -1 = gefüllt
                    cv2.putText(annotated, _pill, (bx1 + 5, by1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1)

                    # ── Skelett zeichnen ──
                    # Während eines erkannten Tricks blinkt es in der Trickfarbe,
                    # sonst in der festen Personenfarbe.
                    _pst = player_state.get(track_id, {})
                    _phase = _pst.get("phase", "ground")
                    _scol = (_pst.get("trick_flash_color", color)
                             if _pst.get("trick_flash_frames", 0) > 0
                             else color)
                    draw_keypoints_filtered(annotated, kps_xy, kps_conf_arr,
                                            _scol, _phase)

                    # ── Trickklassifikator ──
                    clf_label = None
                    if trick_clf is not None:
                        pst = player_state.setdefault(track_id, {})
                        # Ringpuffer der letzten 30 Posen. deque mit maxlen wirft
                        # die älteste Pose automatisch raus — perfekt für ein
                        # gleitendes Zeitfenster.
                        buf = pst.setdefault(
                            "trick_window", deque(maxlen=TRICK_W))
                        vec = normalize_keypoints(kps_xy, kps_conf_arr)
                        buf.append(vec if vec is not None
                                   else np.zeros(51, dtype=np.float32))

                        # Erst wenn 30 Posen beisammen sind, kann klassifiziert werden
                        if len(buf) == TRICK_W:
                            # (30,51) → Tensor → (1,30,51): die 1 ist die
                            # Batchgröße, die das Modell erwartet.
                            seq = torch.from_numpy(
                                np.stack(list(buf)).astype(np.float32)
                            ).unsqueeze(0).to(clf_device)
                            with torch.no_grad():   # kein Lernen → schneller
                                logits = trick_clf(seq)
                                probs = torch.softmax(logits, dim=1)  # in Wahrscheinlichkeiten
                                best = int(probs.argmax(1).item())
                                conf_s = float(probs[0, best].item())

                            # ── Glättung durch Mehrheitsentscheid ──
                            # Die Einzelbildvorhersage springt stark. Über die
                            # letzten 60 Bilder abzustimmen ergibt eine ruhige,
                            # verlässliche Anzeige.
                            vote_buf = pst.setdefault(
                                "pred_vote_buf", deque(maxlen=PRED_VOTE_BUF_LEN))
                            prob_acc = pst.setdefault(
                                "prob_acc_buf", deque(maxlen=PRED_VOTE_BUF_LEN))
                            vote_buf.append(best)
                            prob_acc.append(conf_s)

                            vote_counts = Counter(vote_buf)
                            voted_best, _ = vote_counts.most_common(1)[0]  # häufigste Klasse
                            voted_name = idx_to_label.get(voted_best, "?")
                            # Durchschnittliche Sicherheit NUR der Stimmen für den Sieger
                            voted_conf = float(np.mean(
                                [p for iv, p in zip(vote_buf, prob_acc)
                                 if iv == voted_best]))

                            # Die drei wahrscheinlichsten Klassen merken
                            probs_np = probs[0].cpu().numpy()
                            top3_idx = probs_np.argsort()[::-1][:3]   # sortieren, umdrehen, erste 3
                            pst["top3_names"] = [idx_to_label.get(int(j), "?")
                                                 for j in top3_idx]
                            pst["top3_probs"] = [float(probs_np[j])
                                                 for j in top3_idx]

                            # Zählt, wie lange derselbe Trick schon führt
                            if voted_name == pst.get("last_voted_name"):
                                pst["stable_frames"] = pst.get(
                                    "stable_frames", 0) + 1
                            else:
                                pst["stable_frames"] = 1
                                pst["last_voted_name"] = voted_name

                            # Aufblitzen nur bei stabiler UND sicherer Erkennung —
                            # zwei Bedingungen gegen nervöses Flackern.
                            if (pst["stable_frames"] >= STABLE_FRAMES_MIN
                                    and voted_conf >= TRICK_FLASH_CONF_MIN):
                                if voted_name != pst.get("trick_flash_name"):
                                    pst["trick_flash_name"] = voted_name
                                    pst["trick_flash_frames"] = TRICK_FLASH_FRAMES
                                    # Modulo: verhindert einen Indexfehler, falls
                                    # es mehr Klassen als Farben gibt
                                    pst["trick_flash_color"] = TRICK_CLASS_COLORS[
                                        voted_best % len(TRICK_CLASS_COLORS)]

                            clf_label = voted_name
                            pst["clf_label"] = clf_label
                            pst["clf_conf"] = voted_conf

                    # Alles zu dieser Person gebündelt ablegen
                    player_data_list.append({
                        "track_id": track_id,
                        "color":    color,
                        "conf":     conf,
                        "box":      (bx1, by1, bx2, by2),
                        "kps_xy":   kps_xy,
                        "kps_conf": kps_conf_arr,
                        "clf_label": clf_label,
                    })

            # ── Skateboard-Erkennung ──────────────────────────────────────────
            # Zweites Modell, das nur nach der COCO-Klasse 36 (Skateboard) sucht.
            skate_boxes = []
            det_results = det_model.predict(
                frame, verbose=False,
                conf=self.conf_thresh, iou=self.iou_thresh,
                classes=[SKATEBOARD_CLASS_ID])
            if det_results and det_results[0].boxes is not None:
                for sbox in det_results[0].boxes:
                    sx1, sy1, sx2, sy2 = map(int, sbox.xyxy[0])
                    sconf = float(sbox.conf[0].item())
                    cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2),
                                  COLOR_SKATEBOARD, 2)
                    # BODENHÖHE LERNEN: Der tiefste je gesehene Punkt des Boards
                    # (größtes y) gilt als Boden. Bildkoordinaten wachsen nach
                    # unten, deshalb max() und nicht min().
                    ground_y = sy2 if ground_y is None else max(ground_y, sy2)
                    skate_boxes.append((sx1, sy1, sx2, sy2))

            # Jedes Board der nächstgelegenen Person zuordnen
            skate_assignment = assign_skateboards_to_players(
                player_data_list, skate_boxes)

            # ── Blitz-Zähler herunterzählen ──────────────────────────────────
            for pdata in player_data_list:
                pst = player_state.get(pdata["track_id"], {})
                if pst.get("trick_flash_frames", 0) > 0:
                    pst["trick_flash_frames"] -= 1

            # ── Bewegungsspuren ──────────────────────────────────────────────
            # Verblassende Punkte an Hand- und Fußgelenken zeigen den zurück-
            # gelegten Weg der letzten Bilder.
            for pdata in player_data_list:
                tid = pdata["track_id"]
                pst = player_state.get(tid, {})
                pst.setdefault("trail_history", deque(maxlen=TRAIL_LENGTH))
                pst["trail_history"].append(
                    (pdata["kps_xy"], pdata["kps_conf"]))
                draw_trails(annotated, pst["trail_history"], pdata["color"])

            # ── FPS-Leiste ────────────────────────────────────────────────────
            # Bilder pro Sekunde: einmal pro Sekunde neu berechnen (nicht pro
            # Bild), sonst würde die Zahl unlesbar zappeln.
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            # Halbtransparente Leiste unten: erst auf eine Kopie zeichnen, dann
            # beide Bilder gewichtet mischen (addWeighted) — so entsteht die
            # Durchsichtigkeit, die OpenCV sonst nicht direkt kann.
            fh_f, fw_f = annotated.shape[:2]
            bar_h = 24
            ov = annotated.copy()
            cv2.rectangle(ov, (0, fh_f - bar_h),
                          (fw_f, fh_f), (12, 12, 12), -1)
            cv2.addWeighted(ov, 0.82, annotated, 0.18, 0, annotated)
            cv2.putText(annotated, f"FPS  {fps_display:.0f}",
                        (10, fh_f - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 100), 1)

            # ── Physikwerte je Person berechnen ───────────────────────────────
            #    (dieselbe Logik wie in main.py, hier aber ohne Zeichnen —
            #     die Werte gehen an die Qt-Karten)
            primary_stats = {}   # Tracking-ID → Werte für die Oberfläche
            for pdata in player_data_list:
                tid = pdata["track_id"]
                sbox = skate_assignment.get(tid)
                pst = player_state.setdefault(tid, {})

                # Standardwerte anlegen. setdefault schreibt nur, wenn der
                # Schlüssel noch fehlt — bestehende Werte bleiben erhalten.
                pst.setdefault("max_height_px",  0)
                pst.setdefault("airborne_start", None)
                pst.setdefault("last_airtime",   0.0)
                pst.setdefault("max_airtime",    0.0)
                pst.setdefault("ankle_history",  deque(maxlen=10))
                pst.setdefault("speed_kmh",      0.0)
                pst.setdefault("height_buf",     deque(maxlen=HEIGHT_SMOOTH_N))
                pst.setdefault("stance_votes",   deque(maxlen=30))
                pst.setdefault("stance",         None)
                pst.setdefault("phase",          "ground")
                pst.setdefault("session_score",  0)
                pst.setdefault("trick_count",    0)
                pst.setdefault("last_landing_score", None)
                pst.setdefault("landing_knee_buf",   [])
                pst.setdefault("landing_frames_left", 0)
                pst.setdefault("prev_h_px",      0)
                pst.setdefault("jump_angle",     None)
                pst.setdefault("jump_start_ankle", None)
                pst.setdefault("jump_peak_ankle",  None)
                pst.setdefault("vid_height",     height)

                kps_xy = pdata["kps_xy"]
                kps_conf_arr = pdata["kps_conf"]

                # Aus den Gelenkpunkten abgeleitete Größen
                ankle_y = get_ankle_y(kps_xy, kps_conf_arr)
                knee_left = compute_knee_angle(kps_xy, kps_conf_arr, "left")
                knee_right = compute_knee_angle(kps_xy, kps_conf_arr, "right")

                # Stance (Regular/Goofy) per Mehrheit der letzten 30 Bilder —
                # eine Einzelbildmessung wäre zu unzuverlässig.
                sv = detect_stance(kps_xy, kps_conf_arr)
                if sv:
                    pst["stance_votes"].append(sv)
                if pst["stance_votes"]:
                    pst["stance"] = Counter(
                        pst["stance_votes"]).most_common(1)[0][0]

                # Tempo aus der Knöchelbewegung schätzen (Pixel → grobe km/h)
                if ankle_y is not None:
                    pst["ankle_history"].append(ankle_y)
                if len(pst["ankle_history"]) >= 2:
                    disp = abs(pst["ankle_history"][-1] -
                               pst["ankle_history"][0])
                    pst["speed_kmh"] = disp * SPEED_PX_TO_KMH

                now_t = time.time()
                h_px = h_pct = max_h_pct = 0
                airtime_s = 0.0
                airborne = False

                # Höhe, Flugzeit und Trick lassen sich NUR berechnen, wenn dieser
                # Person ein Board zugeordnet werden konnte.
                if sbox is not None:
                    _, _, _, sb_bottom = sbox
                    # Höhe = Abstand Boden ↔ Unterkante Board; nie negativ
                    raw_h = max(0, (ground_y or height) - sb_bottom)
                    pst["height_buf"].append(raw_h)
                    # Mittelwert über die letzten Bilder glättet das Zittern der
                    # Boxen-Erkennung heraus.
                    h_px = int(sum(pst["height_buf"]) / len(pst["height_buf"]))
                    h_pct = h_px / height * 100 if height > 0 else 0
                    if h_px > pst["max_height_px"]:
                        pst["max_height_px"] = h_px
                    max_h_pct = pst["max_height_px"] / \
                        height * 100 if height > 0 else 0

                    airborne = h_px > AIRBORNE_THRESH
                    pst["phase"] = get_phase(h_px, pst["prev_h_px"], airborne)

                    if airborne:
                        # ── In der Luft ──
                        if pst["airborne_start"] is None:
                            # Der Absprung: Startzeit und Ausgangshöhe merken
                            pst["airborne_start"] = now_t
                            pst["jump_start_ankle"] = ankle_y
                            pst["jump_peak_ankle"] = ankle_y
                        else:
                            # Höchsten Punkt verfolgen (kleineres y = höher im Bild)
                            if (ankle_y is not None
                                    and pst["jump_peak_ankle"] is not None
                                    and ankle_y < pst["jump_peak_ankle"]):
                                pst["jump_peak_ankle"] = ankle_y
                        airtime_s = now_t - pst["airborne_start"]
                    else:
                        # ── Am Boden ──
                        if pst["airborne_start"] is not None:
                            # Genau in diesem Bild ist die Landung passiert
                            finished = now_t - pst["airborne_start"]
                            pst["last_airtime"] = finished
                            pst["max_airtime"] = max(
                                pst["max_airtime"], finished)
                            if pst["jump_start_ankle"] and pst["jump_peak_ankle"]:
                                dy = pst["jump_start_ankle"] - \
                                    pst["jump_peak_ankle"]
                                # arctan2: Winkel aus senkrechter und waagerechter
                                # Komponente; max(1, h_px) verhindert Teilen durch 0
                                pst["jump_angle"] = float(
                                    np.degrees(np.arctan2(dy, max(1, h_px))))
                            # Beobachtungsfenster für die Landequalität öffnen
                            pst["landing_frames_left"] = LANDING_STABLE_FRAMES
                            # Sprungzustand zurücksetzen
                            pst["airborne_start"] = None
                            pst["jump_start_ankle"] = None
                            pst["jump_peak_ankle"] = None

                        # Nach der Landung einige Bilder lang die Kniewinkel
                        # sammeln: ruhige Knie = saubere Landung.
                        if pst["landing_frames_left"] > 0:
                            ka_avg = None
                            if knee_left is not None and knee_right is not None:
                                ka_avg = (knee_left + knee_right) / 2
                            elif knee_left is not None:
                                ka_avg = knee_left      # nur ein Bein sichtbar
                            elif knee_right is not None:
                                ka_avg = knee_right
                            if ka_avg is not None:
                                pst["landing_knee_buf"].append(ka_avg)
                            pst["landing_frames_left"] -= 1
                            if pst["landing_frames_left"] == 0:
                                # Fenster zu Ende → Note berechnen
                                pst["last_landing_score"] = compute_landing_score(
                                    pst["landing_knee_buf"])

                    pst["prev_h_px"] = h_px

                    # Regelbasierte Trickerkennung (unabhängig vom KI-Modell)
                    trick = detect_trick(
                        h_px, knee_left, knee_right, pst["prev_h_px"])
                    if trick:
                        # Nur bei einem NEUEN Trick Punkte vergeben, sonst würde
                        # jedes Bild eines langen Tricks erneut zählen.
                        if pst.get("trick_label") != trick:
                            land_sc = pst.get("last_landing_score") or 50
                            tscore = compute_trick_score(
                                h_px, pst.get("last_airtime", 0),
                                pst.get("jump_angle"), land_sc, height)
                            pst["session_score"] = pst.get(
                                "session_score", 0) + tscore
                            pst["trick_count"] = pst.get(
                                "trick_count",   0) + 1
                        pst["trick_label"] = trick
                    elif not airborne:
                        pst["trick_label"] = None   # zurück am Boden → zurücksetzen

                # Auszug für die Oberfläche zusammenstellen
                primary_stats[tid] = {
                    "phase":       pst["phase"],
                    "speed_kmh":   pst["speed_kmh"],
                    "airtime_s":   airtime_s,
                    "max_airtime": pst["max_airtime"],
                    "h_pct":       h_pct,
                    "max_h_pct":   max_h_pct,
                    "stance":      pst.get("stance") or "--",
                    "score":       pst.get("session_score", 0),
                    "trick_count": pst.get("trick_count", 0),
                    "clf_label":   pst.get("clf_label") or "",
                    "clf_conf":    pst.get("clf_conf", 0.0),
                    "frame_idx":   frame_idx,
                    "fps":         fps_display,
                    "has_board":   sbox is not None,
                }

                # Stimmen über das ganze Video sammeln → Grundlage der Endkarte
                vb = pst.get("pred_vote_buf")
                pb = pst.get("prob_acc_buf")
                if vb and pb:
                    for _v, _p in zip(vb, pb):
                        global_vote_buf.append(_v)
                        global_prob_buf.append(float(_p))
                if pst.get("phase") == "air" and pst.get("airborne_start"):
                    global_stats["total_airtime"] += 1.0 / max(fps_display, 1)
                    _air = now_t - pst["airborne_start"]
                    global_stats["max_airtime"] = max(
                        global_stats["max_airtime"], _air)

            # Bild und Werte an die Oberfläche schicken (threadsicher via Signal)
            self.frame_ready.emit(annotated, primary_stats)

        cap.release()   # Videodatei/Kamera freigeben

        # ── Daten für die Ergebniskarte ───────────────────────────────────────
        # Gesamturteil über das ganze Video: welcher Trick bekam die meisten
        # Stimmen über alle Bilder hinweg?
        final_trick = "Not Recognized"
        final_conf = 0.0
        if global_vote_buf:
            vote_counts = Counter(global_vote_buf)
            top_idx, _ = vote_counts.most_common(1)[0]
            final_trick = idx_to_label.get(top_idx, "Not Recognized")
            relevant = [p for iv, p in zip(global_vote_buf, global_prob_buf)
                        if iv == top_idx]
            final_conf = float(np.mean(relevant)) if relevant else 0.0
        # Unter 30 % Sicherheit lieber ehrlich "nicht erkannt" melden, als
        # einen falschen Trick zu behaupten.
        if final_conf < 0.30:
            final_trick = "Not Recognized"
            final_conf = 0.0

        end_data = {
            "trick":         final_trick,
            "conf":          final_conf,
            "total_airtime": global_stats["total_airtime"],
            "max_airtime":   global_stats["max_airtime"],
        }
        self.finished_run.emit(end_data)
        self.status_msg.emit("Done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Hauptfenster
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """Das Fenster selbst: Knöpfe, Videoanzeige, Statistikkarten.

    Es rechnet nichts — es zeigt nur an, was der InferenceWorker liefert."""

    # Standardpfade zum trainierten Modell
    DEFAULT_TRICK_MODEL = "features/tricks/trick_classifier.pt"
    DEFAULT_LABEL_MAP = "features/tricks/label_map.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skateboard Trick Detector")
        self.resize(1440, 860)
        self._worker = None          # der Hintergrund-Thread (erst beim Start erzeugt)
        self._last_end_data = None

        self._build_ui()
        self._apply_global_style()

    # ── Aufbau der Oberfläche ─────────────────────────────────────────────────
    def _build_ui(self):
        """Baut das gesamte Fenster von oben nach unten auf.

        Qt arbeitet mit 'Layouts': man legt Widgets nicht auf feste Pixel-
        positionen, sondern in Behälter, die sich beim Vergrößern des Fensters
        automatisch mit anpassen."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)          # Hauptachse: von oben nach unten
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        # ─── Werkzeugleiste oben ──────────────────────────────────────────────
        toolbar = QHBoxLayout()              # nebeneinander
        toolbar.setSpacing(6)

        self.btn_open = self._make_btn("📂  Open Video",  self._on_open_video)
        self.btn_camera = self._make_btn(
            "📷  Camera",      self._on_open_camera)
        self.btn_start = self._make_btn(
            "▶  Start",        self._on_start,  active=True)   # hervorgehoben
        self.btn_pause = self._make_btn("⏸  Pause",        self._on_pause)
        self.btn_stop = self._make_btn("⏹  Stop",         self._on_stop)

        for b in (self.btn_open, self.btn_camera, self.btn_start,
                  self.btn_pause, self.btn_stop):
            toolbar.addWidget(b)

        toolbar.addStretch()   # dehnbarer Leerraum: schiebt das Folgende nach rechts

        self._source_lbl = QLabel("No source selected")
        self._source_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px;")
        toolbar.addWidget(self._source_lbl)

        root.addLayout(toolbar)

        # ─── Hauptbereich (Video | Statistik) ─────────────────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(10)

        # Linke Spalte: das Video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 540)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            f"background: #050505; border: 1px solid {BORDER_DIM};"
            " border-radius: 4px;")
        self._show_placeholder()

        # Rechte Spalte: die Werte
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        # ─── Reihe 1: drei große Statistikkarten ──────────────────────────────
        top_cards = QHBoxLayout()
        top_cards.setSpacing(8)

        self.card_phase = StatCard("PHASE",   "--",       PHASE_QC["ground"])
        self.card_speed = StatCard("SPEED",   "--",       ACCENT_CYAN)
        self.card_airtime = StatCard("AIRTIME", "--",       ACCENT_ORG)

        for c in (self.card_phase, self.card_speed, self.card_airtime):
            top_cards.addWidget(c)

        right_col.addLayout(top_cards)

        # ─── Trennlinie ───────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)      # waagerechte Linie
        sep.setStyleSheet(f"color: {BORDER_DIM};")
        right_col.addWidget(sep)

        # ─── Reihe 2: sechs kleine Karten (2 Zeilen × 3 Spalten) ─────────────
        grid = QGridLayout()
        grid.setSpacing(8)

        self.mc_height = MiniCard("HEIGHT %",   "--",  "#00e5ff")
        self.mc_maxair = MiniCard("BEST AIR",   "--",  "#ff9800")
        self.mc_stance = MiniCard("STANCE",     "--",  "#ce93d8")
        self.mc_score = MiniCard("SCORE",      "--",  ACCENT_YEL)
        self.mc_fps = MiniCard("FPS",        "--",  ACCENT_GRN)
        self.mc_frame = MiniCard("FRAME",      "--",  TEXT_GRAY)

        # addWidget(widget, Zeile, Spalte)
        grid.addWidget(self.mc_height, 0, 0)
        grid.addWidget(self.mc_maxair, 0, 1)
        grid.addWidget(self.mc_stance, 0, 2)
        grid.addWidget(self.mc_score,  1, 0)
        grid.addWidget(self.mc_fps,    1, 1)
        grid.addWidget(self.mc_frame,  1, 2)

        right_col.addLayout(grid)
        right_col.addStretch()   # drückt das Folgende an den unteren Rand

        # ─── Trickanzeige (unter den kleinen Karten) ─────────────────────────
        self._trick_banner = QLabel("")
        self._trick_banner.setAlignment(Qt.AlignCenter)
        self._trick_banner.setFont(QFont("Helvetica", 11, QFont.Bold))
        self._trick_banner.setStyleSheet(
            f"color: {ACCENT_YEL}; background: {BG_CARD};"
            f" border: 1px solid {BORDER_DIM}; border-radius: 4px;"
            " padding: 6px;")
        self._trick_banner.setMinimumHeight(36)
        right_col.addWidget(self._trick_banner)

        # ─── Große Ergebniskarte (erst nach Videoende sichtbar) ──────────────
        self._result_card = QFrame()
        self._result_card.setObjectName("ResultCard")
        self._result_card.setStyleSheet(f"""
            QFrame#ResultCard {{
                background: #0a0a0a;
                border: 2px solid {ACCENT_YEL};
                border-radius: 8px;
            }}
        """)
        self._result_card.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._result_card.setMinimumHeight(110)
        rc_layout = QVBoxLayout(self._result_card)
        rc_layout.setContentsMargins(14, 10, 14, 10)
        rc_layout.setSpacing(4)

        self._result_sub = QLabel("DETECTED TRICK")
        self._result_sub.setAlignment(Qt.AlignCenter)
        self._result_sub.setFont(QFont("Helvetica", 9))
        self._result_sub.setStyleSheet(
            f"color: {TEXT_DIM}; background: transparent; letter-spacing: 2px;")

        self._result_name = QLabel("")
        self._result_name.setAlignment(Qt.AlignCenter)
        self._result_name.setFont(QFont("Helvetica", 24, QFont.Bold))
        self._result_name.setStyleSheet(
            f"color: {ACCENT_YEL}; background: transparent;")
        self._result_name.setWordWrap(True)

        self._result_conf = QLabel("")
        self._result_conf.setAlignment(Qt.AlignCenter)
        self._result_conf.setFont(QFont("Helvetica", 10))
        self._result_conf.setStyleSheet(
            f"color: {TEXT_GRAY}; background: transparent;")

        rc_layout.addWidget(self._result_sub)
        rc_layout.addWidget(self._result_name)
        rc_layout.addWidget(self._result_conf)

        self._result_card.hide()      # bleibt verborgen, bis ein Ergebnis vorliegt
        right_col.addWidget(self._result_card)

        # Beide Spalten zusammensetzen
        right_widget = QWidget()
        right_widget.setLayout(right_col)
        right_widget.setFixedWidth(420)   # feste Breite: Werte sollen nicht springen

        # stretch=1 gegen stretch=0: der zusätzliche Platz geht komplett ans Video
        main_row.addWidget(self.video_label, stretch=1)
        main_row.addWidget(right_widget, stretch=0)
        root.addLayout(main_row, stretch=1)

        # ─── Statusleiste am unteren Fensterrand ─────────────────────────────
        self._status = self.statusBar()
        self._status.setStyleSheet(
            f"background: {BG_CARD}; color: {TEXT_GRAY}; font-size: 11px;")
        self._status.showMessage("Ready — open a video or start camera.")

        # ── Ausgangszustand der Knöpfe ────────────────────────────────────────
        self._source = None
        self._set_buttons(running=False)

    def _show_placeholder(self):
        """Hinweistext, solange noch kein Video geladen ist."""
        lbl = self.video_label
        lbl.setText("No source\n\nOpen a video file or start camera")
        lbl.setStyleSheet(
            f"color: {TEXT_DIM}; background: #050505;"
            f" border: 1px solid {BORDER_DIM}; border-radius: 4px;"
            " font-size: 14px;")

    # ── Gestaltung ────────────────────────────────────────────────────────────
    def _apply_global_style(self):
        """Grundgestaltung für alle Widgets des Fensters."""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {BG_DARK};
                color: {TEXT_WHITE};
                font-family: Helvetica, Arial, sans-serif;
            }}
        """)

    def _make_btn(self, text, slot, active=False):
        """Erzeugt einen einheitlich gestalteten Knopf.

        'slot' ist die Funktion, die beim Klick aufgerufen wird — in Qt heißt
        dieses Muster Signal/Slot. Die Pseudozustände :hover, :pressed und
        :disabled geben dem Knopf sein Verhalten unter der Maus."""
        btn = QPushButton(text)
        btn.setFont(QFont("Helvetica", 10))
        btn.setMinimumHeight(34)
        btn.setMinimumWidth(110)
        btn.setCursor(Qt.PointingHandCursor)    # Handsymbol beim Überfahren
        base_col = BTN_ACTIVE if active else BTN_NORMAL
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {base_col};
                color: {TEXT_WHITE};
                border: 1px solid {BORDER_DIM};
                border-radius: 4px;
                padding: 0 14px;
            }}
            QPushButton:hover {{
                background: {BTN_HOVER};
                border-color: #555;
            }}
            QPushButton:pressed {{
                background: #0a2472;
            }}
            QPushButton:disabled {{
                color: {TEXT_DIM};
                background: #111;
            }}
        """)
        btn.clicked.connect(slot)   # Klick → Funktion verbinden
        return btn

    def _set_buttons(self, running: bool):
        """Schaltet Knöpfe je nach Zustand aktiv/inaktiv.

        So kann man nicht zweimal starten oder mitten im Lauf die Quelle
        wechseln — ungültige Bedienschritte werden gar nicht erst möglich."""
        self.btn_start.setEnabled(not running and self._source is not None)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)
        self.btn_open.setEnabled(not running)
        self.btn_camera.setEnabled(not running)

    # ── Reaktionen auf die Knöpfe ─────────────────────────────────────────────
    def _on_open_video(self):
        """Dateiauswahldialog öffnen und die gewählte Videodatei merken."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)")
        if path:      # leer, wenn der Dialog abgebrochen wurde
            self._source = path
            name = os.path.basename(path)
            self._source_lbl.setText(f"📽  {name}")
            self._set_buttons(running=False)
            self._status.showMessage(f"Loaded: {name}")

    def _on_open_camera(self):
        """Auf die eingebaute Kamera umschalten (OpenCV-Index 0)."""
        self._source = 0   # Standardkamera
        self._source_lbl.setText("📷  Live Camera (0)")
        self._set_buttons(running=False)
        self.btn_start.setEnabled(True)
        self._status.showMessage("Camera selected — press Start.")

    def _on_start(self):
        """Startet den Hintergrund-Thread und verbindet dessen Signale."""
        if self._source is None:
            self._status.showMessage("Select a video or camera first.")
            return
        if self._worker and self._worker.isRunning():
            return      # läuft bereits — Doppelstart verhindern
        self._reset_cards()
        self._worker = InferenceWorker(
            source=self._source,
            trick_model_path=self.DEFAULT_TRICK_MODEL,
            label_map_path=self.DEFAULT_LABEL_MAP,
        )
        # Signale des Threads mit den Anzeigefunktionen verbinden
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_msg.connect(self._on_status)
        self._worker.finished_run.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.start()        # ruft intern run() im neuen Thread auf
        self._set_buttons(running=True)

    def _on_pause(self):
        """Ein Knopf für beides: pausieren und fortsetzen (Umschalter)."""
        if self._worker:
            if self._worker._paused:
                self._worker.resume()
                self.btn_pause.setText("⏸  Pause")
                self._status.showMessage("Resumed.")
            else:
                self._worker.pause()
                self.btn_pause.setText("▶  Resume")
                self._status.showMessage("Paused.")

    def _on_stop(self):
        """Beendet den Thread sauber.

        wait(3000) gibt dem Thread bis zu 3 Sekunden, seine Schleife zu
        verlassen und die Videodatei freizugeben. Hartes Abwürgen könnte
        Ressourcen blockieren."""
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        self._set_buttons(running=False)
        self.btn_pause.setText("⏸  Pause")
        self._status.showMessage("Stopped.")

    # ── Empfänger der Thread-Signale (laufen im Haupt-Thread) ─────────────────
    def _on_frame(self, frame: np.ndarray, stats: dict):
        """Zeigt ein neues Bild an und aktualisiert alle Karten.
        Wird bis zu 30-mal pro Sekunde aufgerufen."""
        # OpenCV liefert BGR, Qt erwartet RGB → umwandeln, sonst sind
        # Rot und Blau vertauscht.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # ch * w = Bytes pro Zeile; ohne diese Angabe verzerrt Qt das Bild
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        lbl = self.video_label
        # KeepAspectRatio: kein Verzerren; SmoothTransformation: sauber skaliert
        pix = pix.scaled(lbl.size(), Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        lbl.setPixmap(pix)
        lbl.setStyleSheet(
            f"background: #050505; border: 1px solid {BORDER_DIM};"
            " border-radius: 4px;")

        if not stats:
            return      # niemand erkannt → Karten unverändert lassen

        # Die rechte Spalte zeigt immer nur die ERSTE erkannte Person.
        # (Für die Thesis-Videos mit einer fahrenden Person ausreichend.)
        tid = next(iter(stats))
        s = stats[tid]
        phase = s.get("phase", "ground")
        speed_kmh = s.get("speed_kmh", 0.0)
        airtime_s = s.get("airtime_s", 0.0)
        max_air = s.get("max_airtime", 0.0)
        h_pct = s.get("h_pct", 0.0)
        max_h_pct = s.get("max_h_pct", 0.0)
        stance = s.get("stance", "--")
        score = s.get("score", 0)
        fps_val = s.get("fps", 0.0)
        frame_idx = s.get("frame_idx", 0)
        clf_label = s.get("clf_label", "")
        clf_conf = s.get("clf_conf", 0.0)
        has_board = s.get("has_board", False)

        # Phasenkarte: Text und Farbe wechseln mit der Bewegungsphase
        phase_col = PHASE_QC.get(phase, "#888888")
        phase_lbl = {
            "ground":  "GROUND", "rising": "RISING ↑",
            "peak":    "PEAK ▲", "landing": "LANDING",
        }.get(phase, phase.upper())
        self.card_phase.set_value(phase_lbl)
        self.card_phase.set_accent(phase_col)

        # Tempokarte. "(relative)": der Wert ist ein grober Schätzwert aus
        # Pixelbewegung, keine kalibrierte Messung.
        self.card_speed.set_value(f"{speed_kmh:.1f}",
                                  sub="km/h (relative)")

        # Flugzeitkarte — nur sinnvoll, wenn ein Board erkannt wurde
        if has_board:
            self.card_airtime.set_value(
                f"{airtime_s:.2f}s",
                sub=f"best  {max_air:.2f}s")
        else:
            self.card_airtime.set_value("--", sub="no board detected")

        # Kleine Karten
        if has_board:
            self.mc_height.set_value(f"{h_pct:.1f}%")
            self.mc_maxair.set_value(f"{max_air:.2f}s")
        else:
            self.mc_height.set_value("--")
            self.mc_maxair.set_value("--")

        self.mc_stance.set_value(stance.upper() if stance else "--")
        self.mc_score.set_value(f"{score:,}")     # Tausendertrennzeichen
        self.mc_fps.set_value(f"{fps_val:.0f}")
        self.mc_frame.set_value(f"{frame_idx:,}")

        # Trickanzeige nur bei mindestens 50 % Sicherheit — lieber nichts
        # anzeigen als etwas Falsches.
        if clf_label and clf_conf >= 0.50:
            display = clf_label.replace("_", " ")
            self._trick_banner.setText(
                f"🛹  {display}  ({clf_conf:.0%})")
        else:
            self._trick_banner.setText("")

    def _on_status(self, msg: str):
        """Zeigt eine Meldung des Threads in der Statusleiste."""
        self._status.showMessage(msg)

    def _on_finished(self, end_data: dict):
        """Wird nach Videoende aufgerufen: zeigt die große Ergebniskarte."""
        self._last_end_data = end_data
        self._set_buttons(running=False)
        self.btn_pause.setText("⏸  Pause")

        trick = end_data.get("trick", "Not Recognized")
        conf = end_data.get("conf", 0.0)
        max_a = end_data.get("max_airtime", 0.0)

        if trick != "Not Recognized":
            # ── Fall 1: Trick sicher erkannt → goldene Karte ──
            display = trick.replace("_", " ")
            self._trick_banner.setText(f"🏆  {display}  ({conf:.0%})")
            self._result_name.setText(display)
            self._result_conf.setText(
                f"Confidence  {conf:.0%}   ·   Best air  {max_a:.2f}s")
            self._result_name.setStyleSheet(
                f"color: {ACCENT_YEL}; background: transparent;")
            self._result_card.setStyleSheet(f"""
                QFrame#ResultCard {{
                    background: #0a0a0a;
                    border: 2px solid {ACCENT_YEL};
                    border-radius: 8px;
                }}
            """)
            self._status.showMessage(
                f"Done — {display}  conf {conf:.0%}  best air {max_a:.2f}s")
        else:
            # ── Fall 2: unsicher → neutrale graue Karte ──
            self._trick_banner.setText("🤷  Not Recognized")
            self._result_name.setText("Not Recognized")
            self._result_conf.setText(f"Best air  {max_a:.2f}s")
            self._result_name.setStyleSheet(
                f"color: {TEXT_GRAY}; background: transparent;")
            self._result_card.setStyleSheet(f"""
                QFrame#ResultCard {{
                    background: #0a0a0a;
                    border: 2px solid {BORDER_DIM};
                    border-radius: 8px;
                }}
            """)
            self._status.showMessage(
                "Done — trick not recognized (low confidence)")
        self._result_card.show()

    def _on_worker_error(self, tb: str):
        """Zeigt Fehler des Hintergrund-Threads an, statt still abzustürzen.
        Der vollständige Fehlerbericht geht ins Terminal, die letzte (meist
        aussagekräftigste) Zeile in die Statusleiste."""
        print("=== InferenceWorker error ===")
        print(tb)
        self._set_buttons(running=False)
        self.btn_pause.setText("⏸  Pause")
        # Letzte nicht-leere Zeile des Fehlerberichts suchen
        first_line = next((l.strip()
                          for l in reversed(tb.splitlines()) if l.strip()), tb)
        self._status.showMessage(f"⚠  {first_line}")
        self._trick_banner.setText(f"⚠  Error — check terminal for details")
        self._trick_banner.setStyleSheet(
            f"color: #f44336; background: {BG_CARD};"
            f" border: 1px solid {BORDER_DIM}; border-radius: 4px; padding: 6px;")

    # ── Hilfsfunktionen ───────────────────────────────────────────────────────
    def _reset_cards(self):
        """Setzt alle Anzeigen auf '--' zurück — vor jedem neuen Lauf, damit
        keine Werte des vorherigen Videos stehen bleiben."""
        for card in (self.card_phase, self.card_speed, self.card_airtime):
            card.set_value("--")
        for mc in (self.mc_height, self.mc_maxair, self.mc_stance,
                   self.mc_score, self.mc_fps, self.mc_frame):
            mc.set_value("--")
        self._trick_banner.setText("")
        self.card_phase.set_accent(PHASE_QC["ground"])
        self._result_card.hide()

    def closeEvent(self, event):
        """Wird von Qt beim Schließen des Fensters aufgerufen.

        Ohne dieses Aufräumen liefe der Analyse-Thread weiter und der Prozess
        würde sich nicht beenden lassen."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  Programmstart
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # In den Ordner dieser Datei wechseln, damit die relativen Modellpfade
    # ("features/tricks/...") stimmen, egal von wo aus gestartet wurde.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # plattformübergreifend gleiches Aussehen

    # Dunkle Farbpalette erzwingen, damit auch native Elemente (Dateidialog,
    # Bildlaufleisten) zum dunklen Design passen.
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(13, 13, 13))
    palette.setColor(QPalette.WindowText,      QColor(232, 232, 232))
    palette.setColor(QPalette.Base,            QColor(21, 21, 21))
    palette.setColor(QPalette.AlternateBase,   QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipBase,     QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText,     QColor(232, 232, 232))
    palette.setColor(QPalette.Text,            QColor(232, 232, 232))
    palette.setColor(QPalette.Button,          QColor(30, 30, 30))
    palette.setColor(QPalette.ButtonText,      QColor(232, 232, 232))
    palette.setColor(QPalette.BrightText,      QColor(255, 255, 255))
    palette.setColor(QPalette.Link,            QColor(0, 212, 255))
    palette.setColor(QPalette.Highlight,       QColor(0, 71, 171))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    # exec_() startet die Ereignisschleife und kehrt erst beim Schließen zurück
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
