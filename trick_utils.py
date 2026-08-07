"""
trick_utils.py
--------------
ZENTRALE HILFSBIBLIOTHEK des Projekts.

Diese Datei enthält genau die Bausteine, die von MEHREREN Skripten gebraucht
werden und deshalb nur EINMAL existieren dürfen:

  1) normalize_keypoints()  – wandelt rohe Pixel-Koordinaten der 17 Körperpunkte
                              in ein kamera- und größenunabhängiges Zahlenformat um.
  2) TrickClassifier        – die Definition des neuronalen Netzes (BiLSTM).
  3) load_classifier()      – lädt ein trainiertes Modell + die Klassennamen.

WARUM ZENTRAL?
Beim Training (extract_features.py / train.py) und bei der späteren Erkennung
(main.py / viewer.py) MUSS die Vorverarbeitung 100 % identisch sein. Würde man
die Normalisierung zweimal getrennt programmieren und eine Version leicht
abweichen, bekäme das Modell im Live-Betrieb ganz andere Zahlen zu sehen als im
Training – die Erkennung wäre praktisch zufällig. Deshalb: eine Quelle, alle
importieren daraus.
"""

import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore

# ── Konfiguration des Zeitfensters (muss beim Extrahieren UND beim Erkennen gleich sein) ──
# Das Netz sieht nie ein Einzelbild, sondern immer eine kurze Bewegungssequenz.
WINDOW_SIZE = 30          # Anzahl Frames (Bilder), die ein Eingabe-Sample bilden
FEATURE_DIM = 51          # Zahlen pro Frame: 17 Gelenke × (x, y, Konfidenz) = 51

# ── Konfiguration der Normalisierung ──────────────────────────────────────────
MIN_KP_CONF      = 0.30   # Gelenke, die YOLO unsicherer als 30 % erkennt, werden genullt
L_HIP,  R_HIP    = 11, 12          # Index der linken/rechten Hüfte im COCO-Schema
L_SHOULDER, R_SHOULDER = 5, 6      # Index der linken/rechten Schulter im COCO-Schema


def normalize_keypoints(kps_xy: np.ndarray, kps_conf: np.ndarray | None) -> np.ndarray | None:
    """
    Normalisiert die 17 COCO-Körperpunkte, sodass sie unabhängig werden von:
      - der Kameraposition  (Hüftmitte wird auf den Nullpunkt verschoben)
      - der Personengröße   (Skalierung, sodass die Rumpflänge = 1.0 ist)

    HINTERGRUND:
    YOLO liefert Pixelkoordinaten. Steht die Person weiter links im Bild oder
    näher an der Kamera, ändern sich alle Zahlen massiv – obwohl die Bewegung
    dieselbe ist. Nach dieser Normalisierung beschreibt der Vektor nur noch die
    KÖRPERHALTUNG, nicht mehr Ort oder Entfernung.

    Eingabe:
      kps_xy   : np.ndarray, Form (17, 2)  — Pixelkoordinaten x/y je Gelenk
      kps_conf : np.ndarray, Form (17,)    — Sicherheit je Gelenk (0–1), oder None

    Rückgabe:
      np.ndarray der Form (51,) = [x0, y0, c0,  x1, y1, c1, ...]
      oder None, wenn die Normalisierung scheitert (z. B. Person kaum sichtbar)
    """
    # Schutz: gar keine Punkte oder unvollständiges Skelett → nicht verwertbar
    if kps_xy is None or kps_xy.shape[0] < 17:
        return None

    # Kopie anlegen (wir verändern die Werte gleich, das Original bleibt unberührt)
    kps = kps_xy[:17].copy().astype(np.float32)   # Form (17, 2)

    # ── Referenzpunkte bestimmen ──
    hip_mid      = (kps[L_HIP] + kps[R_HIP]) / 2.0            # Mittelpunkt der Hüfte
    shoulder_mid = (kps[L_SHOULDER] + kps[R_SHOULDER]) / 2.0  # Mittelpunkt der Schultern
    torso_len    = np.linalg.norm(shoulder_mid - hip_mid)     # Rumpflänge = Abstand beider

    # Ist der Rumpf kürzer als 1 Pixel, war die Erkennung fehlerhaft
    # (z. B. alle Punkte auf 0). Dann lieber verwerfen als durch ~0 teilen.
    if torso_len < 1.0:
        return None

    # ── Verschieben + Skalieren ──
    # (kps - hip_mid) → Hüftmitte liegt jetzt bei (0,0)
    # / torso_len     → Rumpf ist jetzt genau 1.0 lang, egal wie groß die Person im Bild war
    kps = (kps - hip_mid) / torso_len   # Form (17, 2)

    # ── Konfidenz-Spalte bauen und unsichere Gelenke ausblenden ──
    conf_col = np.ones((17, 1), dtype=np.float32)   # Standard: alles "sicher" (1.0)
    if kps_conf is not None:
        for i in range(17):
            c = float(kps_conf[i])
            conf_col[i, 0] = c            # echte Sicherheit als 3. Wert mitgeben
            if c < MIN_KP_CONF:
                kps[i] = 0.0              # unzuverlässiges Gelenk auf 0 setzen (maskieren)

    # ── Auf einen flachen Vektor bringen ──
    # Aus der (17,3)-Matrix wird eine Zeile mit 51 Zahlen – das erwartet das Netz.
    return np.concatenate([kps, conf_col], axis=1).flatten()  # (17,3) → (51,)


# ── Das Modell ────────────────────────────────────────────────────────────────

class TrickClassifier(nn.Module):
    """
    Bidirektionales LSTM, das eine Sequenz normalisierter Pose-Frames auf eine
    Skateboard-/Longboard-Trickklasse abbildet.

    WARUM LSTM?
    Ein Trick ist kein Standbild, sondern ein zeitlicher Ablauf (Anfahrt → Absprung
    → Luftphase → Landung). Ein LSTM ist ein Netz mit "Gedächtnis": es verarbeitet
    die 30 Frames nacheinander und behält dabei einen inneren Zustand.

    WARUM BIDIREKTIONAL?
    Das Netz liest die Sequenz zusätzlich rückwärts. Dadurch weiß es an Frame 10
    bereits, was in Frame 25 passiert – bei einer aufgezeichneten Sequenz ist das
    erlaubt und verbessert die Genauigkeit deutlich.

    Eingabe  : (batch, WINDOW_SIZE, FEATURE_DIM)  z. B. (64, 30, 51)
    Ausgabe  : (batch, num_classes)  — rohe Logits (noch keine Wahrscheinlichkeiten)
    """

    def __init__(self, num_classes: int, input_size: int = FEATURE_DIM,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # Der rekurrente Teil: verarbeitet die zeitliche Abfolge der Posen
        self.lstm = nn.LSTM(
            input_size=input_size,      # 51 Zahlen pro Zeitschritt
            hidden_size=hidden_size,    # Größe des "Gedächtnisses" je Richtung
            num_layers=num_layers,      # gestapelte LSTM-Schichten
            batch_first=True,           # Datenformat (Batch, Zeit, Merkmale)
            bidirectional=True,         # vorwärts + rückwärts lesen
            # Dropout wirkt nur ZWISCHEN Schichten – bei nur 1 Schicht sinnlos
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Der Klassifikationskopf: macht aus dem LSTM-Zustand eine Klassenentscheidung
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),        # ×2, weil vorwärts+rückwärts zusammengehängt
            nn.Linear(hidden_size * 2, 64),       # auf 64 Merkmale verdichten
            nn.ReLU(),                            # Nichtlinearität
            nn.Dropout(dropout),                  # gegen Überanpassung (Overfitting)
            nn.Linear(64, num_classes),           # finale Schicht: ein Wert je Trickklasse
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vorwärtsdurchlauf: Sequenz rein, Klassen-Scores raus."""
        out, _ = self.lstm(x)          # (B, T, hidden*2) – Ausgabe für JEDEN Zeitschritt
        out = out[:, -1, :]            # nur letzter Zeitschritt (B, hidden*2) = Zusammenfassung
        return self.head(out)          # (B, num_classes)


def load_classifier(model_path: str, label_map_path: str, device: torch.device):
    """
    Lädt ein trainiertes TrickClassifier-Modell samt Klassennamen von der Festplatte.

    Parameter:
      model_path     : Pfad zu trick_classifier.pt (die gelernten Gewichte)
      label_map_path : Pfad zu label_map.json      (Zuordnung Name ↔ Zahl)
      device         : "mps" (Apple), "cuda" (Nvidia) oder "cpu"

    Rückgabe: (model, idx_to_label_dict)
      idx_to_label_dict: {0: "carve", 1: "slide", ...}
    """
    import json
    with open(label_map_path) as f:
        raw = json.load(f)
    # label_map.json speichert {Klassenname: Index} → hier umdrehen zu {Index: Klassenname},
    # denn das Netz gibt eine Zahl aus und wir brauchen daraus den Namen.
    idx_to_label = {int(v): k for k, v in raw.items()}
    num_classes  = len(idx_to_label)

    # Leeres Modell in der richtigen Größe bauen …
    model = TrickClassifier(num_classes=num_classes)
    # … die trainierten Gewichte laden …
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    # … und in den Auswertungsmodus schalten (Dropout aus, deterministisches Verhalten).
    model.eval()
    return model, idx_to_label
