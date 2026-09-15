"""
train.py
--------
SCHRITT 3 der Pipeline: Trainiert den BiLSTM-Trickklassifikator auf den
Merkmalsdaten, die extract_features.py erzeugt hat.

GESAMTPIPELINE:
  setup_dataset.py    → Videos sortieren
  extract_features.py → Posen extrahieren  → X.npy / y.npy
  train.py            → Modell trainieren  → trick_classifier.pt   ← DIESE DATEI
  main.py / viewer.py → Modell anwenden

WAS HIER PASSIERT:
  1. Daten laden (X = Bewegungsfenster, y = zugehörige Trickklasse)
  2. Aufteilen in Trainings-, Validierungs- und Testdaten
  3. In vielen Durchläufen (Epochen) lernen; nach jeder Epoche prüfen
  4. Immer nur das BESTE Modell speichern; bei Stillstand früh abbrechen
  5. Abschlussbewertung auf den Testdaten + Diagramme und Bericht schreiben

Aufruf
-----
  python train.py --features features/
  python train.py --features features/ --epochs 100 --batch 64 --lr 1e-3

Ergebnisse (landen in <features>/)
--------
  trick_classifier.pt   — Gewichte des besten Modells (nach Validierungsgenauigkeit)
  training_curves.png   — Verlauf von Fehler (Loss) und Genauigkeit
  confusion_matrix.png  — Verwechslungsmatrix je Klasse auf den Testdaten
  training_report.txt   — vollständiger Klassifikationsbericht
"""

import argparse
import sys
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trick_utils import TrickClassifier, WINDOW_SIZE, FEATURE_DIM

# ── Reproduzierbarkeit ─────────────────────────────────────────────────────────
# Feste Startzahl für alle Zufallsprozesse (Gewichtsinitialisierung, Mischen der
# Daten). Ohne das käme bei jedem Lauf ein leicht anderes Ergebnis heraus – für
# eine wissenschaftliche Arbeit muss das Training wiederholbar sein.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Rechengerät wählen (MPS für Apple Silicon, sonst CUDA, sonst CPU) ─────────
def get_device() -> torch.device:
    """Sucht die schnellste verfügbare Recheneinheit.
    MPS  = Apple-Grafikchip (M1/M2/M3), CUDA = Nvidia-Grafikkarte, CPU = Notlösung."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Daten laden und aufteilen ──────────────────────────────────────────────────

def load_data(features_dir: Path):
    """Lädt die von extract_features.py erzeugten Dateien.

    X: Form (N, 30, 51) — N Bewegungsfenster à 30 Frames à 51 Zahlen
    y: Form (N,)        — zu jedem Fenster die richtige Klassennummer
    label_map: {"Caveman": 0, "Shuvit": 1, ...}"""
    X = np.load(features_dir / "X.npy").astype(np.float32)
    y = np.load(features_dir / "y.npy").astype(np.int64)
    with open(features_dir / "label_map.json") as f:
        label_map = json.load(f)
    print(f"[INFO] Loaded X={X.shape}  y={y.shape}  classes={len(label_map)}")
    return X, y, label_map


def stratified_split(X, y, val_ratio=0.15, test_ratio=0.15):
    """Teilt die Daten in Training / Validierung / Test auf und behält dabei die
    Klassenverhältnisse bei ("stratifiziert").

    WARUM STRATIFIZIERT?
    Bei einer rein zufälligen Aufteilung könnte eine seltene Trickklasse zufällig
    komplett im Testteil landen – dann kann das Modell sie nie lernen. Deshalb
    wird JEDE Klasse einzeln im Verhältnis 70/15/15 aufgeteilt.

    WOZU DREI TEILE?
      Training    (70 %) – daraus lernt das Modell
      Validierung (15 %) – dient während des Trainings zur Kontrolle und Modellauswahl
      Test        (15 %) – wird erst ganz am Ende einmal benutzt: das ehrliche Ergebnis
    """
    from collections import defaultdict
    # Schritt 1: Für jede Klasse die Positionen (Indizes) ihrer Beispiele sammeln
    indices_by_class = defaultdict(list)
    for i, label in enumerate(y):
        indices_by_class[int(label)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in indices_by_class.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)          # innerhalb der Klasse mischen
        n       = len(idxs)
        # max(1, ...) garantiert mindestens ein Beispiel je Teilmenge,
        # auch wenn eine Klasse sehr wenige Fenster hat.
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        test_idx .extend(idxs[:n_test])                      # erste Scheibe → Test
        val_idx  .extend(idxs[n_test:n_test + n_val])        # zweite Scheibe → Validierung
        train_idx.extend(idxs[n_test + n_val:])              # Rest → Training

    return (np.array(train_idx, dtype=int),
            np.array(val_idx,   dtype=int),
            np.array(test_idx,  dtype=int))


def clipwise_split(y, clips, val_ratio=0.15, test_ratio=0.15):
    """Teilt auf Ebene der VIDEOS auf, nicht auf Ebene der Fenster.

    WARUM?
    Benachbarte Fenster desselben Videos überlappen bei Fenstergröße 30 und
    Schrittweite 15 zur Hälfte – sie bestehen also zu 50 % aus denselben
    Bildern. Werden die Fenster zufällig aufgeteilt, landen solche fast
    identischen Fenster gleichzeitig im Training und im Test. Das Modell
    bekommt im Test dann Material zu sehen, das es bereits kennt, und die
    gemessene Genauigkeit fällt zu günstig aus.

    Hier werden stattdessen ganze Videos einer Teilmenge zugewiesen. Die
    Klassenverhältnisse bleiben dabei erhalten, weil jede Klasse einzeln
    aufgeteilt wird.
    """
    from collections import defaultdict

    # Fensterpositionen je Klasse und je Video sammeln
    je_klasse = defaultdict(lambda: defaultdict(list))
    for i, (label, clip) in enumerate(zip(y, clips)):
        je_klasse[int(label)][str(clip)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for cls, videos in je_klasse.items():
        clip_ids = list(videos.keys())
        np.random.shuffle(clip_ids)

        gesamt    = sum(len(v) for v in videos.values())
        ziel_test = gesamt * test_ratio
        ziel_val  = gesamt * val_ratio
        im_test = im_val = 0

        for n, cid in enumerate(clip_ids):
            positionen = videos[cid]
            # Das letzte Video einer Klasse geht immer ins Training – so bleibt
            # keine Klasse ohne Trainingsbeispiele, auch bei sehr wenigen Videos.
            rest = len(clip_ids) - n
            if im_test < ziel_test and rest > 2:
                test_idx.extend(positionen);  im_test += len(positionen)
            elif im_val < ziel_val and rest > 1:
                val_idx.extend(positionen);   im_val  += len(positionen)
            else:
                train_idx.extend(positionen)

    return (np.array(train_idx, dtype=int),
            np.array(val_idx,   dtype=int),
            np.array(test_idx,  dtype=int))


# ── Trainingsschleife ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Führt EINEN kompletten Durchlauf durch die Trainingsdaten aus.

    Der Kern jedes Deep Learnings, in fünf Schritten pro Datenpaket (Batch):
      1. Vorhersage berechnen           (forward)
      2. Fehler zur Wahrheit bestimmen  (loss)
      3. Ableitungen berechnen          (backward)
      4. Ableitungen begrenzen          (clipping)
      5. Gewichte anpassen              (optimizer.step)

    Rückgabe: (durchschnittlicher Fehler, Trefferquote) dieser Epoche."""
    model.train()   # Trainingsmodus: Dropout aktiv
    total_loss = total_correct = total_n = 0
    for X_batch, y_batch in loader:
        # Daten auf die Grafikkarte/CPU schieben, auf der das Modell liegt
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()          # alte Ableitungen löschen (sonst summieren sie sich)
        logits = model(X_batch)        # 1. Vorhersage
        loss   = criterion(logits, y_batch)   # 2. Wie falsch lagen wir?
        loss.backward()                # 3. Ableitungen zurückrechnen
        # 4. Gradient Clipping: begrenzt die Schrittweite. Verhindert bei LSTMs
        #    das Problem "explodierender Gradienten", bei dem das Training kippt.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()               # 5. Gewichte in Richtung weniger Fehler anpassen

        # Statistik mitführen (× Batchgröße, damit der Schnitt am Ende stimmt)
        total_loss    += loss.item() * len(y_batch)
        total_correct += (logits.argmax(1) == y_batch).sum().item()   # argmax = wahrscheinlichste Klasse
        total_n       += len(y_batch)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()   # spart Speicher/Zeit: hier wird nichts gelernt, keine Ableitungen nötig
def evaluate(model, loader, criterion, device):
    """Bewertet das Modell auf Validierungs- oder Testdaten – OHNE zu lernen.

    Rückgabe: (Fehler, Genauigkeit, alle Vorhersagen, alle wahren Labels)
    Die letzten beiden werden für Verwechslungsmatrix und Bericht gebraucht."""
    model.eval()    # Auswertungsmodus: Dropout aus
    total_loss = total_correct = total_n = 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        preds  = logits.argmax(1)
        total_loss    += loss.item() * len(y_batch)
        total_correct += (preds == y_batch).sum().item()
        total_n       += len(y_batch)
        # .cpu() ist nötig, weil numpy nicht mit GPU-Tensoren arbeiten kann
        all_preds .extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return total_loss / total_n, total_correct / total_n, np.array(all_preds), np.array(all_labels)


# ── Diagramme ─────────────────────────────────────────────────────────────────

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Zeichnet die Lernkurven: links Fehler, rechts Genauigkeit, je Epoche.

    SO LIEST MAN DIE KURVEN:
      Beide Fehlerkurven fallen        → gesundes Training
      Trainingsfehler fällt, Validierungsfehler steigt → Overfitting
                                          (das Modell lernt die Daten auswendig)"""
    import matplotlib
    matplotlib.use("Agg")   # kein Fenster öffnen, direkt in Datei zeichnen
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))   # zwei Diagramme nebeneinander
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs,   label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")


def plot_confusion(preds, labels, class_names, save_path):
    """Zeichnet die Verwechslungsmatrix (Confusion Matrix).

    LESEART: Zeile = tatsächliche Klasse, Spalte = vorhergesagte Klasse.
    Eine kräftige Diagonale bedeutet gute Erkennung. Ein heller Fleck abseits
    der Diagonale zeigt genau, WELCHE zwei Tricks das Modell verwechselt –
    für die Fehleranalyse in der Thesis der wichtigste Plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    # Zeilenweise auf 1.0 normieren → Anteile statt absoluter Zahlen.
    # Sonst wirken häufige Klassen automatisch "besser".
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Diagrammgröße wächst mit der Klassenanzahl, damit die Beschriftung lesbar bleibt
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (normalized)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main():
    # ── Alle einstellbaren Werte als Kommandozeilenparameter ──
    parser = argparse.ArgumentParser(
        description="Train BiLSTM trick classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,   # zeigt Standardwerte in --help
    )
    parser.add_argument("--features",    type=str, default="features",
                        help="Directory containing X.npy, y.npy, label_map.json")
    # Wie wird in Training / Validierung / Test aufgeteilt?
    #   window = wie bisher: einzelne Fenster werden zufaellig verteilt
    #   clip   = ganze Videos werden einer Teilmenge zugewiesen
    # Bei "window" koennen ueberlappende Fenster desselben Videos gleichzeitig
    # im Training und im Test landen, was die Testgenauigkeit zu guenstig macht.
    parser.add_argument("--split-by",    type=str,   default="window",
                        choices=["window", "clip"],
                        help="window: Fenster zufaellig verteilen (bisheriges Verhalten). "
                             "clip: ganze Videos einer Teilmenge zuweisen, "
                             "verlangt clips.npy")
    # Epoche = ein kompletter Durchlauf durch alle Trainingsdaten
    parser.add_argument("--epochs",      type=int,   default=80)
    # Batch = wie viele Beispiele gleichzeitig verarbeitet werden
    parser.add_argument("--batch",       type=int,   default=64)
    # Lernrate = Schrittweite bei der Gewichtsanpassung (zu groß: instabil, zu klein: langsam)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden",      type=int,   default=128,
                        help="LSTM hidden size")
    parser.add_argument("--layers",      type=int,   default=2,
                        help="Number of LSTM layers")
    # Dropout = Anteil zufällig deaktivierter Neuronen; erzwingt robusteres Lernen
    parser.add_argument("--dropout",     type=float, default=0.3)
    # Geduld: nach so vielen Epochen ohne Verbesserung wird abgebrochen
    parser.add_argument("--patience",    type=int,   default=15,
                        help="Early stopping patience (epochs)")
    # Weight Decay = leichte Bestrafung großer Gewichte, wirkt gegen Overfitting
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    args = parser.parse_args()

    features_dir = Path(args.features)
    device       = get_device()
    print(f"[INFO] Using device: {device}")

    # ── Daten laden ──
    X, y, label_map = load_data(features_dir)
    idx_to_label    = {v: k for k, v in label_map.items()}          # Zahl → Name
    class_names     = [idx_to_label[i] for i in range(len(label_map))]  # in Indexreihenfolge
    num_classes     = len(class_names)

    if args.split_by == "clip":
        clips_datei = features_dir / "clips.npy"
        if not clips_datei.exists():
            sys.exit("[FEHLER] --split-by clip verlangt clips.npy im Merkmalsverzeichnis.\n"
                     "         Diese Datei erzeugt extract_features_manifest.py.")
        clips = np.load(clips_datei, allow_pickle=True)
        if len(clips) != len(y):
            sys.exit(f"[FEHLER] clips.npy ({len(clips)}) passt nicht zu y.npy ({len(y)}).")
        train_idx, val_idx, test_idx = clipwise_split(y, clips)
        n_clips = len(set(clips.tolist()))
        print(f"[INFO] Aufteilung clipweise über {n_clips} Videos")
    else:
        train_idx, val_idx, test_idx = stratified_split(X, y)
        print("[INFO] Aufteilung fensterweise (überlappende Fenster können "
              "auf Training und Test verteilt werden)")
    print(f"[INFO] Split → train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")

    def make_loader(idx, shuffle=True):
        """Baut einen DataLoader: er zerlegt die Daten in Batches und mischt sie.
        Nur die Trainingsdaten werden gemischt – bei Val/Test wäre das sinnlos."""
        ds = TensorDataset(
            torch.from_numpy(X[idx]),   # numpy → PyTorch-Tensor
            torch.from_numpy(y[idx]),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=0,                       # kein Extra-Prozess (macOS-freundlich)
                          pin_memory=(device.type == "cuda"))  # schnellerer Transfer nur bei Nvidia

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # ── Modell erzeugen (Definition steht in trick_utils.py) ──
    model = TrickClassifier(
        num_classes=num_classes,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    # Anzahl trainierbarer Parameter – Kennzahl für die Modellgröße
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total_params:,}")

    # ── Klassengewichte gegen unausgewogene Datenmengen ──
    # Gibt es 1000 "Caveman"-Fenster, aber nur 100 "Shuvit", würde das Modell
    # lernen, einfach immer "Caveman" zu raten. Deshalb bekommen seltene Klassen
    # ein höheres Gewicht im Fehler: 1/(Anzahl+1). Das +1 verhindert Division durch 0.
    class_counts  = np.bincount(y[train_idx], minlength=num_classes).astype(float)
    class_weights = torch.tensor(1.0 / (class_counts + 1), dtype=torch.float32).to(device)
    # CrossEntropyLoss = Standard-Fehlermaß für Mehrklassenprobleme
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    # Adam = robuster Optimierer, passt die Schrittweite je Parameter selbst an
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Scheduler: senkt die Lernrate im Verlauf entlang einer Kosinuskurve ab —
    # anfangs große Schritte (schnell lernen), am Ende feine Schritte (nachjustieren).
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training ──
    best_val_acc  = 0.0
    best_path     = features_dir / "trick_classifier.pt"
    patience_cnt  = 0    # zählt Epochen ohne Verbesserung
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Kopfzeile der Fortschrittstabelle
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        # Eine Runde lernen, danach auf den Validierungsdaten prüfen
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()   # Lernrate für die nächste Epoche absenken

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        # ── Bestes Modell sichern + Early Stopping ──
        flag = ""
        if vl_acc > best_val_acc:
            # Neuer Bestwert → Gewichte speichern und Geduldszähler zurücksetzen.
            # Gespeichert wird IMMER nur das beste, nie das letzte Modell.
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_path)
            patience_cnt = 0
            flag = " ✓"
        else:
            patience_cnt += 1

        print(f"{epoch:>6} {tr_loss:>10.4f} {tr_acc:>9.1%} {vl_loss:>10.4f} {vl_acc:>9.1%}{flag}")

        # Keine Verbesserung mehr über viele Epochen → abbrechen.
        # Spart Rechenzeit und verhindert weiteres Overfitting.
        if patience_cnt >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Abschlussbewertung auf den Testdaten ──
    # Das beste gespeicherte Modell zurückladen (der aktuelle Stand im Speicher
    # kann schlechter sein) und EINMALIG auf den unberührten Testdaten messen.
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\n[RESULT] Best val acc: {best_val_acc:.1%}   Test acc: {test_acc:.1%}")

    # ── Berichte ──
    # classification_report liefert je Klasse Precision, Recall und F1-Wert:
    #   Precision = Wie viele der als X erkannten waren wirklich X?
    #   Recall    = Wie viele der echten X wurden gefunden?
    #   F1        = ausgewogenes Mittel aus beidem
    from sklearn.metrics import classification_report
    report = classification_report(test_labels, test_preds, target_names=class_names)
    print(f"\n{report}")
    report_path = features_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best val accuracy : {best_val_acc:.4f}\n")
        f.write(f"Test accuracy     : {test_acc:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Report saved → {report_path}")

    plot_curves(train_losses, val_losses, train_accs, val_accs,
                features_dir / "training_curves.png")
    plot_confusion(test_preds, test_labels, class_names,
                   features_dir / "confusion_matrix.png")

    # Hinweis, wie es weitergeht: der fertige Befehl für die Live-Erkennung
    print(f"\n[INFO] Model saved → {best_path}")
    print(f"[INFO] Run inference with:")
    print(f"       python main.py --source <video> --trick-model {best_path} "
          f"--label-map {features_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
