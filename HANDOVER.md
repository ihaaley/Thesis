# Übergabe — Stand 15.09.2026

Diese Datei beschreibt den Arbeitsstand, damit auf einem anderen Rechner ohne
Rückfragen weitergearbeitet werden kann.

---

## 1. Was gerade ansteht

Merkmale neu extrahieren und Modell neu trainieren — mit einer **Neutralklasse
„Rollen"**, weil das Modell bisher das An- und Wegrollen keiner Klasse zuordnen
konnte und deshalb Fehlalarme erzeugte.

```bash
# 1) Extraktion  (~2–3 h auf einem schnellen Rechner)
python3 extract_features_manifest.py \
    --manifest   data/manifest_20klassen_max60.csv \
    --output     features/tricks_v2 \
    --frame-skip 4

# 2) Training mit clipweiser Aufteilung
python3 train.py --features features/tricks_v2 --split-by clip

# 3) Vergleichslauf mit der alten, fensterweisen Aufteilung
python3 train.py --features features/tricks_v2 --split-by window
```

Schritt 3 ist wichtig: Die Differenz zwischen beiden Läufen zeigt, wie stark die
bisher berichteten 56,8 % durch überlappende Fenster geschönt waren.

**Voraussetzungen**
- Externe Platte `HHN_LBSBLab` angeschlossen
- `yolov8s-pose.pt` lokal vorhanden (nicht in iCloud ausgelagert)
- Hängt die Platte anders ein, `--video-root /Volumes/<name>` ergänzen

---

## 2. Regeln

**Die externe Platte gehört nicht der Autorin.** Nur lesen — nichts schreiben,
verschieben, umbenennen oder löschen, auch keine Hilfsdateien dort anlegen.
Alle Ausgaben gehören ins Projektverzeichnis.

**Die Word-Datei nie aus Markdown neu erzeugen.** Sie wird direkt über
`python-docx` bearbeitet, sonst gehen Änderungen der Autorin verloren. Vor jedem
Schreibzugriff prüfen, ob `~$Bachelorarbeit.docx` existiert (dann ist sie in Word
geöffnet), und eine Sicherung anlegen.

**Zitate niemals erfinden.** Statt einer Quelle `⟨Beleg einfügen: …⟩` setzen.

---

## 3. Wo was liegt

| | |
|---|---|
| Code | `~/Desktop/Bachelorarbeit-TrickVision/` (dieses Repo) |
| Thesis-Dokument | `~/Desktop/Thesis/Bachelorarbeit.docx` — **eine** Datei, alles darin |
| Gliederung für den Betreuer | `~/Desktop/Thesis/Gliederung_Oeztuerk.docx` |
| Literatur-Startliste | `~/Desktop/Thesis/Literatur_Startliste.md` |
| Videos | externe Platte `HHN_LBSBLab` |

Die Thesis-Dateien liegen **außerhalb** des Repos und müssen getrennt kopiert
werden.

---

## 4. Datenlage

Aus dem Bestand auf der Platte wurde ein Manifest erzeugt, das Dubletten
ausschließt und die Klassenzuordnung dokumentiert.

| | |
|---|---:|
| Videos auf der Platte, eindeutig | 2.585 |
| **gefundene Dubletten** (dieselbe Datei in mehreren Ordnern) | **410** |
| Manifest vollständig, 20 Klassen | 1.089 Clips |
| Manifest begrenzt auf max. 60 je Klasse | **640 Clips** |

`data/manifest_20klassen_max60.csv` ist die empfohlene Grundlage: Die Begrenzung
senkt die Klassenungleichheit von 456:20 auf 60:20 und verkürzt die Rechenzeit.
Die Auswahl zieht gleichmäßig über alle Aufnahmesessions, damit verschiedene
Orte, Lichtverhältnisse und Auflösungen vertreten bleiben (Seed 42).

Die Klasse **Rollen** fasst `pushingV1`, `pushingV2` und `Carving` zusammen.

### Metadaten auf der Platte

`LongboardLab_MetaData_20260325.xlsx` enthält:
- **`board_length_cm = 118`** — Referenzlänge für die geplante Kalibrierung
- Athlet: **nur eine Person** — Angaben dazu stehen in der Tabelle auf der Platte
  und werden hier bewusst nicht wiedergegeben
- **keine** Anfangs-/Endzeiten der Tricks → das Label-Rauschen bleibt bestehen

---

## 5. Zwei Befunde, die in den Text müssen

**Die Videos laufen mit 60 fps, nicht mit 30.** In Kapitel 4 steht derzeit noch
30 fps. Mit `--frame-skip 2` ergeben sich daraus 30 effektive fps und damit
**1-Sekunden-Fenster** statt der im Text behaupteten 2 Sekunden. Deshalb die
Empfehlung `--frame-skip 4`: Das ergibt 15 effektive fps und 2-Sekunden-Fenster,
passend zur tatsächlichen Trickdauer — und halbiert die Rechenzeit.
Nach dem Lauf sind die Angaben in Abschnitt 4.2.1 und 4.2.3 zu korrigieren.

**Die bisherige Aufteilung erzeugt eine Datenleckage.** `stratified_split` mischt
einzelne Fenster. Bei Fenstergröße 30 und Schrittweite 15 überlappen benachbarte
Fenster zur Hälfte — fast identische Fenster landen dadurch gleichzeitig in
Training und Test. Deshalb die neue Option `--split-by clip`. Steht bereits in
Abschnitt 6.5 der Arbeit.

---

## 6. Stand der Arbeit

Rund 32 Seiten Fließtext.

| Kapitel | Stand |
|---|---|
| 1 Einleitung | fertig |
| 2 Grundlagen | fertig, 16 nummerierte Gleichungen |
| 3 Stand der Technik | fertig bis auf **3.2** (Gerüst) |
| 4 Material und Methoden | fertig, 3 Code-Listings |
| 5 Ergebnisse | **wartet auf das neue Training** |
| 6 Diskussion | 6.2, 6.3, 6.5, 6.6 fertig — 6.1 und 6.4 warten |
| 7 Fazit | 7.2 fertig, 7.1 wartet |

Offen: 27 Belegstellen `⟨Beleg einfügen: …⟩`, überwiegend in Kapitel 2 und 3.

**Abschnitt 3.2 lässt sich jetzt schreiben:** Auf der Platte liegt unter
`HHN_GECKO_LBSBLab_literature/` ein systematisches Review mit 16 Arbeiten zur
Skateboard-Trickklassifikation — darunter Shaipee 2020 (videobasiert, direkter
Vorläufer), Groh 2015–2021 und Westenberger 2023. Damit ist auch die
Forschungslücke in 3.4 zu schärfen: Es gibt Vorarbeiten, überwiegend
sensorbasiert.

---

## 7. Werkzeuge

- `pandoc` unter `~/.local/bin/pandoc` (Homebrew scheidet auf macOS 12 aus —
  würde GHC aus dem Quellcode bauen). Wird nur noch selten gebraucht.
- `python-docx` für die Bearbeitung der Word-Datei
- `openpyxl` zum Lesen der Metadaten-Tabelle
