"""
session_db.py — Protokollierung einer Trainings-Session in einer SQLite-Datenbank.

WOZU DAS GANZE?
Während main.py ein Video analysiert, entstehen pro Bild Messwerte (Tempo, Höhe,
Flugzeit, erkannter Trick ...). Diese werden hier dauerhaft in einer Datei
(z. B. session.db) gespeichert. Später kann report.py daraus eine HTML-Auswertung
mit Diagrammen erzeugen – ohne dass das Video erneut berechnet werden muss.

SQLite ist eine Datenbank, die komplett in EINER Datei liegt: kein Server,
keine Installation, einfach eine .db-Datei neben dem Projekt.

Typische Verwendung:
    import session_db
    conn = session_db.init_db("session.db")                       # Datenbank öffnen/anlegen
    session_db.log_frame(conn, frame_idx, player_id, metrics_dict)# pro Bild eine Zeile
    session_db.log_trick(conn, player_id, trick, score, airtime, height)  # pro Trick
    summary = session_db.get_session_summary(conn)                # Auswertung holen
    conn.close()
"""

import sqlite3
from typing import Any


# ── Tabellendefinitionen (das "Schema") ─────────────────────────────────────

# Tabelle 1: eine Zeile PRO BILD und PRO PERSON — die Rohdaten der Session.
# "IF NOT EXISTS" heißt: nur anlegen, wenn die Tabelle noch nicht da ist.
_CREATE_FRAMES = """
CREATE TABLE IF NOT EXISTS frames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- fortlaufende Zeilennummer
    frame_idx     INTEGER NOT NULL,                   -- Bildnummer im Video
    player_id     INTEGER NOT NULL,                   -- Tracking-ID der Person
    speed_kmh     REAL,                               -- geschätztes Tempo
    height_px     REAL,                               -- Boardhöhe über Boden in Pixeln
    airtime_s     REAL,                               -- aktuelle Flugzeit in Sekunden
    trick_label   TEXT,                               -- erkannter Trick (oder leer)
    session_score INTEGER,                            -- Punktestand bis hierher
    phase         TEXT,                               -- ground/rising/peak/landing
    stance        TEXT,                               -- REGULAR oder GOOFY
    bail          INTEGER DEFAULT 0                   -- 1 = Sturz erkannt
);
"""

# Tabelle 2: eine Zeile PRO ABGESCHLOSSENEM TRICK — die verdichteten Ereignisse.
_CREATE_TRICKS = """
CREATE TABLE IF NOT EXISTS tricks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL,
    trick       TEXT,        -- Name des Tricks
    score       INTEGER,     -- vergebene Punkte
    airtime_s   REAL,        -- Flugzeit dieses Tricks
    height_px   REAL,        -- erreichte Höhe
    ts          REAL DEFAULT (strftime('%s','now'))   -- Zeitstempel, automatisch gesetzt
);
"""


def init_db(path: str) -> sqlite3.Connection:
    """Legt die Session-Datenbank an (oder öffnet eine vorhandene) und stellt sicher,
    dass beide Tabellen existieren."""
    # check_same_thread=False: main.py schreibt ggf. aus einem anderen Thread als dem,
    # der die Verbindung geöffnet hat (z. B. bei der Flask-Oberfläche).
    conn = sqlite3.connect(path, check_same_thread=False)
    # WAL = "Write-Ahead Logging": deutlich schnellere Schreibvorgänge und
    # gleichzeitiges Lesen ist möglich, während geschrieben wird.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_FRAMES)
    conn.execute(_CREATE_TRICKS)
    conn.commit()
    return conn


def log_frame(conn: sqlite3.Connection, frame_idx: int,
              player_id: int, metrics: dict[str, Any]) -> None:
    """Schreibt die Messwerte EINES Bildes in die Tabelle 'frames'.

    Wichtig für die Geschwindigkeit: es wird nicht nach jedem Bild endgültig
    gespeichert (commit), sondern nur etwa alle 60 Bilder. Ein commit ist teuer;
    bei 30 fps würde das die Videoanalyse spürbar ausbremsen."""
    conn.execute(
        # Die Fragezeichen sind Platzhalter. Werte werden separat übergeben —
        # das ist sicherer und schneller als Text zusammenzubauen.
        """INSERT INTO frames
           (frame_idx, player_id, speed_kmh, height_px, airtime_s,
            trick_label, session_score, phase, stance, bail)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            frame_idx,
            player_id,
            metrics.get("speed_kmh"),
            metrics.get("height_px"),
            metrics.get("airtime_s"),
            # "or None": leerer Text wird zu NULL, damit Statistiken ihn ignorieren
            metrics.get("trick_label") or None,
            metrics.get("session_score"),
            metrics.get("phase") or None,
            metrics.get("stance") or None,
            int(metrics.get("bail", 0)),   # bool → 0/1, SQLite kennt kein True/False
        ),
    )
    if frame_idx % 60 == 0:
        conn.commit()   # etwa jede Sekunde einmal wirklich auf die Platte schreiben


def log_trick(conn: sqlite3.Connection, player_id: int,
              trick: str, score: int, airtime_s: float, height_px: float) -> None:
    """Speichert einen fertig ausgeführten Trick als eigenes Ereignis.
    Hier wird sofort committet – Tricks sind selten, das kostet keine Leistung."""
    conn.execute(
        """INSERT INTO tricks (player_id, trick, score, airtime_s, height_px)
           VALUES (?,?,?,?,?)""",
        (player_id, trick, score, airtime_s, height_px),
    )
    conn.commit()


def get_session_summary(conn: sqlite3.Connection) -> dict:
    """Berechnet die Gesamtstatistik der Session und gibt sie als Dictionary zurück.

    Struktur der Rückgabe:
      {"players": {1: {"max_speed_kmh": ..., "tricks": [...]}, 2: {...}}}
    """
    # Schritt 1: Welche Personen kommen überhaupt vor?
    players = [r[0] for r in conn.execute(
        "SELECT DISTINCT player_id FROM frames ORDER BY player_id"
    ).fetchall()]

    summary = {"players": {}}
    for pid in players:
        # Schritt 2: Bestwerte dieser Person — die Datenbank rechnet das selbst aus.
        row = conn.execute(
            """SELECT
                   MAX(speed_kmh)     AS max_speed,      -- höchstes Tempo
                   MAX(height_px)     AS max_height,     -- höchster Sprung
                   MAX(airtime_s)     AS max_airtime,    -- längste Flugzeit
                   MAX(session_score) AS final_score,    -- Endpunktestand
                   -- Anzahl UNTERSCHIEDLICHER Tricks (leere Felder zählen nicht mit)
                   COUNT(DISTINCT trick_label) FILTER (WHERE trick_label IS NOT NULL) AS trick_variety,
                   SUM(bail)          AS bails           -- Anzahl Stürze
               FROM frames WHERE player_id = ?""",
            (pid,),
        ).fetchone()

        # Schritt 3: Welche Tricks wie oft, und mit welcher Bestpunktzahl?
        tricks = conn.execute(
            """SELECT trick, COUNT(*) AS n, MAX(score) AS best
               FROM tricks WHERE player_id = ?
               GROUP BY trick ORDER BY n DESC""",
            (pid,),
        ).fetchall()

        # Schritt 4: Ergebnis in ein gut lesbares Dictionary umbauen.
        # (row[0], row[1] ... entsprechen der Reihenfolge im SELECT oben)
        summary["players"][pid] = {
            "max_speed_kmh": row[0],
            "max_height_px": row[1],
            "max_airtime_s": row[2],
            "final_score":   row[3],
            "trick_variety": row[4],
            "bails":         row[5],
            "tricks":        [{"name": t[0], "count": t[1], "best_score": t[2]}
                               for t in tricks],
        }
    return summary
