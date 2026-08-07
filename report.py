"""
report.py — Erzeugt aus einer Session-Datenbank einen HTML-Bericht.

ABLAUF:
  session.db (von session_db.py gefüllt)
      → Statistik auslesen
      → Diagramme mit matplotlib zeichnen
      → Diagramme als Base64-Text in die HTML-Datei einbetten
      → eine einzige, in sich geschlossene .html-Datei schreiben

Der Trick mit Base64: Statt die Bilder als separate .png-Dateien abzulegen,
werden sie direkt als Text in das HTML geschrieben. Ergebnis ist EINE Datei,
die man verschicken oder ins Anhangkapitel der Arbeit legen kann.

Aufruf über die Kommandozeile:
    python report.py --db session.db --output report.html

Oder aus Python heraus:
    from report import generate_report
    generate_report("session.db", "report.html")
"""

import argparse
import base64
import io
import os
import sqlite3

# matplotlib ist optional: fehlt es, läuft der Bericht trotzdem – nur ohne Diagramme.
try:
    import matplotlib # type: ignore
    # "Agg" = Zeichnen ohne Bildschirmfenster (wichtig auf Servern/im Hintergrund).
    # Muss VOR dem Import von pyplot gesetzt werden.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt # type: ignore
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

import session_db


# ── Hilfsfunktionen für die Diagramme ────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Wandelt ein matplotlib-Diagramm in einen Base64-Text um.

    Ablauf: Diagramm in einen Speicherpuffer (statt in eine Datei) als PNG
    schreiben, Fenster schließen (sonst Speicherleck), Bytes lesen und in
    Base64-Text kodieren, der direkt in ein <img src="..."> passt."""
    buf = io.BytesIO()                                        # Datei-Ersatz im RAM
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close(fig)                                            # Speicher freigeben
    buf.seek(0)                                               # zurück an den Anfang
    return base64.b64encode(buf.read()).decode()


def _speed_chart(conn: sqlite3.Connection, pid: int) -> str | None:
    """Liniendiagramm: Geschwindigkeitsverlauf einer Person über das ganze Video."""
    if not _MATPLOTLIB:
        return None
    rows = conn.execute(
        "SELECT frame_idx, speed_kmh FROM frames WHERE player_id=? ORDER BY frame_idx",
        (pid,),
    ).fetchall()
    if not rows:
        return None
    # zip(*rows) dreht die Liste von Paaren in zwei Listen um:
    # [(1, 5.0), (2, 6.0)] → (1, 2) und (5.0, 6.0)
    frames, speeds = zip(*rows)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(frames, speeds, color="#00ffc8", linewidth=1)
    ax.set_title(f"P{pid:03d} Speed (km/h)", color="white")
    # Dunkles Design, passend zum restlichen Dashboard
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():   # spines = die vier Rahmenlinien der Achse
        spine.set_edgecolor("grey")
    return _fig_to_b64(fig)


def _height_chart(conn: sqlite3.Connection, pid: int) -> str | None:
    """Flächendiagramm: Boardhöhe über Boden — die Berge sind die Sprünge."""
    if not _MATPLOTLIB:
        return None
    rows = conn.execute(
        "SELECT frame_idx, height_px FROM frames WHERE player_id=? ORDER BY frame_idx",
        (pid,),
    ).fetchall()
    if not rows:
        return None
    frames, heights = zip(*rows)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.fill_between(frames, heights, color="#00a5ff", alpha=0.6)  # gefüllte Fläche
    ax.plot(frames, heights, color="#00a5ff", linewidth=1)        # Konturlinie darüber
    ax.set_title(f"P{pid:03d} Board Height (px)", color="white")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("grey")
    return _fig_to_b64(fig)


def _tricks_bar(tricks: list) -> str | None:
    """Waagerechtes Balkendiagramm: Wie oft wurde welcher Trick gefahren?"""
    if not _MATPLOTLIB or not tricks:
        return None
    names  = [t["name"] or "unknown" for t in tricks]
    counts = [t["count"] for t in tricks]
    # Höhe wächst mit der Anzahl der Tricks, damit die Balken nicht gequetscht werden
    fig, ax = plt.subplots(figsize=(6, max(2, len(names) * 0.5)))
    bars = ax.barh(names, counts, color="#ffd700")   # barh = horizontale Balken
    ax.set_title("Trick Frequency", color="white")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("grey")
    return _fig_to_b64(fig)


# ── HTML-Vorlage ─────────────────────────────────────────────────────────────

# Das Stylesheet für den Bericht. Dunkles Design mit Gold-/Türkis-Akzenten;
# ".grid" sorgt dafür, dass sich die Spielerkarten automatisch nebeneinander
# anordnen und bei schmalem Fenster umbrechen.
_CSS = """
body{background:#0d0d1a;color:#ddd;font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:24px}
h1{color:#ffd700;text-align:center}
h2{color:#00ffc8;border-bottom:1px solid #333;padding-bottom:6px}
h3{color:#aaa}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin-bottom:32px}
.card{background:#1a1a2e;border-radius:10px;padding:20px;border:1px solid #333}
.stat{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #222}
.stat-label{color:#888}
.stat-value{color:#fff;font-weight:bold}
table{width:100%;border-collapse:collapse;margin-top:12px}
th{background:#222;color:#ffd700;padding:8px;text-align:left}
td{padding:7px 8px;border-bottom:1px solid #222}
tr:hover td{background:#1e1e3a}
img{width:100%;border-radius:6px;margin-top:10px}
.badge{display:inline-block;background:#ffd700;color:#000;border-radius:12px;
       padding:2px 10px;font-size:0.82em;margin-left:8px;font-weight:bold}
"""

def _stat(label: str, value) -> str:
    """Baut eine einzelne Statistik-Zeile: links die Bezeichnung, rechts der Wert."""
    return f'<div class="stat"><span class="stat-label">{label}</span><span class="stat-value">{value}</span></div>'


def _chart_block(b64: str | None, alt: str) -> str:
    """Bettet ein Base64-Diagramm als <img> ein — oder gibt leeren Text zurück,
    falls das Diagramm nicht erzeugt werden konnte (z. B. matplotlib fehlt)."""
    if not b64:
        return ""
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def generate_report(db_path: str, output_html: str) -> None:
    """Liest *db_path* aus und schreibt einen eigenständigen HTML-Bericht nach *output_html*."""
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn    = sqlite3.connect(db_path)
    summary = session_db.get_session_summary(conn)   # gesamte Statistik holen

    # Für jede Person wird eine "Karte" (Card) im HTML erzeugt.
    cards = []
    for pid, stats in summary["players"].items():
        # Die drei Diagramme dieser Person
        speed_img  = _speed_chart(conn, pid)
        height_img = _height_chart(conn, pid)
        tricks_img = _tricks_bar(stats["tricks"])

        # Tabellenzeilen für die Trick-Aufschlüsselung.
        # {t['best_score']:,} setzt Tausendertrennzeichen: 12345 → 12,345
        trick_rows = "".join(
            f"<tr><td>{t['name']}</td><td>{t['count']}</td>"
            f"<td>{t['best_score']:,}</td></tr>"
            for t in stats["tricks"]
        )
        # Gibt es gar keine Tricks, statt leerer Tabelle einen Hinweistext zeigen
        trick_table = (
            "<table><thead><tr><th>Trick</th><th>Count</th><th>Best Score</th></tr></thead>"
            f"<tbody>{trick_rows}</tbody></table>"
            if trick_rows else "<p style='color:#666'>No tricks logged.</p>"
        )

        html_speed  = _chart_block(speed_img,  "Speed chart")
        html_height = _chart_block(height_img, "Height chart")
        html_tricks = _chart_block(tricks_img, "Tricks chart")

        # f-String über mehrere Zeilen: die {...}-Ausdrücke werden eingesetzt.
        # ":03d" formatiert 7 → "007"; ":.1f" rundet auf eine Nachkommastelle.
        # "or 0" fängt ab, dass die Datenbank NULL liefern kann.
        card = f"""
<div class="card">
  <h2>Player P{pid:03d}</h2>
  {_stat("Final Score",     f"{stats['final_score'] or 0:,}")}
  {_stat("Max Speed",       f"{stats['max_speed_kmh'] or 0:.1f} km/h")}
  {_stat("Max Board Height", f"{stats['max_height_px'] or 0:.0f} px")}
  {_stat("Max Airtime",     f"{stats['max_airtime_s'] or 0:.2f} s")}
  {_stat("Trick Variety",   stats['trick_variety'] or 0)}
  {_stat("Bails",           stats['bails'] or 0)}
  {html_speed}
  {html_height}
  <h3>Trick Breakdown</h3>
  {trick_table}
  {html_tricks}
</div>"""
        cards.append(card)

    conn.close()

    # Alle Karten aneinanderhängen; ohne Daten einen Platzhaltertext anzeigen.
    grid    = "\n".join(cards) if cards else "<p>No player data found.</p>"
    db_name = os.path.basename(db_path)
    # Das Grundgerüst der Seite. &#x1F6F9; ist das Skateboard-Emoji als HTML-Code.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Skate Session Report — {db_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>&#x1F6F9; Skate Session Report</h1>
  <p style="text-align:center;color:#666">Source: {db_name}</p>
  <div class="grid">
{grid}
  </div>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report] Saved: {output_html}")


# ── Kommandozeilen-Schnittstelle ─────────────────────────────────────────────

# Dieser Block läuft NUR, wenn die Datei direkt gestartet wird
# (python report.py) — nicht beim Import durch app.py.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML session report from SQLite DB")
    parser.add_argument("--db",     required=True, help="Path to session.db")
    parser.add_argument("--output", default="report.html", help="Output HTML file")
    args = parser.parse_args()
    generate_report(args.db, args.output)
