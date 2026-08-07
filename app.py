"""
app.py — Flask-Webserver (Dashboard) für die Skateboard-Analyse.

WOZU EINE WEBOBERFLÄCHE?
viewer.py zeigt die Analyse in einem Desktop-Fenster auf demselben Rechner.
app.py macht dasselbe im Browser — auch von einem anderen Gerät im Netzwerk aus
(z. B. Tablet am Rand der Halle). Zusätzlich lassen sich hier gespeicherte
Sessions durchblättern und Berichte erzeugen.

Angebotene Adressen (Routen):
  • GET  /              → Bestenliste + Live-Statistik als HTML-Seite
  • GET  /stream        → MJPEG-Livestream des annotierten Videos
  • GET  /api/stats     → JSON-Momentaufnahme aller Spielerwerte
  • GET  /api/leaderboard → JSON-Bestenliste
  • GET  /sessions      → Liste aufgezeichneter Session-Datenbanken
  • GET  /report/<name> → erzeugt und liefert einen HTML-Bericht auf Abruf

Parallel zu main.py starten (main.py legt Bilder in eine gemeinsame Warteschlange):
    python app.py [--db session.db] [--port 5000]

Oder allein starten, um vergangene Sessions anzusehen:
    python app.py --db session.db --no-stream --port 5000
"""

import argparse
import os
import queue
import threading
import time
from io import BytesIO

import cv2

# Flask ist Pflicht für dieses Skript — ohne sie wird sofort mit Hinweis beendet.
try:
    from flask import Flask, Response, jsonify, render_template_string, send_file
    _FLASK = True
except ImportError:
    _FLASK = False
    print("[app] Flask not installed. Run: pip install flask")
    raise SystemExit(1)

import session_db
from report import generate_report

app = Flask(__name__)

# ── Globale, gemeinsam genutzte Zustände ─────────────────────────────────────
# maxsize=2: Es wird bewusst nur das NEUESTE Bild vorgehalten. Ein größerer
# Puffer würde dazu führen, dass der Browser veraltete Bilder anzeigt und der
# Stream immer weiter hinterherhinkt.
_frame_queue: queue.Queue = queue.Queue(maxsize=2)   # jeweils letztes annotiertes Bild
_player_stats: dict = {}                              # wird von main.py befüllt
# Lock (Schloss): Webserver und Analyse laufen in verschiedenen Threads. Ohne
# Sperre könnte der Server Werte lesen, während sie gerade halb geschrieben sind.
_stats_lock = threading.Lock()
_db_path: str | None = None


def push_frame(frame) -> None:
    """Wird von main.py aufgerufen, um das aktuellste annotierte Bild bereitzustellen."""
    # Ist die Warteschlange voll, das alte Bild wegwerfen statt zu warten.
    # Die Analyse darf niemals durch einen langsamen Browser gebremst werden.
    if _frame_queue.full():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            pass    # anderer Thread war schneller — kein Problem
    _frame_queue.put(frame)


def push_stats(stats: dict) -> None:
    """Wird von main.py aufgerufen, um den aktuellen Spielerzustand zu veröffentlichen."""
    with _stats_lock:      # exklusiver Zugriff, bis der Block verlassen wird
        _player_stats.clear()
        _player_stats.update(stats)


# ── MJPEG-Generator ──────────────────────────────────────────────────────────

def _stream_generator():
    """Erzeugt einen endlosen MJPEG-Strom für das <img>-Element im Browser.

    MJPEG ist der einfachste Videostream überhaupt: eine ununterbrochene Folge
    einzelner JPEG-Bilder, getrennt durch eine Markierung. Jeder Browser kann
    das ohne Zusatzsoftware anzeigen.

    'yield' statt 'return': Die Funktion liefert Bild für Bild und läuft danach
    weiter, statt sich zu beenden."""
    placeholder = None
    while True:
        try:
            frame = _frame_queue.get(timeout=0.5)
            placeholder = frame          # letztes gültiges Bild merken
        except queue.Empty:
            if placeholder is None:
                time.sleep(0.05)         # noch nie ein Bild bekommen → kurz warten
                continue
            frame = placeholder          # letztes Bild erneut senden (Bild einfrieren
                                         # statt schwarz werden zu lassen)
        # Qualität 75 statt 95: deutlich kleinere Datenmenge, kaum sichtbarer Unterschied
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        # Das genaue Format, das der Browser für multipart/x-mixed-replace erwartet
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")


# ── HTML-Vorlage der Hauptseite ───────────────────────────────────────────────
# Die komplette Seite steckt als Text im Python-Code. Vorteil: keine zusätzlichen
# Dateien nötig, das ganze Dashboard ist eine einzige .py-Datei.
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Skate Dashboard</title>
  <style>
    body{background:#0d0d1a;color:#ddd;font-family:'Segoe UI',Arial,sans-serif;margin:0}
    header{background:#111;padding:14px 24px;border-bottom:1px solid #333;display:flex;
           align-items:center;gap:16px}
    header h1{margin:0;color:#ffd700;font-size:1.5rem}
    nav a{color:#00ffc8;text-decoration:none;margin:0 10px}
    #stream-wrap{max-width:960px;margin:20px auto}
    #live-stream{width:100%;border-radius:8px;border:1px solid #333}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
          gap:16px;max-width:960px;margin:0 auto 30px}
    .card{background:#1a1a2e;border-radius:10px;padding:18px;border:1px solid #333}
    .card h2{margin:0 0 10px;color:#00ffc8;font-size:1rem}
    .stat{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #222;font-size:.9rem}
    .stat-label{color:#888}.stat-value{font-weight:bold}
    footer{text-align:center;color:#444;font-size:.8rem;padding:20px}
  </style>
</head>
<body>
  <header>
    <h1>&#x1F6F9; Skate Dashboard</h1>
    <nav>
      <a href="/">Live</a>
      <a href="/sessions">Sessions</a>
    </nav>
  </header>
  <div id="stream-wrap">
    <!-- Dieses img zeigt den MJPEG-Stream: der Browser lädt es endlos weiter -->
    <img id="live-stream" src="/stream" alt="Live feed">
  </div>
  <div class="grid" id="stats-grid">
    <p style="color:#555;text-align:center">Waiting for players…</p>
  </div>
  <footer>YOLOv8 Pose | BiLSTM Trick Classifier | &copy; Thesis</footer>
  <script>
    // Holt zweimal pro Sekunde die aktuellen Werte und baut die Karten neu auf.
    // Das Video selbst kommt separat über den MJPEG-Stream oben.
    async function refreshStats() {
      try {
        const r = await fetch('/api/stats');
        const data = await r.json();
        const grid = document.getElementById('stats-grid');
        if (!Object.keys(data).length) return;   // noch keine Daten -> Platzhalter lassen
        grid.innerHTML = '';
        for (const [pid, st] of Object.entries(data)) {
          const card = document.createElement('div');
          card.className = 'card';
          card.innerHTML = `
            <h2>Player P${String(pid).padStart(3,'0')}</h2>
            <div class="stat"><span class="stat-label">Score</span>
              <span class="stat-value">${(st.session_score||0).toLocaleString()}</span></div>
            <div class="stat"><span class="stat-label">Speed</span>
              <span class="stat-value">${(st.speed_kmh||0).toFixed(1)} km/h</span></div>
            <div class="stat"><span class="stat-label">Max Height</span>
              <span class="stat-value">${(st.max_height_px||0)} px</span></div>
            <div class="stat"><span class="stat-label">Max Airtime</span>
              <span class="stat-value">${(st.max_airtime||0).toFixed(2)} s</span></div>
            <div class="stat"><span class="stat-label">Phase</span>
              <span class="stat-value">${st.phase||'--'}</span></div>
            <div class="stat"><span class="stat-label">Stance</span>
              <span class="stat-value">${st.stance||'--'}</span></div>
            <div class="stat"><span class="stat-label">Trick</span>
              <span class="stat-value">${st.trick_label||'--'}</span></div>
          `;
          grid.appendChild(card);
        }
      } catch(e) {}   // Netzwerkfehler still ignorieren, beim nächsten Mal klappt es wieder
    }
    setInterval(refreshStats, 500);   // alle 500 ms wiederholen
    refreshStats();                   // einmal sofort, nicht erst nach 0,5 s
  </script>
</body>
</html>"""

# Seite mit der Liste gespeicherter Sessions.
# {% ... %} sind Jinja2-Befehle: Flask ersetzt sie serverseitig durch echte Werte.
_SESSIONS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><title>Sessions</title>
  <style>
    body{background:#0d0d1a;color:#ddd;font-family:'Segoe UI',Arial,sans-serif;padding:30px}
    h1{color:#ffd700}
    a{color:#00ffc8}
    li{margin:8px 0}
  </style>
</head>
<body>
  <h1>&#x1F4C2; Session Databases</h1>
  {% if sessions %}
  <ul>
    {% for s in sessions %}
    <li><a href="/report/{{ s }}">{{ s }}</a></li>
    {% endfor %}
  </ul>
  {% else %}
  <p style="color:#555">No .db files found in current directory.</p>
  {% endif %}
  <p><a href="/">&#x2190; Back to dashboard</a></p>
</body>
</html>"""


# ── Routen (welche URL löst welche Funktion aus) ──────────────────────────────

@app.route("/")
def index():
    """Startseite: liefert die Dashboard-HTML-Seite aus."""
    return render_template_string(_INDEX_HTML)


@app.route("/stream")
def stream():
    """Der Videostream. mimetype 'multipart/x-mixed-replace' sagt dem Browser:
    'Ersetze das Bild fortlaufend durch das nächste' — das ergibt bewegtes Video."""
    return Response(
        _stream_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stats")
def api_stats():
    """Liefert die aktuellen Werte als JSON.

    Es wird bewusst NICHT _player_stats direkt zurückgegeben: dort stecken auch
    deques, numpy-Werte und interne Felder, die sich nicht in JSON umwandeln
    lassen. Deshalb wird hier ein sauberer, reduzierter Auszug ("safe") gebaut."""
    with _stats_lock:
        safe = {}
        for tid, st in _player_stats.items():
            safe[tid] = {
                "session_score": st.get("session_score", 0),
                "speed_kmh":     round(float(st.get("speed_kmh", 0)), 2),
                "max_height_px": st.get("max_height_px", 0),
                "max_airtime":   round(float(st.get("max_airtime", 0)), 3),
                "phase":         st.get("phase", "ground"),
                "stance":        st.get("stance") or None,
                "trick_label":   st.get("trick_label") or None,
                "bail_detected": bool(st.get("bail_detected", False)),
                "trick_streak":  st.get("trick_streak", 0),
            }
    return jsonify(safe)


@app.route("/api/leaderboard")
def api_leaderboard():
    """Bestenliste als JSON, absteigend nach Punktzahl sortiert."""
    with _stats_lock:
        board = sorted(
            [{"player_id": tid, "score": st.get("session_score", 0),
              "tricks": st.get("trick_count", 0)}
             for tid, st in _player_stats.items()],
            key=lambda x: x["score"], reverse=True   # reverse: größte zuerst
        )
    return jsonify(board)


@app.route("/sessions")
def sessions():
    """Listet alle .db-Dateien im aktuellen Arbeitsverzeichnis auf."""
    dbs = sorted(f for f in os.listdir(".") if f.endswith(".db"))
    return render_template_string(_SESSIONS_HTML, sessions=dbs)


@app.route("/report/<path:db_name>")
def report(db_name: str):
    """Erzeugt auf Abruf einen HTML-Bericht zur gewählten Datenbank."""
    # SICHERHEIT: basename() entfernt jeden Pfadanteil. Ohne das könnte jemand
    # "/report/../../etc/passwd" aufrufen und Dateien außerhalb des Ordners
    # auslesen (sogenannter Path-Traversal-Angriff).
    safe_name = os.path.basename(db_name)
    if not safe_name.endswith(".db") or not os.path.isfile(safe_name):
        return f"Database '{safe_name}' not found.", 404   # 404 = nicht gefunden
    out_html = safe_name.replace(".db", "_report.html")
    try:
        generate_report(safe_name, out_html)
    except Exception as exc:
        # Fehler abfangen und als Meldung zeigen, statt den Server abstürzen zu lassen
        return f"Error generating report: {exc}", 500      # 500 = Serverfehler
    return send_file(out_html, mimetype="text/html")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skate analysis web dashboard")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port")
    # 127.0.0.1 = nur dieser Rechner. Für Zugriff aus dem Netzwerk: --host 0.0.0.0
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    args = parser.parse_args()
    print(f"[app] Dashboard: http://{args.host}:{args.port}/")
    # threaded=True: mehrere Anfragen gleichzeitig bedienen. Nötig, weil der
    # Videostream eine Verbindung dauerhaft offen hält.
    app.run(host=args.host, port=args.port, threaded=True)
