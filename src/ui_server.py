"""
Flask + Server-Sent Events bridge.

Endpoints:
  GET  /            -> stub HTML UI (alerts + log + ask box)
  GET  /events      -> SSE stream of all DMSEvent JSON
  POST /ask         -> { "question": "..." }  -> { "answer": "..." }
  GET  /stats       -> current trip stats snapshot

The sink is thread-safe: emit() is called from the frame-processing thread,
clients read from a queue.Queue per connection.
"""
from __future__ import annotations
import json
import queue
import threading
import time
from typing import List

from flask import Flask, Response, request, jsonify, render_template_string

from src.events import DMSEvent
from src.config import UI_HOST, UI_PORT


# ─── Flask app + sink ────────────────────────────────────────────────────────
app = Flask(__name__)

# Each SSE client gets its own queue. We keep a registry protected by a lock.
class SSESink:
    def __init__(self) -> None:
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()

    def attach(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._clients.append(q)
        return q

    def detach(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def emit(self, event: DMSEvent) -> None:
        payload = json.dumps(event.to_json())
        dead: List[queue.Queue] = []
        with self._lock:
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)


sink = SSESink()


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(STUB_HTML)


@app.route("/events")
def events():
    q = sink.attach()

    def gen():
        try:
            # initial comment to open the stream
            yield ": connected\n\n"
            while True:
                try:
                    msg = q.get(timeout=15.0)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            sink.detach(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/ask", methods=["POST"])
def ask():
    from src.dms import DriverMonitoringSystem  # late import to avoid cycles
    dms: DriverMonitoringSystem = app.config["DMS"]
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400
    ans = dms.ask(question)
    return jsonify({"answer": ans.message, "event": ans.to_json()})


@app.route("/stats")
def stats():
    from src.dms import DriverMonitoringSystem
    dms: DriverMonitoringSystem = app.config["DMS"]
    return jsonify(dms.trip.snapshot())


def start_server(dms) -> None:
    app.config["DMS"] = dms
    app.config["SINK"] = sink
    print(f"[UI] starting SSE server on http://{UI_HOST}:{UI_PORT}")
    app.run(host=UI_HOST, port=UI_PORT, threaded=True, use_reloader=False)


# ─── Minimal stub UI ─────────────────────────────────────────────────────────
STUB_HTML = """
<!doctype html>
<html><head><title>DMS UI (stub)</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; background:#0e1116; color:#e6e6e6; }
  header { padding: 16px 24px; background:#161b22; border-bottom:1px solid #30363d; }
  h1 { margin:0; font-size:18px; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap:16px; padding:16px; }
  .panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }
  .panel h2 { margin:0 0 10px 0; font-size:14px; color:#8b949e; text-transform:uppercase; letter-spacing:1px;}
  .alert { padding:10px 12px; border-radius:6px; margin-bottom:8px; font-size:14px; }
  .CRITICAL { background:#3a0d0d; border-left:4px solid #ff5555; }
  .HIGH     { background:#3a240d; border-left:4px solid #ff9e3d; }
  .MEDIUM   { background:#1f2a0d; border-left:4px solid #c6d63d; }
  .LOW      { background:#0d2a1a; border-left:4px solid #5dd693; }
  .log { font-family: ui-monospace, monospace; font-size:12px; max-height:60vh; overflow:auto; }
  .row { padding:4px 0; border-bottom:1px dashed #30363d; }
  input { width:70%; padding:8px; background:#0e1116; color:#e6e6e6; border:1px solid #30363d; border-radius:6px; }
  button { padding:8px 14px; background:#238636; color:white; border:0; border-radius:6px; cursor:pointer; }
  .small { color:#8b949e; font-size:11px; }
</style></head>
<body>
<header><h1>AI CoE — Edge Driver Monitoring (UI stub)</h1></header>
<main>
  <section class="panel">
    <h2>Live Driver Alert</h2>
    <div id="alert">Waiting…</div>
  </section>
  <section class="panel">
    <h2>Trip Log</h2>
    <div id="log" class="log"></div>
  </section>
  <section class="panel" style="grid-column: span 2">
    <h2>Ask the copilot</h2>
    <input id="q" placeholder="e.g. how many times was I distracted?" />
    <button onclick="ask()">Ask</button>
    <div id="answer" style="margin-top:10px;"></div>
    <div class="small">SSE endpoint: <code>/events</code> &middot; POST <code>/ask</code> &middot; GET <code>/stats</code></div>
  </section>
</main>
<script>
const log = document.getElementById("log");
const alertBox = document.getElementById("alert");
const es = new EventSource("/events");
es.onmessage = (e) => {
  const ev = JSON.parse(e.data);
  if (ev.audience === "DRIVER" || ev.audience === "ADMIN") {
    alertBox.className = "alert " + ev.severity;
    alertBox.textContent = "[" + ev.severity + "] " + ev.message;
  }
  const row = document.createElement("div");
  row.className = "row";
  row.textContent = new Date(ev.timestamp*1000).toLocaleTimeString()
    + " · " + ev.event_type + " · " + ev.severity
    + " · " + (ev.audience || "TRIP")
    + " · " + ev.message;
  log.prepend(row);
};

async function ask() {
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  const r = await fetch("/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({question: q})});
  const j = await r.json();
  document.getElementById("answer").textContent = j.answer || "(no answer)";
}
</script>
</body></html>
"""
