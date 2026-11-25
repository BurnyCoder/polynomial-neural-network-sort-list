"""Zero-dependency real-time web accuracy/loss dashboard for monitoring training runs.

Run standalone via ``uv run python -m pnn_sort.dashboard.server --port 8765``
or launch as a background daemon thread during training.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

from pnn_sort.config import RESULTS_DIR

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>pnn-sort Live Training Monitor</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --orange:#d29922; --red:#f85149; }
  body { margin:0; font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); padding:20px; }
  h1 { margin:0 0 6px 0; font-size:22px; }
  .sub { color:var(--muted); margin-bottom:18px; font-size:13px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:18px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; }
  .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .val { font-size:24px; font-weight:700; margin-top:4px; color:var(--accent); }
  .val.good { color:var(--green); }
  .charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; margin-bottom:18px; }
  svg { width:100%; height:240px; background:#0b0f14; border-radius:8px; display:block; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; }
  select { background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:6px 10px; }
</style>
</head>
<body>
  <h1>pnn-sort — Real-Time Training & Accuracy Dashboard</h1>
  <div class="sub">Live telemetry polling <code>results/runs/*/train_log.jsonl</code> every 1.0s &nbsp;|&nbsp; Active Run: <select id="runSelect"></select></div>

  <div class="grid">
    <div class="card"><div class="label">Current Run</div><div class="val" id="mRun">—</div></div>
    <div class="card"><div class="label">Epoch</div><div class="val" id="mEpoch">—</div></div>
    <div class="card"><div class="label">Val MAE (↓)</div><div class="val good" id="mMae">—</div></div>
    <div class="card"><div class="label">Pairwise Order Acc (↑)</div><div class="val good" id="mPair">—</div></div>
    <div class="card"><div class="label">Monotonicity Rate (↑)</div><div class="val" id="mMono">—</div></div>
    <div class="card"><div class="label">Train Loss</div><div class="val" id="mLoss">—</div></div>
  </div>

  <div class="charts">
    <div class="card">
      <div class="label" style="margin-bottom:8px;">Validation MAE & Train Loss (Log Scale)</div>
      <svg id="chartLoss" viewBox="0 0 600 240"></svg>
    </div>
    <div class="card">
      <div class="label" style="margin-bottom:8px;">Validation Pairwise Accuracy & Monotonicity Rate</div>
      <svg id="chartAcc" viewBox="0 0 600 240"></svg>
    </div>
  </div>

  <div class="card">
    <div class="label" style="margin-bottom:8px;">Completed & Active Runs Summary</div>
    <table>
      <thead><tr><th>Run ID</th><th>Model</th><th>n</th><th>Epoch</th><th>Val MAE</th><th>Best MAE</th><th>Mono Rate</th><th>Pair Acc</th></tr></thead>
      <tbody id="runsBody"></tbody>
    </table>
  </div>

<script>
let selectedRun = null;
function plotLines(svgId, seriesList, yLog) {
  const svg = document.getElementById(svgId);
  const W = 600, H = 240, padL = 56, padR = 16, padT = 16, padB = 28;
  const xs = seriesList.flatMap(s => s.pts.map(p => p[0]));
  const ysRaw = seriesList.flatMap(s => s.pts.map(p => p[1])).filter(v => Number.isFinite(v));
  if (!xs.length || !ysRaw.length) { svg.innerHTML = '<text x="20" y="30" fill="#8b949e">Waiting for training epochs...</text>'; return; }
  const xMin = Math.min(...xs), xMax = Math.max(...xs, xMin + 1);
  const trY = v => yLog ? Math.log10(Math.max(v, 1e-8)) : v;
  const ys = ysRaw.map(trY);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (Math.abs(yMax - yMin) < 1e-6) { yMin -= 0.5; yMax += 0.5; }
  const sx = x => padL + (x - xMin) / (xMax - xMin) * (W - padL - padR);
  const sy = y => H - padB - (trY(y) - yMin) / (yMax - yMin) * (H - padT - padB);
  let html = '';
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yv = yMin + frac * (yMax - yMin);
    const py = H - padB - frac * (H - padT - padB);
    const lbl = yLog ? (10 ** yv).toExponential(1) : yv.toFixed(3);
    html += `<line x1="${padL}" y1="${py}" x2="${W-padR}" y2="${py}" stroke="#21262d" />`;
    html += `<text x="${padL-6}" y="${py+4}" text-anchor="end" font-size="10" fill="#8b949e">${lbl}</text>`;
  }
  seriesList.forEach((s, idx) => {
    const valid = s.pts.filter(p => Number.isFinite(p[1]));
    if (valid.length < 1) return;
    const d = valid.map((p, i) => `${i===0?'M':'L'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
    html += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.2" />`;
    html += `<text x="${padL + 10 + idx*150}" y="14" font-size="11" fill="${s.color}">■ ${s.name}</text>`;
  });
  svg.innerHTML = html;
}

async function refresh() {
  try {
    const res = await fetch('/api/runs');
    const data = await res.json();
    const runs = data.runs || [];
    const sel = document.getElementById('runSelect');
    if (runs.length && !selectedRun) selectedRun = data.active_run || runs[runs.length - 1].run_id;
    const existing = Array.from(sel.options).map(o => o.value);
    const ids = runs.map(r => r.run_id);
    if (JSON.stringify(existing) !== JSON.stringify(ids)) {
      sel.innerHTML = ids.map(id => `<option value="${id}" ${id===selectedRun?'selected':''}>${id}</option>`).join('');
    }
    const tb = document.getElementById('runsBody');
    tb.innerHTML = runs.slice().reverse().map(r => `<tr>
      <td><code>${r.run_id}</code></td><td>${r.model_name}</td><td>${r.n}</td>
      <td>${r.epoch}/${r.total_epochs}</td>
      <td>${Number.isFinite(r.val_mae)?r.val_mae.toExponential(3):'NaN'}</td>
      <td>${Number.isFinite(r.best_val_mae)?r.best_val_mae.toExponential(3):'NaN'}</td>
      <td>${(r.val_mono_rate*100).toFixed(2)}%</td>
      <td>${(r.val_pair_acc*100).toFixed(2)}%</td>
    </tr>`).join('');

    if (selectedRun) {
      const hres = await fetch('/api/history?run_id=' + encodeURIComponent(selectedRun));
      const hdata = await hres.json();
      const hist = hdata.history || [];
      if (hist.length) {
        const last = hist[hist.length - 1];
        document.getElementById('mRun').textContent = last.run_id;
        document.getElementById('mEpoch').textContent = `${last.epoch} / ${last.total_epochs}`;
        document.getElementById('mMae').textContent = Number.isFinite(last.val_mae) ? last.val_mae.toExponential(3) : 'INF';
        document.getElementById('mPair').textContent = `${(last.val_pair_acc*100).toFixed(2)}%`;
        document.getElementById('mMono').textContent = `${(last.val_mono_rate*100).toFixed(2)}%`;
        document.getElementById('mLoss').textContent = Number.isFinite(last.train_loss) ? last.train_loss.toExponential(3) : 'INF';
        plotLines('chartLoss', [
          { name: 'Val MAE', color: '#3fb950', pts: hist.map(h => [h.epoch, h.val_mae]) },
          { name: 'Train Loss', color: '#58a6ff', pts: hist.map(h => [h.epoch, h.train_loss]) },
        ], true);
        plotLines('chartAcc', [
          { name: 'Pairwise Order Acc', color: '#3fb950', pts: hist.map(h => [h.epoch, h.val_pair_acc]) },
          { name: 'Full Monotonicity', color: '#d29922', pts: hist.map(h => [h.epoch, h.val_mono_rate]) },
        ], false);
      }
    }
  } catch (e) { console.error(e); }
}
document.getElementById('runSelect').addEventListener('change', e => { selectedRun = e.target.value; refresh(); });
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def make_handler(results_root: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index.html"):
                body = DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/runs":
                runs_dir = results_root / "runs"
                summaries = []
                if runs_dir.exists():
                    for p in sorted(runs_dir.glob("*/train_log.jsonl")):
                        hist = _read_jsonl(p)
                        if hist:
                            summaries.append(hist[-1])
                active = None
                live_path = results_root / "live_dashboard_state.json"
                if live_path.exists():
                    try:
                        active = json.loads(live_path.read_text(encoding="utf-8")).get("run_id")
                    except Exception:
                        active = None
                self._send_json({"active_run": active, "runs": summaries})
            elif self.path.startswith("/api/history"):
                run_id = self.path.split("run_id=")[-1] if "run_id=" in self.path else ""
                hist = _read_jsonl(results_root / "runs" / run_id / "train_log.jsonl")
                self._send_json({"run_id": run_id, "history": hist})
            else:
                self.send_error(404)

    return DashboardHandler


def start_dashboard_thread(port: int = 8765, results_root: Path = RESULTS_DIR) -> ThreadingHTTPServer:
    """Start the real-time accuracy metrics dashboard in a background daemon thread."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(results_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time pnn-sort training monitor")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(RESULTS_DIR))
    print(f"pnn-sort Live Dashboard listening at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
