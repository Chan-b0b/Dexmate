"""Browser OBB editor: hand-correct the labels model_autolabel_bev.py flagged.

model_autolabel_bev.py writes the labels it trusts and lists the rest in
pending_case_bev/manifest.csv. This serves those frames one at a time with the
model's own prediction as the starting box, so a bad label is a drag away from
a good one: drag inside to move, drag a corner to resize, drag the top handle
to rotate. Saving writes the label in exactly the format the SAM2 labelers
produce (one row, 4 normalized corners), so review.py / prepare_dataset.py /
train.py are unaffected.

The BEV canvas is metric, so the box size is shown in metres against the
measured cfg.CASE_BEV_SIZE_M and turns red when it drifts out of tolerance —
that readout is the fastest check that an edit is right.

stdlib HTTP layer + cv2/numpy for the images (same shape as the dashboard's
review_server.py). Source frames are only ever read, never written.

    python obb_edit_server.py               # http://<robot-ip>:8082/
    python obb_edit_server.py --port 9000

Keys: n/p frame, s save, x no-case, r reset to the prediction, q/e rotate
(shift = coarse), arrows nudge.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from model_autolabel_bev import TARGET, _load_manifest, _write_manifest
from obb_label import points_to_yolo_line, yolo_line_to_points

HERE = Path(__file__).resolve().parent
_LOCK = threading.Lock()   # serialises manifest / label writes


# ---------------------------------------------------------------------------
# Store: the pending manifest + the labeled dir are the whole state
# ---------------------------------------------------------------------------
class Store:
    def __init__(self, labeled: Path, pending: Path) -> None:
        self.labeled = labeled
        self.pending = pending
        self.manifest_path = pending / "manifest.csv"
        self.nocase_path = pending / "nocase.txt"

    def manifest(self) -> dict[str, dict]:
        return _load_manifest(self.manifest_path)

    def nocase(self) -> set[str]:
        return (set(self.nocase_path.read_text().split())
                if self.nocase_path.exists() else set())

    def label_path(self, stem: str) -> Path:
        return self.labeled / "labels" / f"{stem}.txt"

    def src_png(self, stem: str) -> Path | None:
        """The BEV frame `stem` came from (capture run for pending, else the copy)."""
        row = self.manifest().get(stem)
        if row and row["src_png"] and Path(row["src_png"]).exists():
            return Path(row["src_png"])
        img = self.labeled / "images" / f"{stem}.png"
        return img if img.exists() else None

    def rows(self, scope: str) -> list[dict]:
        """Frames to edit: pending first (they need it), then the accepted ones."""
        man = self.manifest()
        rows = [{"stem": s, "reason": man[s]["reason"], "conf": man[s]["conf"],
                 "labeled": False} for s in sorted(man)]
        if scope == "all":
            rows += [{"stem": p.stem, "reason": "", "conf": "", "labeled": True}
                     for p in sorted((self.labeled / "labels").glob("*.txt"))
                     if p.stem not in man]
        return rows

    def obb(self, stem: str) -> dict | None:
        """Starting box for `stem` in BEV px: the model's prediction if it is
        still pending, else the label on disk, else a default-sized box in the
        middle of the canvas (nothing was detected — drag it onto the case)."""
        src = self.src_png(stem)
        if src is None:
            return None
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        row = self.manifest().get(stem)
        if row and row["x1"]:
            pts = [[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 5)]
        elif not row and self.label_path(stem).exists():
            line = self.label_path(stem).read_text().splitlines()[0]
            pts = yolo_line_to_points(line, w, h).astype(float).tolist()
        else:
            long_px = max(cfg.CASE_BEV_SIZE_M) * cfg.BEV_PX_PER_M / 2
            short_px = min(cfg.CASE_BEV_SIZE_M) * cfg.BEV_PX_PER_M / 2
            cx, cy = w / 2, h / 2
            pts = [[cx - long_px, cy - short_px], [cx + long_px, cy - short_px],
                   [cx + long_px, cy + short_px], [cx - long_px, cy + short_px]]
        return {"points": pts, "img_w": w, "img_h": h,
                "reason": row["reason"] if row else "", "conf": row["conf"] if row else ""}

    def save(self, stem: str, points: list[list[float]]) -> str:
        """Write the edited box as a label + copy the frame into the dataset."""
        src = self.src_png(stem)
        if src is None:
            return "no source frame"
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        with _LOCK:
            self.label_path(stem).parent.mkdir(parents=True, exist_ok=True)
            (self.labeled / "images").mkdir(parents=True, exist_ok=True)
            self.label_path(stem).write_text(
                points_to_yolo_line(np.asarray(points, dtype=np.float64), w, h) + "\n")
            dst = self.labeled / "images" / f"{stem}.png"
            if not dst.exists():
                shutil.copy(src, dst)
            self._drop(stem)
        return ""

    def nocase_add(self, stem: str) -> str:
        """Mark `stem` as holding no case: drop it from the dataset for good."""
        with _LOCK:
            names = self.nocase()
            names.add(stem)
            self.nocase_path.write_text("\n".join(sorted(names)) + "\n")
            self.label_path(stem).unlink(missing_ok=True)
            (self.labeled / "images" / f"{stem}.png").unlink(missing_ok=True)
            self._drop(stem)
        return ""

    def _drop(self, stem: str) -> None:
        """Remove `stem` from the pending manifest (caller holds _LOCK)."""
        man = _load_manifest(self.manifest_path)
        if man.pop(stem, None) is not None:
            _write_manifest(self.manifest_path, man)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    store: Store = None  # set in main

    def log_message(self, *_a):  # noqa: D102 - quiet, one line per frame is enough
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _page().encode("utf-8"))
        elif u.path == "/list":
            self._send_json(self.store.rows(q.get("scope", ["pending"])[0]))
        elif u.path.startswith("/obb/"):
            obb = self.store.obb(u.path[len("/obb/"):])
            if obb is None:
                self._send_json({"error": "no source frame"}, 404)
            else:
                self._send_json(obb)
        elif u.path.startswith("/img/"):
            self._send_img(u.path[len("/img/"):].removesuffix(".png"),
                           int(q.get("w", [0])[0]))
        else:
            self._send(404, "text/plain", b"not found")

    def _send_img(self, stem: str, width: int):
        src = self.store.src_png(stem)
        if src is None:
            self._send(404, "text/plain", b"not found")
            return
        img = cv2.imread(str(src))
        if width and width < img.shape[1]:
            s = width / img.shape[1]
            img = cv2.resize(img, (width, max(1, int(img.shape[0] * s))))
        self._send(200, "image/png", cv2.imencode(".png", img)[1].tobytes())

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        body = self._body()
        stem = body.get("stem", "")
        if path == "/save":
            err = self.store.save(stem, body["points"])
        elif path == "/nocase":
            err = self.store.nocase_add(stem)
        else:
            self._send(404, "text/plain", b"not found")
            return
        print(f"{'nocase' if path == '/nocase' else 'saved '}  {stem}"
              + (f"  ERROR {err}" if err else ""))
        self._send_json({"ok": not err, "error": err}, 200 if not err else 400)


def _page() -> str:
    long_m, short_m = max(cfg.CASE_BEV_SIZE_M), min(cfg.CASE_BEV_SIZE_M)
    return (PAGE.replace("__PX_PER_M__", str(cfg.BEV_PX_PER_M))
                .replace("__LONG_M__", str(long_m))
                .replace("__SHORT_M__", str(short_m)))


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>obb_edit</title>
<style>
 body{background:#15181c;color:#dfe3e8;font:13px/1.5 ui-monospace,monospace;margin:0}
 header{padding:8px 12px;border-bottom:1px solid #2b3038;display:flex;gap:14px;align-items:center}
 header b{color:#fff}
 .tab{padding:2px 8px;border:1px solid #3a414b;border-radius:3px;cursor:pointer}
 .tab.on{background:#2d6cdf;border-color:#2d6cdf;color:#fff}
 #wrap{display:flex;height:calc(100vh - 41px)}
 #strip{width:170px;overflow-y:auto;border-right:1px solid #2b3038;padding:6px}
 #strip div{margin-bottom:6px;cursor:pointer;border:2px solid transparent;border-radius:3px}
 #strip div.cur{border-color:#2d6cdf}
 #strip div.done{opacity:.45}
 #strip img{width:100%;display:block;border-radius:2px}
 #strip span{font-size:10px;color:#9aa4b1;display:block;overflow:hidden;white-space:nowrap}
 #main{flex:1;padding:10px;overflow:auto}
 canvas{background:#000;max-width:100%;cursor:crosshair}
 #info{margin:8px 0}
 #size.bad{color:#ff6b6b} #size.ok{color:#5fd38d}
 .k{color:#7d8794}
 button{background:#2b3038;color:#dfe3e8;border:1px solid #3a414b;border-radius:3px;
        padding:4px 10px;cursor:pointer;margin-right:6px}
 button.pri{background:#2d6cdf;border-color:#2d6cdf;color:#fff}
</style>
<header>
  <b>obb_edit</b>
  <span class="tab on" id="tab-pending" onclick="setScope('pending')">pending</span>
  <span class="tab" id="tab-all" onclick="setScope('all')">all</span>
  <span id="pos" class="k"></span>
</header>
<div id="wrap">
  <div id="strip"></div>
  <div id="main">
    <canvas id="cv"></canvas>
    <div id="info">
      <div><span id="stem"></span> <span class="k" id="reason"></span></div>
      <div id="size"></div>
    </div>
    <button class="pri" onclick="save()">save (s)</button>
    <button onclick="nocase()">no case (x)</button>
    <button onclick="reset()">reset (r)</button>
    <button onclick="show(idx-1)">prev (p)</button>
    <button onclick="show(idx+1)">next (n)</button>
    <span class="k">drag inside = move &nbsp; corner = resize &nbsp; top handle = rotate
      &nbsp;|&nbsp; q/e rotate, arrows nudge (shift = coarse)</span>
  </div>
</div>
<script>
const PX_PER_M = __PX_PER_M__, LONG_M = __LONG_M__, SHORT_M = __SHORT_M__, TOL = 0.15;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let rows = [], idx = -1, obb = null, orig = null, drag = null, scope = 'pending';
const img = new Image();

// --- oriented box <-> 4 corners. The box is (center, w, h, theta): w runs
// along u = (cos t, sin t), h along v = (-sin t, cos t). Corner order matches
// what we parse back, so a load/edit/save round trip is exact.
function corners(o){
  const t = o.theta * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const hw = o.w / 2, hh = o.h / 2;
  return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b]) => [
    o.cx + a*hw*c + b*hh*(-s),
    o.cy + a*hw*s + b*hh*c]);
}
function fromPoints(p){
  return {
    cx: (p[0][0]+p[1][0]+p[2][0]+p[3][0]) / 4,
    cy: (p[0][1]+p[1][1]+p[2][1]+p[3][1]) / 4,
    w: Math.hypot(p[1][0]-p[0][0], p[1][1]-p[0][1]),
    h: Math.hypot(p[2][0]-p[1][0], p[2][1]-p[1][1]),
    theta: Math.atan2(p[1][1]-p[0][1], p[1][0]-p[0][0]) * 180 / Math.PI};
}
function toLocal(o, x, y){
  const t = o.theta * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const dx = x - o.cx, dy = y - o.cy;
  return [dx*c + dy*s, -dx*s + dy*c];
}
function rotHandle(o){
  const t = o.theta * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const d = -(o.h/2 + 26);
  return [o.cx + d*(-s), o.cy + d*c];
}

function draw(){
  if (!img.width) return;
  ctx.drawImage(img, 0, 0);
  if (!obb) return;
  const p = corners(obb), rh = rotHandle(obb);
  ctx.lineWidth = 2; ctx.strokeStyle = '#2dd47a';
  ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]);
  p.slice(1).forEach(q => ctx.lineTo(q[0], q[1]));
  ctx.closePath(); ctx.stroke();
  ctx.beginPath();                                    // stem to the rotate handle
  ctx.moveTo(obb.cx, obb.cy); ctx.lineTo(rh[0], rh[1]); ctx.stroke();
  ctx.fillStyle = '#ffd24a';
  p.forEach(q => ctx.fillRect(q[0]-5, q[1]-5, 10, 10));
  ctx.beginPath(); ctx.arc(rh[0], rh[1], 6, 0, 7); ctx.fill();
  const lo = Math.max(obb.w, obb.h) / PX_PER_M, sh = Math.min(obb.w, obb.h) / PX_PER_M;
  const bad = Math.abs(lo-LONG_M)/LONG_M > TOL || Math.abs(sh-SHORT_M)/SHORT_M > TOL;
  const yaw = ((obb.w >= obb.h ? obb.theta : obb.theta + 90) % 180 + 180) % 180;
  const el = document.getElementById('size');
  el.className = bad ? 'bad' : 'ok';
  el.textContent = `${lo.toFixed(3)} x ${sh.toFixed(3)} m  (target ${LONG_M} x ${SHORT_M})`
                 + `   yaw ${yaw.toFixed(1)} deg (bev)`;
}

// --- mouse: canvas px, independent of the CSS scale
function pos(e){
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * cv.width / r.width,
          (e.clientY - r.top) * cv.height / r.height];
}
cv.onmousedown = e => {
  if (!obb) return;
  const [x, y] = pos(e), tol = 10 * cv.width / cv.getBoundingClientRect().width;
  const rh = rotHandle(obb);
  if (Math.hypot(x-rh[0], y-rh[1]) < tol) { drag = {mode:'rot'}; return; }
  const p = corners(obb);
  for (let i = 0; i < 4; i++)
    if (Math.hypot(x-p[i][0], y-p[i][1]) < tol) { drag = {mode:'size'}; return; }
  const [lx, ly] = toLocal(obb, x, y);
  if (Math.abs(lx) <= obb.w/2 && Math.abs(ly) <= obb.h/2)
    drag = {mode:'move', x, y};
};
cv.onmousemove = e => {
  if (!drag) return;
  const [x, y] = pos(e);
  if (drag.mode === 'move') {
    obb.cx += x - drag.x; obb.cy += y - drag.y; drag.x = x; drag.y = y;
  } else if (drag.mode === 'size') {
    const [lx, ly] = toLocal(obb, x, y);
    obb.w = Math.max(8, Math.abs(lx) * 2); obb.h = Math.max(8, Math.abs(ly) * 2);
  } else {
    obb.theta = Math.atan2(y - obb.cy, x - obb.cx) * 180 / Math.PI + 90;
  }
  draw();
};
window.onmouseup = () => { drag = null; };

document.onkeydown = e => {
  if (!obb) { return; }
  const step = e.shiftKey ? 5 : 1;
  const k = e.key;
  if (k === 's') save();
  else if (k === 'x') nocase();
  else if (k === 'r') reset();
  else if (k === 'n') show(idx + 1);
  else if (k === 'p') show(idx - 1);
  else if (k === 'q' || k === 'Q') { obb.theta -= step; draw(); }
  else if (k === 'e' || k === 'E') { obb.theta += step; draw(); }
  else if (k === 'ArrowLeft')  { obb.cx -= step; draw(); }
  else if (k === 'ArrowRight') { obb.cx += step; draw(); }
  else if (k === 'ArrowUp')    { obb.cy -= step; draw(); }
  else if (k === 'ArrowDown')  { obb.cy += step; draw(); }
  else return;
  e.preventDefault();
};

function strip(){
  document.getElementById('strip').innerHTML = rows.map((r, i) =>
    `<div class="${i === idx ? 'cur ' : ''}${r.labeled ? 'done' : ''}" onclick="show(${i})">
       <img src="/img/${r.stem}.png?w=140" loading="lazy">
       <span>${r.stem.split('__').pop()} ${r.reason}</span></div>`).join('');
  document.getElementById('pos').textContent =
    rows.length ? `${idx+1}/${rows.length}` : 'nothing to edit';
}

async function show(i){
  if (i < 0 || i >= rows.length) return;
  idx = i;
  const r = rows[i];
  const d = await (await fetch('/obb/' + r.stem)).json();
  if (d.error) { alert(d.error); return; }
  orig = fromPoints(d.points); obb = Object.assign({}, orig);
  document.getElementById('stem').textContent = r.stem;
  document.getElementById('reason').textContent =
    (d.reason ? '  ' + d.reason : '') + (d.conf ? '  conf ' + d.conf : '');
  img.onload = () => { cv.width = d.img_w; cv.height = d.img_h; draw(); };
  img.src = '/img/' + r.stem + '.png';
  strip();
}
function reset(){ if (orig) { obb = Object.assign({}, orig); draw(); } }

async function post(url, body){
  const r = await (await fetch(url, {method:'POST', body: JSON.stringify(body)})).json();
  if (!r.ok) alert(r.error || 'failed');
  return r.ok;
}
async function save(){
  if (!obb) return;
  if (await post('/save', {stem: rows[idx].stem, points: corners(obb)})) {
    rows[idx].labeled = true; rows[idx].reason = '';
    if (idx + 1 < rows.length) show(idx + 1); else strip();
  }
}
async function nocase(){
  if (!obb) return;
  if (await post('/nocase', {stem: rows[idx].stem})) {
    rows[idx].labeled = true; rows[idx].reason = 'no case';
    if (idx + 1 < rows.length) show(idx + 1); else strip();
  }
}
function setScope(s){
  scope = s;
  document.getElementById('tab-pending').className = 'tab' + (s === 'pending' ? ' on' : '');
  document.getElementById('tab-all').className = 'tab' + (s === 'all' ? ' on' : '');
  load();
}
async function load(){
  rows = await (await fetch('/list?scope=' + scope)).json();
  idx = -1; obb = null;
  if (rows.length) show(0); else { ctx.clearRect(0,0,cv.width,cv.height); strip(); }
}
load();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=f"{cfg.LABELED_DIR}_{TARGET}_bev",
                    help="dataset dir with images/ + labels/ (written to)")
    ap.add_argument("--pending", default=f"pending_{TARGET}_bev",
                    help="dir holding manifest.csv from model_autolabel_bev.py")
    ap.add_argument("--port", type=int, default=8082)
    args = ap.parse_args()

    labeled, pending = HERE / args.labeled, HERE / args.pending
    if not (pending / "manifest.csv").exists() and not (labeled / "labels").exists():
        raise SystemExit(f"Nothing to edit: no {pending/'manifest.csv'} and no "
                         f"{labeled/'labels'}. Run model_autolabel_bev.py first.")
    _Handler.store = Store(labeled, pending)
    n_pending = len(_Handler.store.manifest())
    print(f"pending {n_pending}   labeled {len(list((labeled/'labels').glob('*.txt')))}")
    print(f"open  http://<robot-ip>:{args.port}/     (Ctrl-C to stop)")
    ThreadingHTTPServer(("0.0.0.0", args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
