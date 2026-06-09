"""Standalone web dashboard for the case + battery demo.

Reads the spool written by DashboardPublisher (frame.jpg + state.json) and
serves a browser page that shows the head-camera feed alongside live joint
values, EE pose and the wrist wrench, plus an accumulating time-series chart
of any selected signal. Uses only the Python stdlib, so it needs nothing
installed and runs as its own process:

    # in one terminal — run the demo with publishing on:
    python -m case_battery_demo.run_demo --dashboard

    # in another terminal — serve the viewer:
    python -m case_battery_demo.dashboard.server --spool /tmp/cns_dashboard

Then open http://<robot-ip>:8080/ in a browser.

The chart history is accumulated in the browser, so the server stays a thin
file server. Pointing it at a recorded session directory (same frame.jpg +
state.json layout) is all that a future "saved data" mode needs.
"""

from __future__ import annotations

import argparse
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Case + Battery Demo — Live</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#21262d; --txt:#c9d1d9;
          --muted:#8b949e; --accent:#58a6ff; --good:#3fb950; --warn:#d29922; --bad:#f85149; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { padding:10px 16px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:12px; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  #status { font-size:12px; color:var(--muted); }
  #dot { display:inline-block; width:9px; height:9px; border-radius:50%;
         background:var(--bad); margin-right:6px; vertical-align:middle; }
  .wrap { display:flex; flex-direction:column; gap:14px; padding:14px; }
  /* top row: force / EE pose / joints side by side */
  .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; align-items:start; }
  @media (max-width:900px){ .metrics{ grid-template-columns:1fr; } }
  /* image strip below the metrics */
  .images { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; align-items:start; }
  @media (max-width:1100px){ .images{ grid-template-columns:repeat(2,1fr); } }
  @media (max-width:600px){ .images{ grid-template-columns:1fr; } }
  /* two selectable time-series charts side by side */
  .charts { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
  @media (max-width:900px){ .charts{ grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
             margin:0; padding:8px 12px; border-bottom:1px solid var(--line);
             display:flex; align-items:center; gap:10px; }
  .card .body { padding:10px 12px; }
  #cam, #depth, #detect, #barcode { width:100%; display:block; background:#000; max-height:300px; object-fit:contain; }
  .caption { font-size:11px; color:var(--muted); padding:6px 12px 0;
             text-transform:uppercase; letter-spacing:.05em; }
  table { width:100%; border-collapse:collapse; }
  td { padding:3px 4px; white-space:nowrap; }
  td.k { color:var(--muted); }
  td.v { text-align:right; font-variant-numeric:tabular-nums; }
  .grp { margin-bottom:12px; }
  .grp h3 { font-size:11px; color:var(--accent); margin:0 0 4px; text-transform:uppercase; letter-spacing:.05em; }
  /* joints: pack the groups across the card width and keep each name next to
     its value (auto-width tables instead of full-width right-aligned ones). */
  #joints { display:flex; flex-wrap:wrap; gap:2px 30px; }
  #joints .grp { margin-bottom:6px; }
  #joints table { width:auto; }
  #joints td.k { padding-right:22px; }
  .force-mag { font-size:30px; font-weight:700; font-variant-numeric:tabular-nums; }
  .force-mag .u { font-size:14px; color:var(--muted); font-weight:400; margin-left:4px; }
  .bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; margin:8px 0 12px; }
  .bar > div { height:100%; background:var(--good); width:0%; transition:width .1s linear; }
  .six { display:grid; grid-template-columns:repeat(3,1fr); gap:6px 14px; }
  .six div span { color:var(--muted); }
  .six div b { font-weight:600; font-variant-numeric:tabular-nums; }
  /* time-series chart */
  .chartcard h2 .ctrls { margin-left:auto; display:flex; align-items:center; gap:8px;
                         text-transform:none; letter-spacing:0; }
  select { background:#0d1117; color:var(--txt); border:1px solid var(--line);
           border-radius:5px; padding:3px 6px; font:12px ui-monospace,Menlo,Consolas,monospace; }
  .cur { color:var(--accent); font-weight:600; font-variant-numeric:tabular-nums; min-width:110px; text-align:right; }
  .chart { width:100%; height:200px; display:block; }
</style>
</head>
<body>
<header>
  <h1>Case + Battery Demo</h1>
  <span id="status"><span id="dot"></span><span id="statustxt">connecting…</span></span>
</header>
<div class="wrap">
  <div class="metrics">
    <div class="card">
      <h2>Force / wrench</h2>
      <div class="body">
        <div class="grp">
          <h3>Tared magnitude (suction arm)</h3>
          <div><span class="force-mag" id="fmag">--</span><span class="u">N</span></div>
          <div class="bar"><div id="fbar"></div></div>
          <div class="six">
            <div><span>fx </span><b id="fx">--</b></div>
            <div><span>fy </span><b id="fy">--</b></div>
            <div><span>fz </span><b id="fz">--</b></div>
            <div><span>tx </span><b id="tx">--</b></div>
            <div><span>ty </span><b id="ty">--</b></div>
            <div><span>tz </span><b id="tz">--</b></div>
          </div>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>EE pose <span id="eeframe" style="text-transform:none;color:var(--muted)"></span></h2>
      <div class="body"><table id="ee"></table></div>
    </div>
    <div class="card">
      <h2>Joints (deg)</h2>
      <div class="body" id="joints"></div>
    </div>
  </div>

  <div class="images">
    <div class="card">
      <h2>Head camera</h2>
      <img id="cam" alt="waiting for RGB…"/>
    </div>
    <div class="card">
      <h2>Depth</h2>
      <img id="depth" alt="waiting for depth…"/>
      <div class="caption" id="depthcap">depth (m, near→far)</div>
    </div>
    <div class="card">
      <h2>Bin detection</h2>
      <img id="detect" alt="detector not running"/>
      <div class="caption" id="detectcap">bin detection</div>
    </div>
    <div class="card">
      <h2>Barcode reader</h2>
      <img id="barcode" alt="waiting for reader…"/>
      <div class="caption" id="barcodecap">cognex · waiting…</div>
    </div>
  </div>

  <div class="charts">
    <div class="card chartcard">
      <h2>Time series A
        <span class="ctrls">
          <select id="signal0" title="signal to plot"></select>
          <select id="window0" title="time window">
            <option value="30">30 s</option>
            <option value="60" selected>1 min</option>
            <option value="120">2 min</option>
            <option value="300">5 min</option>
          </select>
          <span class="cur" id="cur0">--</span>
        </span>
      </h2>
      <div class="body"><canvas id="chart0" class="chart"></canvas></div>
    </div>
    <div class="card chartcard">
      <h2>Time series B
        <span class="ctrls">
          <select id="signal1" title="signal to plot"></select>
          <select id="window1" title="time window">
            <option value="30">30 s</option>
            <option value="60" selected>1 min</option>
            <option value="120">2 min</option>
            <option value="300">5 min</option>
          </select>
          <span class="cur" id="cur1">--</span>
        </span>
      </h2>
      <div class="body"><canvas id="chart1" class="chart"></canvas></div>
    </div>
  </div>
</div>
<script>
const FORCE_FULL_SCALE = 15.0;   // N — bar fills to FORCE_HARD_LIMIT_N
const HISTORY_S = 300;           // keep up to 5 min of history client-side
const $ = id => document.getElementById(id);
let lastSeq = -1;

function fmt(x, d=3){ return (x===null||x===undefined||Number.isNaN(x)) ? "--" : Number(x).toFixed(d); }
function deg(r){ return r * 180 / Math.PI; }

function setOnline(on, ageMs){
  $("dot").style.background = on ? "var(--good)" : "var(--bad)";
  $("statustxt").textContent = on ? ("live · " + (ageMs/1000).toFixed(1) + "s ago") : "no data";
}

function renderEE(ee){
  if(!ee){ $("ee").innerHTML = "<tr><td class='k'>unavailable</td></tr>"; $("eeframe").textContent=""; return; }
  $("eeframe").textContent = "(" + ee.frame + ", base_link)";
  const p = ee.pos, r = ee.rpy;
  $("ee").innerHTML =
    `<tr><td class='k'>x / y / z (m)</td><td class='v'>${fmt(p[0])} &nbsp; ${fmt(p[1])} &nbsp; ${fmt(p[2])}</td></tr>` +
    `<tr><td class='k'>roll/pitch/yaw (deg)</td><td class='v'>${fmt(deg(r[0]),1)} &nbsp; ${fmt(deg(r[1]),1)} &nbsp; ${fmt(deg(r[2]),1)}</td></tr>`;
}

function renderJoints(joints){
  if(!joints){ $("joints").innerHTML=""; return; }
  let html = "";
  for(const comp of ["left_arm","right_arm","torso","head"]){
    if(!joints[comp]) continue;
    html += `<div class='grp'><h3>${comp.replace('_',' ')}</h3><table>`;
    for(const [name, val] of Object.entries(joints[comp])){
      html += `<tr><td class='k'>${name}</td><td class='v'>${fmt(deg(val),1)}</td></tr>`;
    }
    html += `</table></div>`;
  }
  $("joints").innerHTML = html;
}

function renderForce(w){
  if(!w){ $("fmag").textContent="--"; return; }
  const mag = (w.tared_mag!==null && w.tared_mag!==undefined) ? w.tared_mag : w.raw_mag;
  $("fmag").textContent = fmt(mag, 2);
  const pct = Math.max(0, Math.min(100, 100*mag/FORCE_FULL_SCALE));
  const bar = $("fbar");
  bar.style.width = pct + "%";
  bar.style.background = mag > FORCE_FULL_SCALE ? "var(--bad)" : (mag > 0.66*FORCE_FULL_SCALE ? "var(--warn)" : "var(--good)");
  $("fx").textContent=fmt(w.fx,2); $("fy").textContent=fmt(w.fy,2); $("fz").textContent=fmt(w.fz,2);
  $("tx").textContent=fmt(w.tx,2); $("ty").textContent=fmt(w.ty,2); $("tz").textContent=fmt(w.tz,2);
}

// ---- time-series: flatten a state snapshot into selectable signals --------
function flatten(s){
  const out = {};  // path -> {group, label, unit, value}
  const w = s.wrench;
  if(w){
    if(w.tared_mag!==null && w.tared_mag!==undefined)
      out["force.tared_mag"] = {group:"Force", label:"|F| tared", unit:"N", value:w.tared_mag};
    if(w.raw_mag!==undefined) out["force.raw_mag"] = {group:"Force", label:"|F| raw", unit:"N", value:w.raw_mag};
    for(const k of ["fx","fy","fz"]) if(w[k]!==undefined) out["force."+k]={group:"Force", label:k, unit:"N", value:w[k]};
    for(const k of ["tx","ty","tz"]) if(w[k]!==undefined) out["force."+k]={group:"Force", label:k, unit:"N·m", value:w[k]};
  }
  const ee = s.ee;
  if(ee){
    out["ee.x"]={group:"EE pose", label:"x", unit:"m", value:ee.pos[0]};
    out["ee.y"]={group:"EE pose", label:"y", unit:"m", value:ee.pos[1]};
    out["ee.z"]={group:"EE pose", label:"z", unit:"m", value:ee.pos[2]};
    out["ee.roll"]={group:"EE pose", label:"roll", unit:"°", value:deg(ee.rpy[0])};
    out["ee.pitch"]={group:"EE pose", label:"pitch", unit:"°", value:deg(ee.rpy[1])};
    out["ee.yaw"]={group:"EE pose", label:"yaw", unit:"°", value:deg(ee.rpy[2])};
  }
  const j = s.joints || {};
  for(const comp of Object.keys(j)){
    for(const [name, val] of Object.entries(j[comp])){
      out["joints."+comp+"."+name] = {group:"Joints · "+comp.replace('_',' '), label:name, unit:"°", value:deg(val)};
    }
  }
  return out;
}

const samples = [];     // {t: epoch_seconds, vals:{path:number}}
const knownSig = {};    // path -> {group,label,unit}  (union over time, never shrinks)
let sigKey = "";

// Two independent charts. A defaults to tared |F|, B to raw |F|; each keeps its
// own selected signal + time window, chosen from the same shared signal list.
const CHARTS = [
  {canvas:"chart0", sig:"signal0", win:"window0", cur:"cur0", def:"force.tared_mag", selected:null},
  {canvas:"chart1", sig:"signal1", win:"window1", cur:"cur1", def:"force.raw_mag",   selected:null},
];

function ensureSelected(ch){
  if(ch.selected && knownSig[ch.selected]) return;
  ch.selected = knownSig[ch.def] ? ch.def : (Object.keys(knownSig)[0] || null);
}

function fillSelect(sel){
  const groups = {};
  for(const [path, m] of Object.entries(knownSig)) (groups[m.group] ||= []).push([path, m.label]);
  sel.innerHTML = "";
  for(const g of Object.keys(groups)){
    const og = document.createElement("optgroup"); og.label = g;
    for(const [path, label] of groups[g]){
      const o = document.createElement("option"); o.value = path; o.textContent = label; og.appendChild(o);
    }
    sel.appendChild(og);
  }
}

function rebuildSelect(){
  for(const ch of CHARTS){
    fillSelect($(ch.sig));
    ensureSelected(ch);
    if(ch.selected) $(ch.sig).value = ch.selected;
  }
}

function drawChart(ch){
  const c = $(ch.canvas); if(!c) return;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  c.width = W*dpr; c.height = H*dpr;
  const ctx = c.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  const padL=56, padR=12, padT=10, padB=22;
  const x0=padL, x1=W-padR, y0=padT, y1=H-padB;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1; ctx.strokeRect(x0,y0,x1-x0,y1-y0);
  if(!ch.selected) return;
  const meta = knownSig[ch.selected] || {};
  const winS = parseInt($(ch.win).value, 10);
  const now = samples.length ? samples[samples.length-1].t : Date.now()/1000;
  const tmin = now - winS;
  const pts = [];
  for(const s of samples){
    if(s.t < tmin) continue;
    const v = s.vals[ch.selected];
    if(v!==undefined && v!==null && isFinite(v)) pts.push([s.t, v]);
  }
  // x time labels regardless of data
  ctx.fillStyle="#8b949e"; ctx.font="11px ui-monospace,monospace";
  ctx.textAlign="center"; ctx.textBaseline="top";
  for(let i=0;i<=4;i++){ const frac=i/4; const ago=Math.round(winS*(1-frac));
    ctx.fillText(ago===0?"now":("-"+ago+"s"), x0+frac*(x1-x0), y1+5); }
  if(pts.length < 2){
    ctx.fillStyle="#8b949e"; ctx.textAlign="left"; ctx.textBaseline="middle";
    ctx.fillText("collecting…", x0+8, (y0+y1)/2);
    $(ch.cur).textContent = "--";
    return;
  }
  let vmin=Infinity, vmax=-Infinity;
  for(const [,v] of pts){ if(v<vmin)vmin=v; if(v>vmax)vmax=v; }
  if(vmin===vmax){ vmin-=1; vmax+=1; }
  const pad=(vmax-vmin)*0.08; vmin-=pad; vmax+=pad;
  const sx = t => x0 + (t-tmin)/winS*(x1-x0);
  const sy = v => y1 - (v-vmin)/(vmax-vmin)*(y1-y0);
  // y gridlines + labels
  ctx.textAlign="right"; ctx.textBaseline="middle";
  for(let i=0;i<=4;i++){ const v=vmin+(vmax-vmin)*i/4, y=sy(v);
    ctx.strokeStyle="#161b22"; ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillStyle="#8b949e"; ctx.fillText(v.toFixed(2), x0-6, y); }
  // series
  ctx.strokeStyle="#58a6ff"; ctx.lineWidth=1.5; ctx.beginPath();
  pts.forEach(([t,v],i)=>{ const x=sx(t), y=sy(v); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke();
  const [lt,lv]=pts[pts.length-1];
  ctx.fillStyle="#58a6ff"; ctx.beginPath(); ctx.arc(sx(lt),sy(lv),3,0,2*Math.PI); ctx.fill();
  $(ch.cur).textContent = lv.toFixed(3) + (meta.unit ? (" "+meta.unit) : "");
}

function drawAll(){ for(const ch of CHARTS) drawChart(ch); }

function accumulate(s){
  const flat = flatten(s);
  const vals = {};
  for(const [path, m] of Object.entries(flat)){
    vals[path] = m.value;
    if(!knownSig[path]) knownSig[path] = {group:m.group, label:m.label, unit:m.unit};
  }
  const k = Object.keys(knownSig).sort().join("|");
  if(k !== sigKey){ sigKey = k; rebuildSelect(); }
  const t = s.stamp || Date.now()/1000;
  samples.push({t, vals});
  const cut = t - HISTORY_S;
  while(samples.length && samples[0].t < cut) samples.shift();
}

async function tick(){
  try{
    const r = await fetch("/state.json", {cache:"no-store"});
    if(!r.ok) throw new Error("no state");
    const s = await r.json();
    const ageMs = Date.now() - (s.stamp*1000);
    setOnline(ageMs < 2000, ageMs);
    if(s.seq !== lastSeq){
      lastSeq = s.seq;
      if(s.has_image) $("cam").src = "/frame.jpg?seq=" + s.seq;
      if(s.has_depth) $("depth").src = "/depth.jpg?seq=" + s.seq;
    }
    if(s.depth_range_m) $("depthcap").textContent =
      `depth (${s.depth_range_m[0]}–${s.depth_range_m[1]} m, near→far)`;
    renderForce(s.wrench);
    renderEE(s.ee);
    renderJoints(s.joints);
    accumulate(s);
    drawAll();
  }catch(e){
    setOnline(false, 0);
  }
}
// detection overlay: separate process / cadence, so poll it independently.
let detectSeq = -1;
async function tickDetect(){
  try{
    const r = await fetch("/detect.json", {cache:"no-store"});
    if(!r.ok) throw new Error("no detector");
    const d = await r.json();
    if(d.seq !== detectSeq){ detectSeq = d.seq; $("detect").src = "/detect.jpg?seq=" + d.seq; }
    const age = ((Date.now() - d.stamp*1000)/1000).toFixed(1);
    $("detectcap").textContent = d.found
      ? `bin detection · conf ${d.conf} · ${age}s ago`
      : `bin detection · no bin · ${age}s ago`;
  }catch(e){
    $("detectcap").textContent = "bin detection · detector not running";
  }
}

// barcode reader: independent process/cadence (1 Hz image-only pull).
let barcodeSeq = -1;
async function tickBarcode(){
  try{
    const r = await fetch("/barcode.json", {cache:"no-store"});
    if(!r.ok) throw new Error("no reader");
    const d = await r.json();
    if(d.ok && d.seq !== barcodeSeq){ barcodeSeq = d.seq; $("barcode").src = "/barcode.jpg?seq=" + d.seq; }
    const age = ((Date.now() - d.stamp*1000)/1000).toFixed(1);
    $("barcodecap").textContent = d.ok
      ? `cognex · ${age}s ago`
      : `cognex · ${d.error||"no image"} · ${age}s ago`;
  }catch(e){
    $("barcodecap").textContent = "cognex · reader not running";
  }
}

for(const ch of CHARTS){
  $(ch.sig).addEventListener("change", e => { ch.selected = e.target.value; drawChart(ch); });
  $(ch.win).addEventListener("change", () => drawChart(ch));
}
window.addEventListener("resize", drawAll);
setInterval(tick, 100);
setInterval(tickDetect, 200);
setInterval(tickBarcode, 1000);
tick();
tickDetect();
tickBarcode();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    spool_dir = "/tmp/cns_dashboard"

    def _send(self, code: int, content_type: str, body: bytes, no_store: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self._send(404, "text/plain", b"not found yet", no_store=True)
            return
        self._send(200, content_type, body, no_store=True)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
        elif path == "/state.json":
            self._send_file(os.path.join(self.spool_dir, "state.json"), "application/json")
        elif path == "/frame.jpg":
            self._send_file(os.path.join(self.spool_dir, "frame.jpg"), "image/jpeg")
        elif path == "/depth.jpg":
            self._send_file(os.path.join(self.spool_dir, "depth.jpg"), "image/jpeg")
        elif path == "/detect.jpg":
            self._send_file(os.path.join(self.spool_dir, "detect.jpg"), "image/jpeg")
        elif path == "/detect.json":
            self._send_file(os.path.join(self.spool_dir, "detect.json"), "application/json")
        elif path == "/barcode.jpg":
            self._send_file(os.path.join(self.spool_dir, "barcode.jpg"), "image/jpeg")
        elif path == "/barcode.json":
            self._send_file(os.path.join(self.spool_dir, "barcode.json"), "application/json")
        else:
            self._send(404, "text/plain", b"not found")

    do_HEAD = do_GET

    def log_message(self, *args) -> None:  # silence per-request logging
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Case+battery demo web dashboard")
    parser.add_argument("--spool", default="/tmp/cns_dashboard",
                        help="directory the demo publisher writes frame.jpg + state.json to")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8080, help="port (default: 8080)")
    parser.add_argument("--no-launch-camera", action="store_true",
                        help="don't auto-launch the head_camera dexsensor on the nano over SSH")
    parser.add_argument("--camera-host", default=None,
                        help="override the nano SSH target (default: dexmate-nano@192.168.50.22)")
    args = parser.parse_args()

    if not args.no_launch_camera:
        from .camera_launch import NANO_HOST, ensure_camera
        ensure_camera(host=args.camera_host or NANO_HOST)

    _Handler.spool_dir = os.path.abspath(args.spool)
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    shown_host = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"Dashboard serving {_Handler.spool_dir}")
    print(f"  open  http://{shown_host}:{args.port}/   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
