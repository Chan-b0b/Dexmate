"""Offline review dashboard for recorded takes (the sibling of server.py).

server.py polls a single live spool; this serves random access over the many
*immutable* takes written by recorder.py to a recordings dir, so you can review
them before they become VLA training data:

  * gallery  GET /            — every take as a card (thumb, instruction, frames,
                                duration, dropped-frame badge, success/fail chip)
  * player   GET /take/<name> — scrub one take frame-by-frame with RGB + colorized
                                depth + synced wrench / EE / joints / suction and
                                whole-episode timeline charts with a playhead

It is stdlib-only for the HTTP layer (like server.py), plus cv2/numpy for depth
colorizing (already deps). Run it as its own process:

    python -m case_battery_demo.dashboard.review_server --root LGES/recordings
    #   open  http://<robot-ip>:8081/

Curation lives in a sidecar this server owns (``<root>/.review.json``); deletes
move a take to ``<root>/.trash/`` (reversible). Take contents are never mutated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Same depth mapping the recorder uses for depth_preview.jpg, so scrubbed depth
# matches the saved preview. One source of truth.
from .recorder import _DEPTH_CMAP, _PREVIEW_DEPTH_RANGE_M

CACHE_DIR = "/tmp/cns_review_cache"
SIDECAR = ".review.json"
TRASH = ".trash"
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")   # take dir names; rejects path traversal


# ---- take discovery / disk helpers ----------------------------------------
# The recorder groups takes by task: <root>/<phase|manual>/<take>/. Take names
# stay globally unique (timestamp + ep counter), so they remain the identifier
# everywhere (URLs, ratings sidecar, depth cache, trash) and only the on-disk
# location gains a task level. Legacy flat takes (<root>/<take>/) still work.

def _ok_name(name: str) -> bool:
    return bool(_NAME_RE.match(name)) and name not in (TRASH, ".pending")


def _discover(root: str) -> dict[str, str]:
    """Map take name -> task subfolder ('' for legacy flat takes)."""
    out: dict[str, str] = {}
    try:
        entries = os.listdir(root)
    except OSError:
        return out
    for e in entries:
        if not _ok_name(e):
            continue
        p = os.path.join(root, e)
        if os.path.isfile(os.path.join(p, "meta.json")):
            out[e] = ""          # legacy flat take
        elif os.path.isdir(p):
            try:
                subs = os.listdir(p)
            except OSError:
                continue
            for n in subs:
                if _ok_name(n) and os.path.isfile(os.path.join(p, n, "meta.json")):
                    out[n] = e
    return out


def _find_take(root: str, name: str) -> str | None:
    """Return the take's directory (flat or one task-folder deep), or None."""
    if not _ok_name(name):
        return None
    p = os.path.join(root, name)
    if os.path.isfile(os.path.join(p, "meta.json")):
        return p
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    for task in entries:
        if not _ok_name(task):
            continue
        p = os.path.join(root, task, name)
        if os.path.isfile(os.path.join(p, "meta.json")):
            return p
    return None


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _ratings(root: str) -> dict:
    return _read_json(os.path.join(root, SIDECAR))


def _write_ratings(root: str, data: dict) -> None:
    path = os.path.join(root, SIDECAR)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _quat_to_rpy(q) -> list[float]:
    """quat_wxyz -> [roll, pitch, yaw] (xyz euler), inverse of the recorder."""
    w, x, y, z = q
    rpy = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
    return [float(v) for v in rpy]


# Curation is a single success/fail label per take. The recorder's meta.json
# success flag is the starting point (null/missing counts as success); a click
# in the player writes an override to the sidecar — take files are never
# mutated. Legacy good/bad sidecar entries map to success/fail.
_LEGACY_RATING = {"good": "success", "bad": "fail"}


def _label(meta: dict, ratings: dict, name: str) -> str:
    override = ratings.get(name, {}).get("rating", "")
    override = _LEGACY_RATING.get(override, override)
    if override in ("success", "fail"):
        return override
    return "fail" if meta.get("success") is False else "success"


def _take_summary(tdir: str, name: str, task: str, ratings: dict) -> dict:
    meta = _read_json(os.path.join(tdir, "meta.json"))
    return {
        "name": name,
        "task": task,
        "created": meta.get("created", ""),
        "instruction": meta.get("instruction", ""),
        "frames": meta.get("frames", 0),
        "duration_s": meta.get("duration_s", 0.0),
        "dropped_frames": meta.get("dropped_frames", 0),
        "label": _label(meta, ratings, name),
    }


def _load_states(tdir: str) -> list[dict]:
    """Parse states.jsonl, adding ee.rpy (deg-free) for display."""
    out = []
    path = os.path.join(tdir, "states.jsonl")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for key in ("ee", "ee_right"):
                    ee = row.get(key)
                    if ee and "quat_wxyz" in ee:
                        ee["rpy"] = _quat_to_rpy(ee["quat_wxyz"])
                out.append(row)
    except (OSError, ValueError):
        pass
    return out


# ---- depth colorizing (cached) ---------------------------------------------

def _colorize_depth(tdir: str, name: str, idx: int) -> bytes | None:
    cache = os.path.join(CACHE_DIR, name, f"{idx:06d}.jpg")
    if os.path.isfile(cache):
        try:
            with open(cache, "rb") as f:
                return f.read()
        except OSError:
            pass
    src = os.path.join(tdir, "head_depth", f"{idx:06d}.png")
    d = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / 1000.0   # mm -> m
    near, far = _PREVIEW_DEPTH_RANGE_M
    valid = d > 0.0
    norm = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
    norm[~valid] = 0.0
    color = cv2.applyColorMap((norm * 255.0).astype(np.uint8), _DEPTH_CMAP)
    color[~valid] = (0, 0, 0)
    ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    body = buf.tobytes()
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            f.write(body)
    except OSError:
        pass
    return body


# ---- shared CSS (matches server.py's dark GitHub theme) --------------------

_CSS = """
  :root { --bg:#0d1117; --panel:#161b22; --line:#21262d; --txt:#c9d1d9;
          --muted:#8b949e; --accent:#58a6ff; --good:#3fb950; --warn:#d29922; --bad:#f85149; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  a { color:var(--accent); text-decoration:none; }
  header { padding:10px 28px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .sub { font-size:12px; color:var(--muted); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
             margin:0; padding:8px 12px; border-bottom:1px solid var(--line); }
  .card .body { padding:10px 12px; }
  table { width:100%; border-collapse:collapse; }
  td { padding:3px 4px; white-space:nowrap; }
  td.k { color:var(--muted); } td.v { text-align:right; font-variant-numeric:tabular-nums; }
  .chip { font-size:11px; padding:2px 8px; border-radius:10px; border:1px solid var(--line);
          text-transform:uppercase; letter-spacing:.04em; }
  .chip.good { color:var(--good); border-color:var(--good); }
  .chip.bad  { color:var(--bad);  border-color:var(--bad); }
  .chip.warn { color:var(--warn); border-color:var(--warn); }
  .btn { background:#21262d; color:var(--txt); border:1px solid var(--line); border-radius:6px;
         padding:7px 14px; cursor:pointer; font:13px/1 ui-monospace,Menlo,Consolas,monospace; font-weight:600; }
  .btn:hover { border-color:var(--accent); }
  .btn.good { border-color:var(--good); color:var(--good); }
  .btn.bad  { border-color:var(--bad);  color:var(--bad); }
  .btn.on   { background:var(--accent); border-color:var(--accent); color:#0d1117; }
  .btn.good.on { background:var(--good); border-color:var(--good); color:#0d1117; }
  .btn.bad.on  { background:var(--bad);  border-color:var(--bad);  color:#0d1117; }
"""

GALLERY_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Recordings — Review</title>
<style>__CSS__
  .wrap { padding:16px 40px; }
  .filters { display:flex; gap:8px; margin-bottom:16px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }
  .take { display:block; }
  .take img { width:100%; aspect-ratio:16/9; object-fit:cover; background:#000; display:block; }
  .take .meta { padding:10px 12px; }
  .take .name { font-weight:600; color:var(--accent); }
  .take .instr { color:var(--txt); margin:4px 0; white-space:normal; }
  .take .nums { color:var(--muted); font-size:12px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .empty { color:var(--muted); padding:40px; text-align:center; }
</style></head><body>
<header><h1>Recordings</h1><span class="sub" id="sub">loading…</span></header>
<div class="wrap">
  <div class="filters">
    <button class="btn on" data-f="all">All</button>
    <button class="btn" data-f="success">Success</button>
    <button class="btn" data-f="fail">Fail</button>
    <select id="taskf" style="background:#0d1117;color:var(--txt);border:1px solid var(--line);
            border-radius:6px;padding:7px 10px;font:13px ui-monospace,Menlo,Consolas,monospace;">
      <option value="">all tasks</option>
    </select>
  </div>
  <div class="grid" id="grid"></div>
</div>
<script>
const $ = id => document.getElementById(id);
let takes = [], filter = "all", taskFilter = "";
function esc(s){ return String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])); }
function matches(t){
  if(taskFilter && (t.task||"") !== taskFilter) return false;
  if(filter==="all") return true;
  return t.label===filter;
}
function render(){
  const g = $("grid");
  const shown = takes.filter(matches);
  if(!shown.length){ g.innerHTML = "<div class='empty'>no takes</div>"; return; }
  g.innerHTML = shown.map(t => {
    const drop = t.dropped_frames>0 ? `<span class="chip warn">⚠ ${t.dropped_frames} dropped</span>` : "";
    const task = t.task ? `<span class="chip">${esc(t.task)}</span>` : "";
    const label = `<span class="chip ${t.label==="fail"?"bad":"good"}">${t.label}</span>`;
    return `<a class="card take" href="/take/${encodeURIComponent(t.name)}">
      <img loading="lazy" src="/thumb/${encodeURIComponent(t.name)}" alt=""/>
      <div class="meta">
        <div class="name">${esc(t.name)}</div>
        <div class="instr">${esc(t.instruction)}</div>
        <div class="nums">${t.frames} frames · ${Number(t.duration_s).toFixed(1)}s ${task} ${label} ${drop}</div>
      </div></a>`;
  }).join("");
}
async function load(){
  const r = await fetch("/api/takes",{cache:"no-store"});
  takes = await r.json();
  $("sub").textContent = takes.length + " takes";
  const tasks = [...new Set(takes.map(t => t.task||""))].filter(Boolean).sort();
  $("taskf").innerHTML = '<option value="">all tasks</option>'
    + tasks.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join("");
  $("taskf").value = taskFilter;
  render();
}
$("taskf").addEventListener("change", () => { taskFilter = $("taskf").value; render(); });
for(const b of document.querySelectorAll(".filters .btn")){
  b.addEventListener("click", () => {
    filter = b.dataset.f;
    for(const o of document.querySelectorAll(".filters .btn")) o.classList.toggle("on", o===b);
    render();
  });
}
// Re-fetch labels whenever the gallery is shown again — coming back from the
// player (incl. browser Back / bfcache restore) or refocusing the tab — so a
// take just tagged fail no longer shows its stale success label. The selected
// filters live in JS state, so they survive the refresh.
load();
window.addEventListener("pageshow", load);
document.addEventListener("visibilitychange", () => { if(!document.hidden) load(); });
</script></body></html>
"""

PLAYER_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Review — take</title>
<style>__CSS__
  .wrap { display:flex; flex-direction:column; gap:14px; padding:14px 40px; }
  .top { display:grid; grid-template-columns:1.4fr 1.4fr 1fr; gap:14px; }
  @media (max-width:1100px){ .top{ grid-template-columns:1fr; } }
  #rgb, #depth { width:100%; display:block; background:#000; object-fit:contain; max-height:46vh; }
  .caption { font-size:11px; color:var(--muted); padding:6px 12px; }
  .panel .grp { margin-bottom:12px; }
  .panel h3 { font-size:11px; color:var(--accent); margin:0 0 4px; text-transform:uppercase; letter-spacing:.05em; }
  .force-mag { font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }
  .force-mag .u { font-size:13px; color:var(--muted); font-weight:400; margin-left:4px; }
  .bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; margin:6px 0 10px; }
  .bar > div { height:100%; background:var(--good); width:0%; }
  .six { display:grid; grid-template-columns:repeat(3,1fr); gap:6px 14px; }
  .six div span { color:var(--muted); } .six div b { font-weight:600; font-variant-numeric:tabular-nums; }
  #joints { display:flex; flex-wrap:wrap; gap:2px 24px; } #joints table{ width:auto; } #joints td.k{ padding-right:18px; }
  .suction { font-size:16px; font-weight:700; }
  .suction.on { color:var(--good); } .suction.off { color:var(--muted); }
  /* transport */
  .transport { display:flex; align-items:center; gap:10px; }
  .transport .btn { padding:8px 12px; }
  /* scrub + suction strip share the full card width so a spot on the strip
     lines up with the same spot on the scrub bar. */
  #scrub { width:100%; display:block; margin:8px 0 0; padding:0; }
  .tinfo { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; min-width:200px; text-align:right; }
  /* fixed-size thumb so the strip below can inset to match its travel */
  input[type=range]{ -webkit-appearance:none; appearance:none; height:6px; background:var(--line);
                     border-radius:3px; cursor:pointer; }
  input[type=range]::-webkit-slider-thumb{ -webkit-appearance:none; width:12px; height:12px;
                     border-radius:50%; background:var(--accent); cursor:pointer; }
  input[type=range]::-moz-range-thumb{ width:12px; height:12px; border:none; border-radius:50%;
                     background:var(--accent); cursor:pointer; }
  select { background:#0d1117; color:var(--txt); border:1px solid var(--line); border-radius:5px;
           padding:3px 6px; font:12px ui-monospace,Menlo,Consolas,monospace; }
  .charts { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
  @media (max-width:900px){ .charts{ grid-template-columns:1fr; } }
  .chartcard h2 { display:flex; align-items:center; gap:8px; }
  .chartcard h2 .ctrls { margin-left:auto; text-transform:none; letter-spacing:0; }
  .cur { color:var(--accent); font-weight:600; min-width:96px; text-align:right; font-variant-numeric:tabular-nums; }
  .chart { width:100%; height:170px; display:block; cursor:crosshair; }
  #suctionstrip { width:100%; height:22px; display:block; }
</style></head><body>
<header>
  <a href="/" class="btn">← all</a>
  <h1 id="name">…</h1>
  <span class="sub" id="instr"></span>
  <span style="margin-left:auto; display:flex; gap:8px;">
    <button class="btn good" id="goodbtn">Success</button>
    <button class="btn bad" id="badbtn">Fail</button>
    <button class="btn bad" id="delbtn">Delete</button>
  </span>
</header>
<div class="wrap">
  <div class="top">
    <div class="card"><h2>Head camera</h2><img id="rgb" alt=""/></div>
    <div class="card"><h2>Depth</h2><img id="depth" alt=""/>
      <div class="caption" id="depthcap"></div></div>
    <div class="card panel"><div class="body">
      <div class="grp"><h3>Wrench (suction arm)</h3>
        <div><span class="force-mag" id="fmag">--</span><span class="u">N tared</span></div>
        <div class="bar"><div id="fbar"></div></div>
        <div class="six">
          <div><span>fx </span><b id="fx">--</b></div><div><span>fy </span><b id="fy">--</b></div>
          <div><span>fz </span><b id="fz">--</b></div><div><span>tx </span><b id="tx">--</b></div>
          <div><span>ty </span><b id="ty">--</b></div><div><span>tz </span><b id="tz">--</b></div>
        </div>
      </div>
      <div class="grp"><h3>Suction</h3><div class="suction off" id="suction">○ off</div></div>
      <div class="grp"><h3>Gripper</h3><div class="suction off" id="gripper">n/a</div></div>
      <div class="grp"><h3>EE — L gripper</h3><table id="ee"></table></div>
      <div class="grp"><h3>EE — R gripper</h3><table id="ee_r"></table></div>
    </div></div>
  </div>

  <div class="card"><div class="body">
    <div class="transport">
      <button class="btn" id="first">⏮</button>
      <button class="btn" id="prev">◀</button>
      <button class="btn" id="play">▶</button>
      <button class="btn" id="next">▶</button>
      <button class="btn" id="last">⏭</button>
      <select id="speed">
        <option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option>
        <option value="1">1×</option><option value="2">2×</option><option value="4">4×</option>
      </select>
      <span class="tinfo" id="tinfo">--</span>
    </div>
    <input type="range" id="scrub" min="0" max="0" value="0"/>
    <canvas id="suctionstrip"></canvas>
    <div class="caption">suction timeline (green = on)</div>
  </div></div>

  <div class="card panel"><div class="body"><div class="grp"><h3>Joints (deg)</h3><div id="joints"></div></div></div></div>

  <div class="charts">
    <div class="card chartcard"><h2>Timeline A
      <span class="ctrls"><select id="sig0"></select> <span class="cur" id="cur0">--</span></span></h2>
      <div class="body"><canvas id="chart0" class="chart"></canvas></div></div>
    <div class="card chartcard"><h2>Timeline B
      <span class="ctrls"><select id="sig1"></select> <span class="cur" id="cur1">--</span></span></h2>
      <div class="body"><canvas id="chart1" class="chart"></canvas></div></div>
  </div>
</div>
<script>
const FORCE_FULL_SCALE = 15.0;
const $ = id => document.getElementById(id);
const NAME = decodeURIComponent(location.pathname.replace(/^\\/take\\//, ""));
function fmt(x,d=3){ return (x===null||x===undefined||Number.isNaN(x))?"--":Number(x).toFixed(d); }
function deg(r){ return r*180/Math.PI; }
function esc(s){ return String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])); }

let frames = [], t0 = 0, cur = 0, playing = false, playTimer = null;

// ---- flatten one frame into selectable signals (mirrors server.py) --------
function flatten(s){
  const out = {};
  const w = s.wrench;
  if(w){
    if(w.tared_mag!=null) out["force.tared_mag"]={group:"Force",label:"|F| tared",unit:"N",value:w.tared_mag};
    if(w.raw_mag!=null) out["force.raw_mag"]={group:"Force",label:"|F| raw",unit:"N",value:w.raw_mag};
    for(const k of ["fx","fy","fz"]) if(w[k]!=null) out["force."+k]={group:"Force",label:k,unit:"N",value:w[k]};
    for(const k of ["tx","ty","tz"]) if(w[k]!=null) out["force."+k]={group:"Force",label:k,unit:"N·m",value:w[k]};
  }
  const ee = s.ee;
  if(ee){
    out["ee.x"]={group:"EE pose",label:"x",unit:"m",value:ee.pos[0]};
    out["ee.y"]={group:"EE pose",label:"y",unit:"m",value:ee.pos[1]};
    out["ee.z"]={group:"EE pose",label:"z",unit:"m",value:ee.pos[2]};
    if(ee.rpy){
      out["ee.roll"]={group:"EE pose",label:"roll",unit:"°",value:deg(ee.rpy[0])};
      out["ee.pitch"]={group:"EE pose",label:"pitch",unit:"°",value:deg(ee.rpy[1])};
      out["ee.yaw"]={group:"EE pose",label:"yaw",unit:"°",value:deg(ee.rpy[2])};
    }
  }
  out["suction"]={group:"Other",label:"suction",unit:"",value:s.suction_cmd?1:0};
  if(s.gripper_pos!=null) out["gripper.pos"]={group:"Other",label:"gripper pos",unit:"",value:s.gripper_pos};
  const j = s.joints||{};
  for(const comp of Object.keys(j))
    for(const [nm,val] of Object.entries(j[comp]))
      out["joints."+comp+"."+nm]={group:"Joints · "+comp.replace('_',' '),label:nm,unit:"°",value:deg(val)};
  return out;
}
let knownSig = {};
function buildSignals(){
  knownSig = {};
  for(const s of frames) for(const [p,m] of Object.entries(flatten(s)))
    if(!knownSig[p]) knownSig[p]={group:m.group,label:m.label,unit:m.unit};
}
function fillSelect(sel, def){
  const groups = {};
  for(const [p,m] of Object.entries(knownSig)) (groups[m.group]||=[]).push([p,m.label]);
  sel.innerHTML="";
  for(const g of Object.keys(groups)){
    const og=document.createElement("optgroup"); og.label=g;
    for(const [p,label] of groups[g]){ const o=document.createElement("option"); o.value=p; o.textContent=label; og.appendChild(o); }
    sel.appendChild(og);
  }
  if(knownSig[def]) sel.value=def;
}

// ---- per-frame panels (mirror server.py renderers) ------------------------
function renderForce(w){
  if(!w){ $("fmag").textContent="--"; return; }
  const mag = w.tared_mag!=null ? w.tared_mag : w.raw_mag;
  $("fmag").textContent=fmt(mag,2);
  const pct=Math.max(0,Math.min(100,100*mag/FORCE_FULL_SCALE)), bar=$("fbar");
  bar.style.width=pct+"%";
  bar.style.background = mag>FORCE_FULL_SCALE?"var(--bad)":(mag>0.66*FORCE_FULL_SCALE?"var(--warn)":"var(--good)");
  for(const k of ["fx","fy","fz","tx","ty","tz"]) $(k).textContent=fmt(w[k],2);
}
function renderEE(ee, tableId){
  const tbl=$(tableId);
  if(!ee){ tbl.innerHTML="<tr><td class='k'>unavailable</td></tr>"; return; }
  const p=ee.pos, r=ee.rpy||[0,0,0];
  tbl.innerHTML =
    `<tr><td class='k'>x/y/z (m)</td><td class='v'>${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])}</td></tr>`+
    `<tr><td class='k'>r/p/y (deg)</td><td class='v'>${fmt(deg(r[0]),1)} ${fmt(deg(r[1]),1)} ${fmt(deg(r[2]),1)}</td></tr>`;
}
function renderJoints(j){
  if(!j){ $("joints").innerHTML=""; return; }
  let h="";
  for(const comp of ["left_arm","right_arm","torso","head"]){
    if(!j[comp]) continue;
    h+=`<div class='grp'><h3>${comp.replace('_',' ')}</h3><table>`;
    for(const [nm,val] of Object.entries(j[comp])) h+=`<tr><td class='k'>${nm}</td><td class='v'>${fmt(deg(val),1)}</td></tr>`;
    h+="</table></div>";
  }
  $("joints").innerHTML=h;
}
function renderSuction(on){ const e=$("suction"); e.className="suction "+(on?"on":"off"); e.textContent=on?"● ON":"○ off"; }
function renderGripper(pos){
  const e=$("gripper");
  if(pos==null){ e.className="suction off"; e.textContent="n/a"; return; }
  const closed = pos>=128;   // 0=open..255=closed; midpoint split for the label
  e.className="suction "+(closed?"on":"off");
  e.textContent=(closed?"● closed":"○ open")+" · pos "+pos;
}

// ---- charts ---------------------------------------------------------------
const CHARTS=[{canvas:"chart0",sel:"sig0",cur:"cur0",def:"force.tared_mag"},
              {canvas:"chart1",sel:"sig1",cur:"cur1",def:"ee.z"}];
function drawChart(ch){
  const c=$(ch.canvas); if(!c) return;
  const dpr=window.devicePixelRatio||1, W=c.clientWidth, H=c.clientHeight;
  c.width=W*dpr; c.height=H*dpr;
  const ctx=c.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
  const padL=52,padR=12,padT=10,padB=20, x0=padL,x1=W-padR,y0=padT,y1=H-padB;
  ctx.strokeStyle="#21262d"; ctx.strokeRect(x0,y0,x1-x0,y1-y0);
  const sig=$(ch.sel).value, meta=knownSig[sig]||{};
  const pts=[]; let vmin=Infinity,vmax=-Infinity;
  frames.forEach((s,i)=>{ const f=flatten(s)[sig]; if(f && isFinite(f.value)){ pts.push([i,f.value]); if(f.value<vmin)vmin=f.value; if(f.value>vmax)vmax=f.value; }});
  if(pts.length<2){ return; }
  if(vmin===vmax){ vmin-=1; vmax+=1; }
  const pad=(vmax-vmin)*0.08; vmin-=pad; vmax+=pad;
  const N=frames.length-1 || 1;
  const sx=i=>x0+i/N*(x1-x0), sy=v=>y1-(v-vmin)/(vmax-vmin)*(y1-y0);
  ctx.textAlign="right"; ctx.textBaseline="middle"; ctx.font="11px ui-monospace,monospace";
  for(let i=0;i<=4;i++){ const v=vmin+(vmax-vmin)*i/4, y=sy(v);
    ctx.strokeStyle="#161b22"; ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillStyle="#8b949e"; ctx.fillText(v.toFixed(2),x0-6,y); }
  ctx.strokeStyle="#58a6ff"; ctx.lineWidth=1.5; ctx.beginPath();
  pts.forEach(([i,v],k)=>{ const x=sx(i),y=sy(v); k?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
  // playhead
  const px=sx(cur);
  ctx.strokeStyle="#d29922"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(px,y0); ctx.lineTo(px,y1); ctx.stroke();
  const f=flatten(frames[cur])[sig];
  $(ch.cur).textContent = (f?f.value.toFixed(3):"--")+(meta.unit?" "+meta.unit:"");
}
const STRIP_PAD = 6;   // = half the range thumb (12px), so the strip lines up with the scrub travel
function drawSuctionStrip(){
  const c=$("suctionstrip"); if(!c) return;
  const dpr=window.devicePixelRatio||1, W=c.clientWidth, H=c.clientHeight;
  c.width=W*dpr; c.height=H*dpr;
  const ctx=c.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
  const N=frames.length, x0=STRIP_PAD, innerW=Math.max(1, W-2*STRIP_PAD);
  for(let i=0;i<N;i++){ ctx.fillStyle=frames[i].suction_cmd?"#3fb950":"#21262d";
    ctx.fillRect(x0+i/N*innerW,2,innerW/N+1,H-4); }
  const px=x0+cur/(N-1||1)*innerW;
  ctx.strokeStyle="#d29922"; ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H); ctx.stroke();
}
function drawCharts(){ for(const ch of CHARTS) drawChart(ch); drawSuctionStrip(); }

// ---- frame sync -----------------------------------------------------------
// Keep a sliding window of decoded Images warm so playback never waits on a
// per-frame fetch (the visible <img> then swaps from browser cache instantly).
const AHEAD = 24, BEHIND = 6;
const imgCache = new Map();   // "kind:idx" -> Image (held so it stays cached)
function url(kind,i){ return `/${kind}/${encodeURIComponent(NAME)}/${i}`; }
function prefetchOne(kind,i){
  const k = kind+":"+i;
  if(imgCache.has(k)) return;
  const im = new Image(); im.src = url(kind,i); imgCache.set(k, im);
}
function prefetchWindow(c){
  for(let j=c; j<Math.min(frames.length, c+AHEAD); j++){ prefetchOne("rgb",j); prefetchOne("depth",j); }
  for(const k of [...imgCache.keys()]){ const i=+k.split(":")[1]; if(i<c-BEHIND || i>c+AHEAD) imgCache.delete(k); }
}
function show(i){
  cur = Math.max(0, Math.min(frames.length-1, i|0));
  prefetchOne("rgb",cur); prefetchOne("depth",cur);   // current frame first
  $("scrub").value = cur;
  $("rgb").src = url("rgb",cur);
  $("depth").src = url("depth",cur);
  const s = frames[cur];
  renderForce(s.wrench); renderEE(s.ee,"ee"); renderEE(s.ee_right,"ee_r");
  renderJoints(s.joints); renderSuction(s.suction_cmd); renderGripper(s.gripper_pos);
  const rel = (s.t - t0);
  $("tinfo").textContent = `frame ${cur+1}/${frames.length} · t+${rel.toFixed(2)}s`;
  drawCharts();
  prefetchWindow(cur);
}
function stop(){ playing=false; $("play").textContent="▶"; $("play").classList.remove("on"); if(playTimer){ clearTimeout(playTimer); playTimer=null; } }
function step(){
  if(!playing) return;
  if(cur>=frames.length-1){ stop(); return; }
  const speed = parseFloat($("speed").value);
  const dt = Math.max(0.001, (frames[cur+1].t - frames[cur].t)) / speed;
  show(cur+1);
  playTimer = setTimeout(step, dt*1000);
}
function play(){ if(playing){ stop(); return; } if(cur>=frames.length-1) show(0); playing=true; $("play").textContent="⏸"; $("play").classList.add("on"); step(); }

// ---- curation -------------------------------------------------------------
function setLabel(l){
  $("goodbtn").classList.toggle("on", l==="success");
  $("badbtn").classList.toggle("on", l==="fail");
}
async function rate(action){
  const res = await fetch(`/api/take/${encodeURIComponent(NAME)}?action=${action}`,{method:"POST",cache:"no-store"});
  if(!res.ok){ alert("labeling failed ("+res.status+")"); return; }
  const d = await res.json().catch(()=>({}));
  setLabel(d.label||"success");
}
async function del(){
  if(!confirm(`Move ${NAME} to .trash/ ?`)) return;
  await fetch(`/api/take/${encodeURIComponent(NAME)}?action=delete`,{method:"POST",cache:"no-store"});
  location.href="/";
}

// ---- load -----------------------------------------------------------------
async function load(){
  const r = await fetch(`/api/take/${encodeURIComponent(NAME)}`,{cache:"no-store"});
  if(!r.ok){ $("name").textContent="not found"; return; }
  const d = await r.json();
  frames = d.frames || [];
  $("name").textContent = d.meta.name || NAME;
  $("instr").textContent = d.meta.instruction || "";
  setLabel(d.label||"success");
  $("depthcap").textContent = `depth ${_PREVIEW_LABEL} (turbo, near→far)`;
  if(!frames.length){ $("tinfo").textContent="no frames"; return; }
  t0 = frames[0].t;
  $("scrub").max = frames.length-1;
  buildSignals();
  fillSelect($("sig0"), CHARTS[0].def);
  fillSelect($("sig1"), CHARTS[1].def);
  show(0);
}
const _PREVIEW_LABEL = "0.3–1.0 m";

$("scrub").addEventListener("input", e=>{ stop(); show(parseInt(e.target.value,10)); });
$("first").addEventListener("click", ()=>{ stop(); show(0); });
$("last").addEventListener("click", ()=>{ stop(); show(frames.length-1); });
$("prev").addEventListener("click", ()=>{ stop(); show(cur-1); });
$("next").addEventListener("click", ()=>{ stop(); show(cur+1); });
$("play").addEventListener("click", play);
$("speed").addEventListener("change", ()=>{ if(playing){ stop(); play(); } });
for(const ch of CHARTS) $(ch.sel).addEventListener("change", ()=>drawChart(ch));
for(const ch of CHARTS){
  const c=$(ch.canvas);
  c.addEventListener("click", e=>{
    const rect=c.getBoundingClientRect(), padL=52,padR=12;
    const frac=(e.clientX-rect.left-padL)/(rect.width-padL-padR);
    stop(); show(Math.round(frac*(frames.length-1)));
  });
}
$("suctionstrip").addEventListener("click", e=>{
  const c=$("suctionstrip"), rect=c.getBoundingClientRect();
  const frac=(e.clientX-rect.left-STRIP_PAD)/Math.max(1, rect.width-2*STRIP_PAD);
  stop(); show(Math.round(Math.max(0,Math.min(1,frac))*(frames.length-1)));
});
$("goodbtn").addEventListener("click", ()=>rate("success"));
$("badbtn").addEventListener("click", ()=>rate("fail"));
$("delbtn").addEventListener("click", del);
document.addEventListener("keydown", e=>{
  if(e.target.tagName==="SELECT") return;
  if(e.key===" "){ e.preventDefault(); play(); }
  else if(e.key==="ArrowLeft"){ stop(); show(cur-1); }
  else if(e.key==="ArrowRight"){ stop(); show(cur+1); }
});
window.addEventListener("resize", drawCharts);
load();
</script></body></html>
"""


# ---- HTTP handler ----------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    root = "recordings"

    def _send(self, code, ctype, body, no_store=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"), no_store=True)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self._send(404, "text/plain", b"not found", no_store=True)
            return
        self._send(200, ctype, body, no_store=True)

    # -- parse  /thing/<name>/<idx>  with name validated against disk --------
    def _take_arg(self, prefix):
        """Return (take_dir, name, idx); (None, None, None) if unknown."""
        path = urlparse(self.path).path
        rest = path[len(prefix):]
        parts = [p for p in rest.split("/") if p != ""]
        if not parts:
            return None, None, None
        from urllib.parse import unquote
        name = unquote(parts[0])
        tdir = _find_take(self.root, name)
        if tdir is None:
            return None, None, None
        idx = None
        if len(parts) > 1:
            try:
                idx = int(parts[1].split(".")[0])
            except ValueError:
                idx = None
        return tdir, name, idx

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8",
                       GALLERY_HTML.replace("__CSS__", _CSS).encode("utf-8"))
        elif path.startswith("/take/"):
            self._send(200, "text/html; charset=utf-8",
                       PLAYER_HTML.replace("__CSS__", _CSS).encode("utf-8"))
        elif path == "/api/takes":
            ratings = _ratings(self.root)
            takes = _discover(self.root)
            self._send_json([
                _take_summary(os.path.join(self.root, task, name), name, task, ratings)
                for name, task in sorted(takes.items(), reverse=True)
            ])
        elif path.startswith("/api/take/"):
            tdir, name, _ = self._take_arg("/api/take/")
            if tdir is None:
                self._send_json({"error": "unknown take"}, 404)
                return
            meta = _read_json(os.path.join(tdir, "meta.json"))
            self._send_json({
                "meta": meta,
                "frames": _load_states(tdir),
                "label": _label(meta, _ratings(self.root), name),
            })
        elif path.startswith("/thumb/"):
            tdir, _, _ = self._take_arg("/thumb/")
            if tdir is None:
                self._send(404, "text/plain", b"not found", no_store=True)
                return
            self._send_file(os.path.join(tdir, "head_rgb", "000000.jpg"), "image/jpeg")
        elif path.startswith("/rgb/"):
            tdir, _, idx = self._take_arg("/rgb/")
            if tdir is None or idx is None:
                self._send(404, "text/plain", b"not found", no_store=True)
                return
            self._send_file(os.path.join(tdir, "head_rgb", f"{idx:06d}.jpg"), "image/jpeg")
        elif path.startswith("/depth/"):
            tdir, name, idx = self._take_arg("/depth/")
            if tdir is None or idx is None:
                self._send(404, "text/plain", b"not found", no_store=True)
                return
            body = _colorize_depth(tdir, name, idx)
            if body is None:
                self._send(404, "text/plain", b"no depth", no_store=True)
                return
            self._send(200, "image/jpeg", body, no_store=True)
        else:
            self._send(404, "text/plain", b"not found", no_store=True)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/take/"):
            self._send(404, "text/plain", b"not found", no_store=True)
            return
        tdir, name, _ = self._take_arg("/api/take/")
        action = (parse_qs(parsed.query).get("action", [""])[0]).strip()
        if tdir is None:
            self._send_json({"error": "unknown take"}, 404)
            return
        if action in ("success", "fail"):
            r = _ratings(self.root)
            r[name] = {"rating": action, "ts": time.time()}
            _write_ratings(self.root, r)
            self._send_json({"ok": True, "label": action})
        elif action == "delete":
            # Trash is flat regardless of task folder — names are unique.
            trash = os.path.join(self.root, TRASH)
            os.makedirs(trash, exist_ok=True)
            os.replace(tdir, os.path.join(trash, name))
            r = _ratings(self.root)
            r.pop(name, None)
            _write_ratings(self.root, r)
            self._send_json({"ok": True, "deleted": name})
        else:
            self._send_json({"error": "bad action"}, 400)

    do_HEAD = do_GET

    def log_message(self, *args):  # silence per-request logging
        pass


def main():
    parser = argparse.ArgumentParser(description="Recorded-take review dashboard")
    parser.add_argument("--root", default="recordings",
                        help="recordings directory written by recorder.py")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: all)")
    parser.add_argument("--port", type=int, default=8081, help="port (default: 8081)")
    args = parser.parse_args()

    _Handler.root = os.path.abspath(args.root)
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    shown = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"Review dashboard serving {_Handler.root}")
    print(f"  open  http://{shown}:{args.port}/   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
