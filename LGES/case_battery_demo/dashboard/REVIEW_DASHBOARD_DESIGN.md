# Review Dashboard — Design

A browser tool to **review recorded takes** in `LGES/recordings/` before they
become VLA training data: browse all takes, play each one back frame-by-frame
with synced telemetry, and curate (flag good/bad, delete bad ones).

This is the offline counterpart to the live [`server.py`](server.py). It is a
**separate process** (`review_server.py`, port **8081**) so neither tool
complicates the other. Same constraints as the live dashboard: **Python stdlib
only** for the HTTP layer, plus `cv2`/`numpy` (already deps) for depth
colorizing. Same dark GitHub-style CSS.

```
# point it at the recordings root and open a browser
python -m case_battery_demo.dashboard.review_server --root LGES/recordings
#   open http://<robot-ip>:8081/
```

---

## What a reviewer needs (the requirements)

1. **See every take at a glance** — which exist, how long, how many frames, any
   dropped frames, what the instruction was.
2. **Play one back** — scrub a timeline, play/pause, step single frames, change
   speed; RGB **and** depth move together with the telemetry.
3. **Read the telemetry at the scrubbed frame** — wrench (is the suction
   contact force sane?), EE pose, joints, suction on/off.
4. **See the whole-episode shape** — a force/height/suction curve with a
   playhead, to spot a bad grasp or an early release without scrubbing frame by
   frame.
5. **Curate** — flag a take good/bad and delete the junk, reversibly.

---

## On-disk source (already produced by `recorder.py`)

Per take `LGES/recordings/<YYYYmmdd-HHMMSS_epNNNN>/`:

| file | content |
|---|---|
| `meta.json` | `instruction`, `frames`, `duration_s`, `dropped_frames`, `arm_side`, `ee_frame`, `depth_preview_frame`, conventions |
| `head_rgb/NNNNNN.jpg` | per-frame RGB, q95 |
| `head_depth/NNNNNN.png` | per-frame depth, **uint16 mm, 0 = invalid** (renders black in a browser) |
| `depth_preview.jpg` | one pre-colorized sample frame (eyeballing only) |
| `states.jsonl` | one line/frame: `i`, `t`, `joints`, `wrench{fx..tz,raw_mag,tared_mag}`, `suction_cmd`, `ee{pos,quat_wxyz}`, `ee_right{...}` |

`states.jsonl` is ~100 lines/take (a few hundred KB). Small enough to parse
**server-side once per request** and ship as JSON arrays the page drives
everything off of — no need for a per-frame state endpoint.

Curation state is **not** in the recorder's output, so the review server owns a
small sidecar it writes itself (see *Curation*).

---

## Two views

### 1. Gallery — `GET /`

Grid of take cards, newest first. Each card:

- thumbnail (`/thumb/<name>` → first RGB frame)
- name + `created`
- instruction (the language label)
- `frames` · `duration_s` · a **⚠ dropped N** badge when `dropped_frames > 0`
- a status chip from the curation sidecar: **unrated · good · bad**

Filter row: *all / good / bad / unrated*. Click a card → player.

### 2. Player — `GET /take/<name>`

```
┌───────────────────────────── header: name · instruction · [Good] [Bad] [Delete] ──┐
├──────────────────────────┬──────────────────────────┬─────────────────────────────┤
│   RGB  (/rgb/.../<idx>)   │  DEPTH (/depth/.../<idx>) │  Wrench  |F| bar + fx..tz    │
│                           │   colorized, turbo        │  EE pose  L + R grippers     │
│                           │                           │  Joints   (deg)              │
│                           │                           │  Suction  ● ON / ○ off       │
├───────────────────────────────────────────────────────────────────────────────────┤
│  ◀◀ ◀ ▶/⏸ ▶ ▶▶   [============o==================]  frame 42/105 · t+3.4s · 1.0×    │  ← transport + scrub
├───────────────────────────────────────────────────────────────────────────────────┤
│  timeline charts:  |F| tared ───────╮ · ee.z · suction band   (click/drag = seek)   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

- **One frame index drives everything.** Scrub bar, transport buttons, chart
  playhead, and the side panels all read/write the same `frame` integer.
- **Playback**: `play` advances the index on a timer paced by the real
  inter-frame `t` deltas (so 15 Hz plays at true speed); speed selector
  0.25×–4×. Neighbor RGB/depth images are preloaded so scrubbing is smooth.
- **Side panels** reuse the live dashboard's renderers (wrench bar, EE table,
  joints) — copied, fed from the scrubbed frame instead of a poll. `ee` is
  `quat_wxyz` here (vs `rpy` live), converted to RPY for display.
- **Timeline charts** plot the full episode (precomputed arrays) with a vertical
  playhead at the current frame; click/drag on a chart seeks. Default signals:
  `|F| tared`, `ee.z`, and a `suction` on/off band — the three that reveal a bad
  grasp fastest. Signal picker reuses the live `flatten()` signal list.

---

## Endpoints

| route | returns |
|---|---|
| `GET /` | gallery HTML |
| `GET /take/<name>` | player HTML |
| `GET /api/takes` | `[{name, created, instruction, frames, duration_s, dropped_frames, rating}]` |
| `GET /api/take/<name>` | `{meta, frames:[…parsed states.jsonl…], rating}` |
| `GET /thumb/<name>` | first RGB frame (jpeg) |
| `GET /rgb/<name>/<idx>` | that RGB frame (jpeg, served from disk) |
| `GET /depth/<name>/<idx>` | **colorized** depth (jpeg, see below) |
| `POST /api/take/<name>?action=good\|bad\|unrate` | sets rating in sidecar |
| `POST /api/take/<name>?action=delete` | moves take to `<root>/.trash/` |

All paths validated against the known on-disk take names (no traversal); `<idx>`
clamped to `[0, frames)`.

---

## Depth colorizing (server-side, per the decision)

Raw `head_depth/*.png` are uint16 mm and look black in a browser. `/depth/...`
reads the PNG and applies **the exact same mapping the recorder uses for
`depth_preview.jpg`** so previews and scrubbed depth match:

- turbo colormap (`cv2.COLORMAP_TURBO`, JET fallback)
- range **0.3–1.0 m**, `0 mm` (invalid) → black

```
mm  = cv2.imread(png, IMREAD_UNCHANGED).astype(float32) / 1000   # → metres
norm = clip((mm - 0.3) / 0.7, 0, 1);  norm[mm<=0] = 0
jpg  = applyColorMap((norm*255).uint8, TURBO);  jpg[mm<=0] = black
```

These constants (`_DEPTH_CMAP`, `_PREVIEW_DEPTH_RANGE_M`) already live in
`recorder.py`; the review server imports/mirrors them so there's one source of
truth. Colorized JPEGs are **cached** to `/tmp/cns_review_cache/<name>/<idx>.jpg`
(keyed by name+idx) so a re-scrub doesn't re-decode.

---

## Curation (per the decision: flag + delete, reversible)

The recorder doesn't track review state, so the review server keeps a sidecar at
the recordings root:

```
LGES/recordings/.review.json     {"<take-name>": {"rating": "good|bad", "ts": …}, …}
```

- **Good / Bad** buttons write `rating` for the take; **Unrate** clears it.
- **Delete** is reversible: move the take dir to `LGES/recordings/.trash/<name>`
  (an `os.replace`, instant, recoverable by moving it back) rather than
  `rmtree`. Gallery hides `.trash` and `.pending`.
- The sidecar is the only thing the review tool writes into `recordings/`; it
  never mutates take contents.

---

## Why these choices

- **Separate file/process, not extending `server.py`.** The live page polls a
  single mutating spool; the review page is random-access over many immutable
  takes with a scrub timeline. Forcing both into one page/handler would tangle
  two unrelated state models. They still share the CSS and the panel renderers.
- **Client-side frame state, server-side parse.** Takes are ~100 frames, so
  shipping the whole parsed `states.jsonl` once lets the timeline + scrub be
  instant and offline, while keeping the server a thin file/JSON server like the
  live one.
- **Reuse the recorder's depth mapping.** Anything else means the scrubbed depth
  disagrees with the saved preview — confusing during review.

---

## Build plan (after this doc is approved)

1. `review_server.py` skeleton: arg parsing, `ThreadingHTTPServer`, take
   discovery + name validation, `/api/takes`, gallery HTML. → *verify: gallery
   lists the existing takes with correct counts.*
2. `/rgb`, `/thumb`, `/depth` (colorize + cache), `/api/take/<name>`. → *verify:
   curl each for an existing take returns the right bytes; depth is colored.*
3. Player HTML/JS: scrub + transport + image swap + synced panels. → *verify:
   scrub moves RGB/depth/wrench together; play runs at ~real speed.*
4. Timeline charts with playhead + click-to-seek. → *verify: playhead tracks
   frame; clicking seeks.*
5. Curation: sidecar read/write, good/bad/delete (`.trash`), gallery filter. →
   *verify: rating persists across reload; delete hides the take and leaves it
   recoverable in `.trash`.*

Est. one new file, ~600–700 lines (comparable to `server.py`'s 672), no new
dependencies.
