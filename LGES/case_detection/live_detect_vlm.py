"""Capture-and-detect viewer like live_detect_bev.py, with Qwen3.5-35B-A3B as
the detector.

Same job — a base_link pose (x, y, yaw, size) for each object in view — but
the detector is the local VLM served by vLLM instead of the trained
YOLO-OBB weights, and the task is open: EVERY object standing on the white
table in front of the robot, whatever it is, each with the noun the model gives
it. No class list, no EE-target marker (nothing to look an ik_demo offset up
for). Each label gets its own colour (hashed, so stable across captures), and
labels naming the robot's own arm or gripper are dropped: the head camera sees
the arm at the image edge whenever it is over the table.

It is NOT a live detector. The camera is streamed live (raw frame + BEV of the
table plane) so you can see what you are about to shoot; pressing c (or Space)
CAPTURES that one frame, sends it to the VLM and shows the result — quads on
the captured frame, OBBs on its BEV, numbers in the legend — until the next
capture, or l to go back to the live view. The answer is STREAMED and drawn
as it comes: the first object appears ~1.0 s after the capture and each
further one ~0.7 s later (FP8 server), so a 3-object scene is complete at
~2.5 s but readable from 1 s. At that speed a "live" overlay would only ever
show boxes for a picture seconds old, hence capture-on-key.

Except in TRACK mode (t): the VLM answers ONCE, and from then on the objects
are followed from where they were (Tracker): SAM2 re-finds each mask 5 times
a second (hiera_large ~135 ms a pass, --sam2 tiny ~65 ms with the same
tracking quality on the recorded test), and between those passes a CPU
template match slides each box along, so the display runs at TRACK_FPS —
with the VLM re-checking the scene in the background for objects put down or
taken away — only when there is a reason: a track lost, or a blob on the
table that no track accounts for (Tracker._unexplained), with --redetect
(default 30 s) as a safety net. The positions are live; only the vocabulary
is seconds old. The two share one GPU, so SAM2 pauses while the VLM answers
(the boxes keep following on the CPU), and the VLM is sent only the table
ROI at 640 px (_table_roi), about half the prefill of the full frame.

What differs from live_detect_bev, and why:

  * The VLM sees the RAW left camera frame, not the BEV canvas. A VLM was
    trained on photographs; a metric top-down warp of a transparent case is
    not one. So detection has no warp step — the model answers in image
    pixels and the geometry is done on the plane afterwards. The BEV is only
    DISPLAYED, beside the raw frame: the same frame warped at the same plane,
    with the fitted OBB drawn from its base-frame corners, so the yaw and
    size the legend reports can be checked against a top-down picture.
  * A VLM grounds with an axis-aligned bbox, which has no yaw — and asking it
    for anything else is unstable (numbers in VlmDetector). So the VLM only
    says WHAT and roughly WHERE (label + bbox_2d, its native task), and SAM2
    (Segmenter, hiera_large from checkpoints/, ~25 ms per box on Thor) says
    exactly where: the mask inside the box. The mask OUTLINE is cast from
    the image onto the table plane (pixel -> plane is the inverse of the BEV
    homography, bev.build_mapper) and cv2.minAreaRect fitted to it in
    base_link — center, metric size and yaw with the perspective undone.
    The mask is the object's whole silhouette, not just its top face, so a
    tall object (a bin) reads a few cm larger on the near side than its rim;
    flat things on a table are unaffected.
  * Objects whose OBB center falls outside the table ROI (cfg.BEV_X/Y_RANGE)
    are dropped: the bbox prompt also lists a gripper, tools on a shelf, a
    bin cut off at the image edge, and all of those cast off the table.
  * The answer is forced through vLLM STRUCTURED OUTPUT (response_format
    json_schema), so it is always the exact JSON the parser expects.
  * No confidence. The model does not have one, and a self-reported number
    would be invented. Detections are listed in the order the model gave.
  * Latency is output-token bound: ~40 tokens per object at ~49 tok/s on the
    FP8 server (prefill of the 960x600 frame is ~0.45 s of it). Streaming
    does not shorten that, it only moves the first box forward; see the
    VlmDetector docstring for the parallel-per-object idea that did NOT pay.

The plane is the WHITE TABLE at a FIXED height, TABLE_Z_M (0.715 m base z,
the depth-mode measurement of 2026-09-06; --plane overrides). No depth is
used anywhere: the table does not move, and the per-object depth refinement
that live_detect_bev does was dropped on purpose — it moved the reported
center off the OBB fitted on the table plane (the cross landed off the box
in the BEV), and one plane means one center: the minAreaRect center IS the
point reported and drawn. The cost is a known bias for objects with height —
a top face h above the table casts onto the table plane magnified about the
camera nadir by (C_z - z_table) / (C_z - z_table - h), roughly +8 % in
position-from-nadir and size for h = 10 cm with the camera 1.2 m up — small
for flat things on a table, and a systematic one a caller can undo if it
knows the object's height.

Coordinates are 0-1000 normalized on both axes, the Qwen3-VL/Qwen3.5
grounding convention — confirmed on this checkpoint (y values up to 1000 on
the 600-px-tall frame). Corner accuracy on a clearly ROTATED object is still
to be checked on the robot; the saved frames only had near-axis-aligned ones.

Needs the vLLM server up (Thor container, see vlm_ad/README.md). The FP8
checkpoint (Qwen/Qwen3.5-35B-A3B-FP8 from ModelScope, 35 GB) decodes ~1.8x
faster than BF16 here — 43 tok/s vs ~25 measured with this very prompt —
because decode is weight-bandwidth bound and the weights are half the size:

    sudo docker run -it --rm --runtime=nvidia --network host \\
        --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \\
        -e VLLM_TEST_FORCE_FP8_MARLIN=1 \\
        -v /home/dexmate/nvidia/models:/models \\
        ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \\
        vllm serve /models/Qwen3.5-35B-A3B-FP8 --served-model-name qwen3.5 \\
            --max-model-len 8192 --gpu-memory-utilization 0.4 \\
            --limit-mm-per-prompt '{"image":1,"video":0}' --enable-prefix-caching

VLLM_TEST_FORCE_FP8_MARLIN=1 is REQUIRED on Thor for the FP8 checkpoint. Thor
is compute capability 11.0; vLLM 0.19's "is this Blackwell" test for its
CUTLASS block-FP8 GEMM accepts 100 <= cc < 120, but the kernel is built for
the sm100 family only, so the first forward printed "This kernel only
supports sm100f." from the device and the engine died with a cuBLAS error
(2026-09-06). The flag routes every FP8 linear and the MoE experts through
Marlin (plain CUDA, sm75+), which is what we want anyway: Marlin is a
weight-only FP8 kernel, i.e. exactly the halved weight traffic. Output on the
saved frames was corner-for-corner the same as BF16.

--gpu-memory-utilization 0.4 is plenty (35 GB weights + KV for 300k tokens);
0.7 was for the 67 GB BF16 weights. For BF16, drop the env flag and use
/models/Qwen3.5-35B-A3B with 0.7.

Optional, +10 % measured (43 -> 47.5 tok/s on the same frames): the
checkpoint's MTP head as speculative decoder,
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
The drafts are good (mean acceptance length 2.45 of 3) but the grammar-
constrained verify step eats most of the gain, and it costs ~1 min more
startup and a second model load. Not worth much here; harmless to add.

Robot-side (head camera, RGB only). Keys: c/Space capture + detect (in track
mode: ask the VLM again now), t track mode on/off, l back to live, s save the
shown picture, q/Esc quit. The installed cv2 is headless, so on this robot use
--serve:

    python live_detect_vlm.py --serve 8088
    python live_detect_vlm.py --plane 0.699 --serve 8088
    python live_detect_vlm.py --serve 8088 --redetect 0    # track: VLM only on t and c

Without the robot, on a sequence recorded with record_track.py (the tracker
is judged on replays of the same motions; the recording loops):

    python live_detect_vlm.py --replay data/track_seq/<timestamp> --serve 8088
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import re
import sys
import threading
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import live_detect_bev as ld          # plane seeding, text plates, HTTP viewer
# ld first: it puts ../perception on sys.path, which bev/depth_plane need.
import config as cfg
import bev
import depth_plane as dp                # base_to_pixel only; no depth is read
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import set_head_pitch

_COLOR = (0, 255, 0)                    # header/footer text only

# base_link z of the white table top. Depth mode inside the BEV ROI on the
# 2026-09-06 14:28 floor_measure frame gave 0.7146; fixed here by decision.
TABLE_Z_M = 0.715

LIVE_FPS = 10.0                         # preview publish rate (see main)
TRACK_FPS = 15.0                        # track-mode step + publish rate (see main)

# One colour per LABEL, chosen by hashing the label: the same noun gets the
# same colour in every capture and every run, with no table to maintain for
# an open vocabulary. 12 hues that stay apart from each other and from the
# grid's dark green (BGR).
_PALETTE = [(0, 255, 0), (255, 160, 0), (0, 165, 255), (255, 0, 255),
            (0, 255, 255), (255, 255, 0), (80, 80, 255), (0, 200, 120),
            (200, 120, 255), (255, 220, 120), (120, 200, 255), (255, 120, 180)]


def _color(label: str) -> tuple[int, int, int]:
    return _PALETTE[zlib.crc32(label.encode()) % len(_PALETTE)]


# Labels that are the robot looking at itself, dropped even if the model
# names them: the head camera sees its own arm at the image edge whenever the
# arm is over the table, and "ignore the robot" in the prompt is not enough.
_IGNORE = ("robot", "arm", "gripper", "manipulator", "hand", "finger", "wrist",
           "sleeve", "person", "shadow", "reflection", "glare", "table")

_SYSTEM = (
    "You are a precise visual grounding model for a robot camera. "
    "Answer with JSON only: no prose, no markdown fences.")

_USER = """The image is from a robot's head camera looking down at a WHITE TABLE in
front of it. Find every object standing on that table. Ignore the table itself,
the floor, walls, people and anything not on the table. The ROBOT'S OWN ARMS
and grippers (the mechanical manipulators reaching in from the image edges) are
part of the robot, not objects on the table: never list them.

For each object, give its 2D bounding box [x1,y1,x2,y2], normalized to 0-1000
(0,0 = top-left of the image, 1000,1000 = bottom-right).

Output compact single-line JSON, one entry per object: "l" is ONE lowercase
word naming it, "b" is its bounding box:
{"o":[{"l":"cup","b":[x1,y1,x2,y2]}, ...]}
Use {"o":[]} if there is none."""

# Grammar for the answer (vLLM structured output). Every output token is
# ~20 ms of latency, so: one-letter keys and a one-word label. ~25 tokens
# per object.
_SCHEMA = {
    "type": "object",
    "properties": {"o": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "l": {"type": "string"},
            "b": {"type": "array", "minItems": 4, "maxItems": 4,
                  "items": {"type": "integer", "minimum": 0, "maximum": 1000}}},
        "required": ["l", "b"], "additionalProperties": False}}},
    "required": ["o"], "additionalProperties": False}

# One complete object entry of that answer, as it appears in the stream. The
# grammar fixes the shape, so this only has to tolerate whitespace.
_OBJ_RE = re.compile(
    r'\{\s*"l"\s*:\s*"([^"]*)"\s*,\s*"b"\s*:\s*\[\s*'
    r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*\}')


class VlmDetector:
    """Label + 2D bounding box of everything on the table, from the vLLM
    OpenAI-compatible endpoint, in ONE request whose answer is STREAMED: each
    object's JSON entry is complete ~25 tokens (~0.5 s) after the previous
    one, and ``progress`` is called with the objects parsed so far as each
    closes, so the caller can draw the first object at ~1 s instead of
    everything at the end. Thinking is off: the answer is a short
    grammar-bound JSON and every reasoning token is another ~20 ms.

    bbox_2d is the grounding format the model was TRAINED on and it is
    stable: across six small perturbations of one frame (2-3 px shifts, JPEG
    quality, +5 % gain) the boxes of the three main objects moved by at most
    6-7 units of 1000 (~6 px). Asking for the four corners of the top face
    instead (what this script did first) is out of distribution: the same
    bin came back as a perspective trapezoid on one variant and an axis-
    aligned box on the next, with single corners jumping ~100 units, and the
    object order changed too. Geometry is therefore left to Segmenter.

    Why not one request per object in parallel (tried 2026-09-06, 3 frames):
    it was SLOWER — a LIST pass (1.4-1.9 s) plus a batch of 4-6 CORNERS
    requests that took 2.1-4 s, against 2.9 s for the single request. This
    model routes every token to 8 of 256 experts, so a decode step for N
    concurrent sequences reads up to 8N experts' weights: batching is not free
    on a fine-grained MoE the way it is on a dense model, and the whole
    premise fell over."""

    def __init__(self, base_url: str, model: str, timeout_s: float,
                 max_tokens: int) -> None:
        self._client = OpenAI(base_url=base_url, api_key="EMPTY",
                              timeout=timeout_s, max_retries=0)
        self._model, self._max_tokens = model, max_tokens
        self.last_raw = ""

    def check(self) -> None:
        """Fail at startup, not on the first frame, if the server is down or
        serves a different model id (--served-model-name mismatch -> 404)."""
        ids = [m.id for m in self._client.models.list().data]
        if self._model not in ids:
            raise SystemExit(f"vLLM at {self._client.base_url} serves {ids}, "
                             f"not '{self._model}' (--vlm-model)")

    def detect(self, bgr: np.ndarray, progress=None, roi=None) -> list[dict]:
        """[{label, bbox_px (x1,y1,x2,y2)}] in answer order, dropping labels
        in _IGNORE. ``progress(objs)`` is called with the objects parsed so far
        each time one more entry closes in the stream. With ``roi`` (x1,y1,
        x2,y2, see _table_roi) only that crop is sent, resized to a long side
        of 640 px: prefill is per image token (~0.45 s for the full 960x600
        frame), so the crop roughly halves it, and a small object on the
        table is larger in the image the model grounds on. Boxes come back in
        FULL-frame pixels."""
        x1, y1 = (0, 0) if roi is None else (int(roi[0]), int(roi[1]))
        crop = bgr if roi is None else bgr[y1:int(roi[3]) + 1, x1:int(roi[2]) + 1]
        s = min(1.0, 640.0 / max(crop.shape[:2]))
        if s < 1.0:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return []
        b64 = base64.b64encode(buf).decode()
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": [
                          {"type": "text", "text": _USER},
                          {"type": "image_url", "image_url": {
                              "url": f"data:image/jpeg;base64,{b64}"}}]}],
            max_tokens=self._max_tokens, temperature=0.0, stream=True,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "boxes", "schema": _SCHEMA}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        text, objs, n_seen = "", [], 0
        for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            text += chunk.choices[0].delta.content
            found = _OBJ_RE.findall(text)
            if len(found) > n_seen:
                for m in found[n_seen:]:
                    o = self._object(m, crop.shape[1], crop.shape[0])
                    if o is not None:
                        o["bbox_px"] = o["bbox_px"] / s + [x1, y1, x1, y1]
                        objs.append(o)
                n_seen = len(found)
                if progress:
                    progress(objs)
        self.last_raw = text
        return objs

    @staticmethod
    def _object(m, w: int, h: int) -> dict | None:
        """One _OBJ_RE match -> {label, bbox_px, ignore}, or None for an empty
        box. ``ignore`` marks a label in _IGNORE (the robot, a hand): not an
        object, but its box is kept — the Tracker will not seed an object
        inside it (something held in a hand is passing through)."""
        label = m[0].strip().lower().replace(" ", "_")
        b = np.asarray(m[1:], dtype=np.float64) / 1000.0 * [w, h, w, h]
        b = np.array([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])])
        b = np.clip(b, 0, [w - 1, h - 1, w - 1, h - 1])
        if b[2] - b[0] < 4 or b[3] - b[1] < 4:
            return None
        return dict(label=label, bbox_px=b, ignore=any(word in label for word in _IGNORE))


class Segmenter:
    """SAM2 image predictor: the mask inside a VLM box. The geometry — outline,
    center, size, yaw — comes from this mask, not from the VLM (see
    VlmDetector). hiera_large from cfg.SAM2_CHECKPOINT; on Thor the image
    encode is ~60 ms after warm-up and each box prompt ~25 ms."""

    # --sam2 size -> (hydra config in the sam2 package, checkpoint in checkpoints/).
    # large is cfg.SAM2_*; the others are the matching sam2.1 releases
    # (https://dl.fbaipublicfiles.com/segment_anything_2/092824/<file>).
    SIZES = {
        "large": (cfg.SAM2_MODEL_CFG, cfg.SAM2_CHECKPOINT),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    }

    def __init__(self, size: str = "large") -> None:
        import torch  # noqa: PLC0415 (heavy deps, loaded once here)
        from sam2.build_sam import build_sam2  # noqa: PLC0415
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415
        self._torch = torch
        model_cfg, ckpt = self.SIZES[size]
        ckpt = _HERE / ckpt
        if not ckpt.exists():
            raise SystemExit(f"SAM2 checkpoint not found at {ckpt}")
        self._pred = SAM2ImagePredictor(build_sam2(model_cfg, str(ckpt), device="cuda"))
        self.last_logits = None                 # low-res logits of the last outline()
        # First encode pays for CUDA init/autotune (~0.4 s); take it here.
        self.set_image(np.zeros((cfg.IMG_H, cfg.IMG_W, 3), np.uint8))

    def _ctx(self):
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def set_image(self, rgb: np.ndarray) -> None:
        with self._torch.inference_mode(), self._ctx():
            self._pred.set_image(rgb)

    def outline(self, bbox_px: np.ndarray, prior=None) -> list[np.ndarray] | None:
        """Outer contours (each Nx2 pixels) of the object in the box, or None
        if SAM2 finds nothing. Two choices matter, both from the taped paper
        box of 2026-09-06: the prompt is the box PLUS its center point (the
        box alone left the tape stripe out of the mask, 48k vs 63k px), and
        of SAM2's three candidate masks the one whose own bounding box
        overlaps the prompt box most is taken, not the highest-scored one,
        because the VLM's box says how big the whole thing is. Every
        component at least 5 % of the largest is returned: that mask came in
        two pieces, one per side of the tape, and the largest piece alone
        measured the box at 0.57 x 0.18 m instead of 0.57 x 0.53.

        ``prior`` (Tracker): the low-res mask logits this returned for the
        same object on the previous frame (``self.last_logits`` after a
        call). The prompt is then the box plus that mask, single output — a
        mask MEMORY. Over 12 recorded frames of the black tray and a
        cylinder it held both at a constant footprint (~0.5x0.4 m, ~0.1 m)
        where box+point let the tray grow to 1.0x0.6 and the cylinder to
        0.4 m by taking in the tray around it (prompt_cmp, 2026-09-07)."""
        box = np.asarray(bbox_px, np.float64)
        with self._torch.inference_mode(), self._ctx():
            if prior is None:
                ctr = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], np.float32)
                masks, _, lows = self._pred.predict(
                    box=box.astype(np.float32)[None], point_coords=ctr,
                    point_labels=np.ones(1, int), multimask_output=True)
            else:
                masks, _, lows = self._pred.predict(
                    box=box.astype(np.float32)[None], mask_input=prior[None],
                    multimask_output=False)
        best, best_iou = None, -1.0
        for k, mk in enumerate(masks.astype(bool)):
            ys, xs = np.nonzero(mk)
            if xs.size == 0:
                continue
            mb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)
            inter = max(0.0, min(mb[2], box[2]) - max(mb[0], box[0])) * \
                max(0.0, min(mb[3], box[3]) - max(mb[1], box[1]))
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
            iou = inter / max(area(mb) + area(box) - inter, 1e-9)
            if iou > best_iou:
                best, best_iou = k, iou
        if best is None:
            return None
        self.last_logits = lows[best]
        return self._contours(masks[best])

    @staticmethod
    def _contours(mask) -> list[np.ndarray] | None:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        areas = [cv2.contourArea(c) for c in cnts]
        keep = [c.reshape(-1, 2).astype(np.float64)
                for c, a in zip(cnts, areas) if a >= 0.05 * max(areas)]
        return keep or None

    def outline_many(self, boxes, priors) -> list[tuple[list[np.ndarray] | None, np.ndarray]]:
        """outline(box, prior) for several objects in ONE decoder pass:
        (contours or None, low-res logits) per input. The prompt encoder and
        mask decoder take a batch, so five tracks cost 11 ms instead of
        5 x 9 (Thor, hiera_large; the per-call overhead dominates). Uses the
        predictor's _predict with the tensors it would build itself, because
        the public predict() only takes one mask_input."""
        if not boxes:
            return []
        torch, pred = self._torch, self._pred
        with torch.inference_mode(), self._ctx():
            bx = torch.as_tensor(np.asarray(boxes, np.float32), device=pred.device)
            bx = pred._transforms.transform_boxes(bx, normalize=True, orig_hw=pred._orig_hw[-1])
            mi = torch.as_tensor(np.stack(priors)[:, None], device=pred.device)
            masks, _, lows = pred._predict(None, None, bx, mi, multimask_output=False)
            masks = masks[:, 0].cpu().numpy()
            lows = lows[:, 0].float().cpu().numpy()
        return [(self._contours(mk), lo) for mk, lo in zip(masks, lows)]


def _px_to_plane(mapper: bev.BevMapper, quad_px: np.ndarray) -> np.ndarray:
    """Image pixels -> base (X, Y) on the mapper's plane, via the same
    homography the BEV warp uses (image -> canvas -> base, both exact)."""
    p = mapper.img_to_bev @ np.column_stack([quad_px, np.ones(len(quad_px))]).T
    p = (p[:2] / p[2]).T
    return np.array([mapper.bev_px_to_base(u, v) for u, v in p])


def _on_table(X: float, Y: float) -> bool:
    """Inside the table ROI (the BEV canvas extent, cfg.BEV_X/Y_RANGE). The
    bbox prompt lists more than the table — a gripper, tools on a shelf, a
    bin cut off at the image edge — and those land outside once cast onto the
    table plane, so the OBB center is the filter."""
    return (cfg.BEV_X_RANGE[0] <= X <= cfg.BEV_X_RANGE[1]
            and cfg.BEV_Y_RANGE[0] <= Y <= cfg.BEV_Y_RANGE[1])


def _table_roi(q_torso, q_head, plane_z: float, shape, margin: float = 0.05) -> np.ndarray:
    """Image box (x1,y1,x2,y2) around the table ROI: cfg.BEV_X/Y_RANGE on the
    table plane projected with the current joints, grown by ``margin`` of
    its extent, clipped to the image. What the VLM is sent (VlmDetector.detect)."""
    H, W = shape[:2]
    pts = [dp.base_to_pixel((X, Y, plane_z), q_torso, q_head)
           for X in cfg.BEV_X_RANGE for Y in cfg.BEV_Y_RANGE]
    u, v = [p[0] for p in pts], [p[1] for p in pts]
    mu, mv = (max(u) - min(u)) * margin, (max(v) - min(v)) * margin
    return np.array([max(min(u) - mu, 0), max(min(v) - mv, 0),
                     min(max(u) + mu, W - 1), min(max(v) + mv, H - 1)]).astype(int)


def _to_det(o: dict, seg: Segmenter, mapper: bev.BevMapper) -> dict | None:
    """One VLM object -> detection: SAM2 mask in its box, outline cast onto
    the mapper's plane, OBB fitted in base_link. The OBB's center is THE
    center: no depth refinement (see the module docstring). None if the mask
    is empty, degenerate, or centered off the table."""
    cnts = seg.outline(o["bbox_px"], o.get("prior"))
    return _det_from(o["label"], o["bbox_px"], cnts, seg.last_logits, mapper)


def _det_from(label: str, bbox_px, cnts, logits, mapper: bev.BevMapper) -> dict | None:
    """The detection dict for a mask given as its contours (see _to_det)."""
    if cnts is None:
        return None
    cnt = np.vstack(cnts)                            # all pieces, one footprint
    if len(cnt) < 3:
        return None
    P = _px_to_plane(mapper, cnt)
    rect = cv2.minAreaRect(P.astype(np.float32))
    (X, Y), (w, h), ang = rect
    if min(w, h) < 0.005 or not _on_table(X, Y):
        return None
    # minAreaRect's angle is the w side's; yaw is the LONG side's, in the
    # base frame already (P is base XY), [0, 180) like bev_yaw_to_base.
    yaw = float((ang + (90.0 if w < h else 0.0)) % 180.0)
    return dict(
        cls=label, bbox_px=bbox_px, contours_px=cnts,
        logits=logits,                              # Tracker's prior for the next frame
        base_xy=(float(X), float(Y)),
        rect_base=cv2.boxPoints(rect),              # fitted OBB corners, base XY
        yaw=yaw, dims_m=(float(max(w, h)), float(min(w, h))),
        color=_color(label),
    )


def detect(rgb, q_torso, q_head, vlm: VlmDetector, seg: Segmenter,
           plane_z: float, progress=None) -> list[dict]:
    """Detections on ``plane_z`` for ``rgb``: the VLM's boxes, each segmented
    and measured as soon as it closes in the streamed answer. ``progress(dets)``
    gets the detections so far on each. SAM2 encodes the image once, before
    the VLM answers, so per object only the ~25 ms box prompt remains."""
    mapper = bev.build_mapper(q_torso, q_head, plane_z)
    seg.set_image(rgb)
    dets: list[dict] = []
    done = 0

    def on_objs(objs):
        nonlocal done
        for o in objs[done:]:
            d = None if o["ignore"] else _to_det(o, seg, mapper)
            if d is not None:
                dets.append(d)
        done = len(objs)
        if progress:
            progress(dets)

    objs = vlm.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), progress=on_objs,
                      roi=_table_roi(q_torso, q_head, plane_z, rgb.shape))
    on_objs(objs)                                   # anything the stream ended on
    return dets


def _iou(a, b) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(_area(a) + _area(b) - inter, 1e-9)


def _area(b) -> float:
    return float((b[2] - b[0]) * (b[3] - b[1]))


def _mask_box(cnts) -> np.ndarray:
    """Bounding box (x1,y1,x2,y2) of a set of contours."""
    P = np.vstack(cnts)
    return np.array([P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()], np.float64)


def _grow(box, f: float, w: int, h: int) -> np.ndarray:
    """``box`` enlarged by the fraction ``f`` about its center, clipped to the image."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * (1 + f) / 2, (box[3] - box[1]) * (1 + f) / 2
    return np.clip([cx - bw, cy - bh, cx + bw, cy + bh], 0, [w - 1, h - 1, w - 1, h - 1])


def _match(win: np.ndarray, tmpl: np.ndarray, exclude=(), origin=(0, 0)
           ) -> tuple[tuple[int, int], float]:
    """Best placement (top-left, in ``win``'s frame coordinates given its
    ``origin``) of ``tmpl`` in ``win`` by normalized cross-correlation, and
    its score. Placements centered inside a box of ``exclude`` are skipped.
    A placement on a FLAT patch is rejected (score 0): NCC divides by the
    patch's own contrast, so sensor noise on bare white table correlated with
    a cylinder template at 0.6+ and the tracker 'found' cylinders on empty
    table (2026-09-07 recording). The patch must have at least half the
    template's contrast; the three best placements are tried."""
    th, tw = tmpl.shape[:2]
    if win.shape[0] < th or win.shape[1] < tw:
        return (0, 0), 0.0
    R = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
    for b in exclude:
        v1, v2 = int(b[1] - th / 2 - origin[1]), int(b[3] - th / 2 - origin[1]) + 1
        u1, u2 = int(b[0] - tw / 2 - origin[0]), int(b[2] - tw / 2 - origin[0]) + 1
        R[max(v1, 0):max(v2, 0), max(u1, 0):max(u2, 0)] = -1.0
    need = 0.5 * float(tmpl.std())
    for _ in range(3):
        _, mx, _, (u, v) = cv2.minMaxLoc(R)
        if mx <= 0:
            break
        if float(win[v:v + th, u:u + tw].std()) >= need:
            return (u + origin[0], v + origin[1]), float(mx)
        R[max(v - th // 2, 0):v + th // 2 + 1, max(u - tw // 2, 0):u + tw // 2 + 1] = -1.0
    return (0, 0), 0.0


def _stands_out(rgb: np.ndarray, box, min_diff: float = 20.0) -> bool:
    """Is there something in ``box`` at all: its mean colour differs from the
    ring around it (box grown 60 %) by at least ``min_diff`` (RGB L2). SAM2
    prompted with a box on bare table returns a box-shaped mask that passes
    every shape test (class docstring), and a stale VLM box seeded such
    ghosts where a moved object HAD been; the table is the same colour
    inside and out, an object is not. (A white object on the white table
    would fail this too — nothing here would find it reliably anyway.)"""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2 + 1, W), min(y2 + 1, H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    gx, gy = int((x2 - x1) * 0.3) + 2, int((y2 - y1) * 0.3) + 2
    X1, Y1, X2, Y2 = max(x1 - gx, 0), max(y1 - gy, 0), min(x2 + gx, W), min(y2 + gy, H)
    inner = rgb[y1:y2, x1:x2].reshape(-1, 3).astype(np.float64)
    outer = rgb[Y1:Y2, X1:X2].reshape(-1, 3).astype(np.float64)
    n_ring = len(outer) - len(inner)
    if n_ring < 16:
        return True                                  # at the image edge: no verdict
    ring = (outer.sum(0) - inner.sum(0)) / n_ring
    return float(np.linalg.norm(inner.mean(0) - ring)) >= min_diff


def _has_edges(rgb: np.ndarray, cnts, min_median: float = 30.0) -> bool:
    """Does the mask outline run along real edges: median Sobel magnitude of
    the grey image at the contour pixels >= ``min_median``. On the
    2026-09-07 recording real objects scored 180-490 (cylinders, tray); the
    ghost SAM2 drew on bare table beside a hand's shadow — which
    _stands_out let through, the shadow being darker than the ring — scored
    4, bare table 0-6, a hand 0, a shadow edge 35. Computed on the crop
    around the contours only (~0.3 ms). The median is over the outline's
    PIXELS (the contours rasterized), not its vertices: CHAIN_APPROX_SIMPLE
    puts one vertex on a long straight edge and dozens where the mask
    meanders over a smooth shadow, and by vertices the tray with a hand
    over its rim scored 6 while its rim itself is a 480 edge."""
    P = np.vstack(cnts)
    x1, y1 = int(max(P[:, 0].min() - 2, 0)), int(max(P[:, 1].min() - 2, 0))
    x2, y2 = int(min(P[:, 0].max() + 3, rgb.shape[1])), int(min(P[:, 1].max() + 3, rgb.shape[0]))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return False
    g = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    outline = np.zeros(g.shape, np.uint8)
    cv2.drawContours(outline, [(c - [x1, y1]).astype(np.int32) for c in cnts], -1, 255, 1)
    on = mag[outline > 0]
    return on.size > 0 and float(np.median(on)) >= min_median


def _ncc(rgb: np.ndarray, tmpl: np.ndarray, box) -> float:
    """How much the frame around ``box`` still looks like ``tmpl`` (the
    object's appearance when seeded): best match of the template over the
    box grown 25 % (and at least the template's size, so a shrunken mask
    still gets a fair look). 1.0 at the image edge, where the window cannot
    hold the template: no verdict rather than a false one."""
    th, tw = tmpl.shape[:2]
    H, W = rgb.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw = max((box[2] - box[0]) * 0.625, tw / 2 + 4)
    hh = max((box[3] - box[1]) * 0.625, th / 2 + 4)
    x1, y1 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
    x2, y2 = int(min(cx + hw, W)), int(min(cy + hh, H))
    win = rgb[y1:y2, x1:x2]
    if win.shape[0] < th or win.shape[1] < tw:
        return 1.0
    return _match(win, tmpl)[1]


def _find(rgb: np.ndarray, tmpl: np.ndarray, others) -> tuple[tuple[int, int], float]:
    """Best placement (top-left) of ``tmpl`` anywhere in the frame, ignoring
    placements centered inside any box in ``others`` (the other tracks:
    identical objects — four batteries in a row — must not pull a searching
    track onto a neighbour that is already tracked). Returns (loc, score)."""
    return _match(rgb, tmpl, exclude=others)


class Tracker:
    """Live positions after ONE VLM answer. The VLM names and boxes the
    objects once; from then on, every frame, SAM2 re-finds each object's mask
    from where it was in the previous frame — its mask's bbox grown 10 %,
    plus the previous mask itself as a prompt (Segmenter.outline ``prior``,
    the mask memory that keeps the footprint constant) — all tracks in ONE
    decoder pass (Segmenter.outline_many) — and the OBB is fitted again on
    the table plane (_det_from). That SAM2 pass costs, with five tracks and
    GPU-synchronized on Thor, hiera_large 110 ms encode + 15 ms masks + 10 ms
    checks = ~135 ms, hiera_tiny (--sam2 tiny) 34 + 15 + 10 = ~65 ms (tiny
    followed the 2026-09-07 recording exactly as well as large: same ids kept,
    same found-again events, fewer stray seeds). It runs at ``sam2_period``
    (5 Hz); every other frame is a LIGHT pass (_light) that slides each box to
    its template's best match nearby on the CPU and drags the outline, OBB,
    base point and mask prior along — so the picture moves at camera rate
    and the GPU stays mostly free for the VLM. The plane geometry is rebuilt
    per frame from the current joints, so a moving head is fine. While the
    VLM is answering, only light passes run: SAM2 beside the VLM slowed both
    (the answer 1.5-2.5x, the display to a stutter, and even one pass a
    second was felt on the robot); the boxes keep moving on the CPU and the
    verdicts resume with the full pass that merges the answer.

    Why SAM2 as the tracker and not a tracker: the official sam2 package has
    no camera predictor (its video predictor wants all frames up front), the
    headless cv2 here has no CSRT/KCF, and a mask is what the geometry needs
    anyway — a bbox tracker would still have to call SAM2 for it.

    SAM2 cannot say "gone". A box prompt where the object WAS returns a
    box-shaped mask of bare table at score 0.96 (ghost.py, 2026-09-07), and
    an object 45 px away is not found at all: on its own the tracker would
    sit on the old spot forever. So every track keeps its APPEARANCE too — a
    crop of the object when it was seeded (``tmpl``) — and its metric
    footprint (``seed``). Each frame the followed mask must (1) overlap the
    previous box (IoU >= 0.3, area within 2x), (2) keep the seed footprint
    (long side 0.6-1.6x, area 0.5-1.8x: not crept onto a hand, a neighbour or
    the table), and (3) still look like the template (NCC >= 0.4, which the
    ghost's white table fails). If not, the template is SEARCHED for over the
    whole frame (_find, ~ms; other tracks' boxes excluded) and, on a match
    >= 0.6, SAM2 is re-prompted there with box + center: a moved object is
    picked up again under the same id the next frame, however far it went.
    Failing that the track goes PENDING — not drawn, searched for on every
    frame for ``patience`` frames (the hand carrying it, the arm over it),
    then dropped. Two tracks on one mask keep the older.

    What no template finds — a NEW object put down, or one taken away for
    good — the VLM re-checks in a BACKGROUND thread: every ``period_s``
    (0 = never), immediately when a track is lost, or on request (c). Its
    answer is seconds old when it lands (5-6 s for 9 objects beside the
    tracker) and is matched to the active tracks by bbox IoU: matched tracks
    go on, unmatched VLM boxes seed new tracks (mask, footprint and template
    from the CURRENT frame), and an active track the VLM missed TWICE in a
    row is dropped — once is a VLM miss, they happen. Pending tracks stay out
    of that matching, so a moved object the search has not caught yet gets a
    fresh track from the VLM rather than blocking one. Track ids are stable,
    so the legend does not renumber when one goes.

    Against objects that "appear" although nothing was put down (the
    2026-09-07 recording, hand shuffling four identical cylinders): a VLM box
    seeds a track only where something stands out from the table AND the
    mask outline runs along real edges (_has_edges; a hand's shadow passed
    the first), not inside a box the VLM labelled a hand or the robot (a
    carried cylinder is passing through), and if a pending track has the same
    footprint and its template matches there, that track comes back instead
    of a new id. A new seed is FRESH — followed but not drawn — until the
    next full pass confirms it. A VLM miss counts against a track only if it
    sat still in view through the whole request. On the other side, one
    failed pass is forgiven (SAM2 returns a partial mask now and then) and
    a track is lost only on the second."""

    # Frames a lost track is searched for before it is dropped: ~6 s at 5
    # fps. A hand carrying a cylinder across the table took 4-5 s on the
    # 2026-09-07 recording; with 10 frames the track was gone before the
    # cylinder was put down and the VLM had to find it again from scratch.
    patience = 30
    # Seconds between SAM2 (full) passes; the frames in between are followed
    # on the CPU (_light). 5 Hz keeps the GPU mostly free for the VLM and is
    # plenty for judging lost/found; positions still update at camera rate.
    sam2_period = 0.2

    def __init__(self, vlm: VlmDetector, seg: Segmenter, plane_z: float,
                 period_s: float) -> None:
        self._vlm, self._seg, self._z, self._period = vlm, seg, plane_z, period_s
        self.tracks: list[dict] = []
        self.pending: list[dict] = []         # lost, still being searched for
        self._next_id = 0
        self._job: tuple[threading.Thread, dict] | None = None   # running VLM request
        self._job_boxes: dict[int, np.ndarray] = {}   # track boxes when it was started
        self._t_vlm = 0.0                     # when the last request was started
        self._t_full = 0.0                    # when the last SAM2 pass ran
        self.full_ms = 0.0                    # and how long it took
        self.vlm_ms: float | None = None      # duration of the last finished one
        self.err = ""
        self.want = True                      # a VLM pass is due (start, lost, c)
        self.n_calls = 0                      # VLM requests made
        self._clutter: list[tuple[float, float]] = []   # table blobs the VLM does not call objects
        self._unexplained_n = 0               # consecutive checks with a new blob
        self._n_full = 0                      # full passes so far

    @property
    def busy(self) -> bool:
        return self._job is not None

    def request(self) -> None:
        self.want = True

    def _kick(self, rgb: np.ndarray, q_torso, q_head) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        roi = _table_roi(q_torso, q_head, self._z, rgb.shape)
        holder: dict = {}
        t0 = time.time()
        self.n_calls += 1

        def run():
            try:
                holder["objs"] = self._vlm.detect(bgr, roi=roi)
            except Exception as e:  # noqa: BLE001 - tracking must keep going
                holder["err"] = f"vlm error: {str(e).splitlines()[0][:90]}"
            holder["ms"] = (time.time() - t0) * 1e3

        th = threading.Thread(target=run, daemon=True)
        th.start()
        self._job, self._t_vlm, self.want = (th, holder), t0, False
        # Where the tracks were on the frame the VLM is looking at: its answer
        # is matched against THESE, not against where they are when it lands
        # seconds later (a moved track would otherwise miss its own box, and
        # the box would seed a second track on the old spot).
        self._job_boxes = {t["id"]: t["bbox_px"].copy() for t in self.tracks}
        for t in self.tracks:
            t["still"] = True                 # cleared if it moves or is re-found meanwhile

    def step(self, rgb: np.ndarray, q_torso, q_head) -> list[dict]:
        """One frame. Every ``sam2_period`` s (or when a VLM answer is in) a
        FULL pass: SAM2 re-finds every track, pending tracks are searched
        for, the VLM answer is merged and the next request started. The
        frames in between — and ALL frames while the VLM is answering — get
        a LIGHT pass: each track's box is moved to where its template matches
        nearby, on the CPU, so the display runs at camera rate while the GPU
        is used 5 times a second at most. Returns the active tracks
        (detection dicts with a stable ``id``)."""
        job_done = self._job is not None and not self._job[0].is_alive()
        if not self.tracks and not self.pending and not job_done:
            # Nothing to follow and no answer to seed from: leave the GPU to
            # the VLM (SAM2 at 5 fps beside it stretched the first answer
            # from ~3 s to ~8 s on the recorded sequences).
            if self.want and self._job is None:
                self._kick(rgb, q_torso, q_head)
            return self.visible
        # While the VLM is answering the GPU is its alone: no SAM2 at all
        # (beside SAM2 at 5 Hz an 8-object answer took 9.6 s instead of
        # ~3.5, and even a 1 Hz pass was felt as a slowdown on the robot).
        # Boxes keep following on the CPU every frame (_light, with the wide
        # window and neighbour exclusion); lost/found verdicts and pending
        # searches resume with the full pass that merges the answer.
        if not job_done and (self.busy or time.time() - self._t_full < self.sam2_period):
            self._light(rgb, q_torso, q_head)
            return self.visible
        t0 = time.time()
        self._full(rgb, q_torso, q_head, job_done)
        self._t_full = time.time()
        self.full_ms = (self._t_full - t0) * 1e3
        return self.visible

    def _light(self, rgb: np.ndarray, q_torso, q_head) -> None:
        """Between SAM2 passes: slide each track's box to where its template
        matches within a window three times the box (~0.3 ms a track), and
        move its outline, OBB, base point and mask prior along with it. No
        verdicts here — a poor match just leaves the box where it was until
        the next full pass, which judges the track properly."""
        H, W = rgb.shape[:2]
        mapper = None
        for t in self.tracks:
            b = t["bbox_px"]
            th, tw = t["tmpl"].shape[:2]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            # Window: three times the box, at least +-60 px (a hand pushing a
            # 40 px cylinder moves it more than a box a frame; with the window
            # at 3x only, boxes froze while the VLM was answering).
            hw, hh = max(b[2] - b[0], tw, 60), max(b[3] - b[1], th, 60)
            x1, y1 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
            x2, y2 = int(min(cx + hw, W)), int(min(cy + hh, H))
            # A big template (the tray, ~200x250 px) costs ~15 ms at full
            # resolution; matched at 1/2 or 1/4 it is ~1 ms and off by at
            # most that many px, which the next SAM2 pass corrects anyway.
            s = 1 if max(tw, th) <= 96 else 2 if max(tw, th) <= 192 else 4
            win, tmpl = rgb[y1:y2, x1:x2], t["tmpl"]
            if s > 1:
                win = cv2.resize(win, (win.shape[1] // s, win.shape[0] // s), interpolation=cv2.INTER_AREA)
                tmpl = cv2.resize(tmpl, (tw // s, th // s), interpolation=cv2.INTER_AREA)
            # Never onto another track: four identical cylinders 50 px apart,
            # and a box slid onto its neighbour, was found a duplicate at the
            # next full pass and silently dropped (2026-09-07 recording).
            others = [(_grow(u["bbox_px"], 0.5, W, H) - [x1, y1, x1, y1]) / s
                      for u in self.tracks if u is not t]
            (x, y), score = _match(win, tmpl, exclude=others)
            # A big template matched at reduced scale is the one that slid
            # onto a hand crossing it (the tray): stricter score, small jumps
            # only. Small objects may jump up to 60 px a frame (~1 m/s).
            if score < (0.5 if s == 1 else 0.7):
                continue
            dx, dy = x1 + (x + tmpl.shape[1] / 2) * s - cx, y1 + (y + tmpl.shape[0] / 2) * s - cy
            if (abs(dx) < 1 and abs(dy) < 1) or max(abs(dx), abs(dy)) > (60 if s == 1 else 15):
                continue
            t["still"] = False
            if mapper is None:
                mapper = bev.build_mapper(q_torso, q_head, self._z)
            p0, p1 = _px_to_plane(mapper, np.array([[cx, cy], [cx + dx, cy + dy]]))
            dxy = p1 - p0
            t["bbox_px"] = np.clip(b + [dx, dy, dx, dy], 0, [W - 1, H - 1, W - 1, H - 1])
            t["contours_px"] = [c + [dx, dy] for c in t["contours_px"]]
            t["base_xy"] = (t["base_xy"][0] + float(dxy[0]), t["base_xy"][1] + float(dxy[1]))
            t["rect_base"] = t["rect_base"] + dxy.astype(np.float32)
            # The low-res logits live on SAM2's 1024x1024 input, the frame
            # resized without keeping aspect: 256/W per px across, 256/H down.
            M = np.float32([[1, 0, dx * 256 / W], [0, 1, dy * 256 / H]])
            t["logits"] = cv2.warpAffine(t["logits"], M, (256, 256),
                                         borderValue=float(t["logits"].min()))

    def _full(self, rgb: np.ndarray, q_torso, q_head, job_done: bool) -> None:
        """The SAM2 pass: follow the tracks, search for the pending ones,
        take in a finished VLM answer, start the next one if due."""
        mapper = bev.build_mapper(q_torso, q_head, self._z)
        self._seg.set_image(rgb)

        kept, pend, lost = [], [], 0

        def keep(d) -> bool:
            # Two tracks on one mask (objects pushed together and apart, or
            # the VLM boxing one object twice): the older track keeps it.
            if any(_iou(d["bbox_px"], u["bbox_px"]) >= 0.5 for u in kept):
                return False
            d["fresh"] = False                        # confirmed by a full pass
            kept.append(d)
            return True

        h, w = rgb.shape[:2]
        found = self._seg.outline_many([_grow(t["bbox_px"], 0.1, w, h) for t in self.tracks],
                                       [t["logits"] for t in self.tracks])
        for t, (cnts, logits) in zip(self.tracks, found):
            d, why = self._follow(t, rgb, mapper, cnts, logits)
            if d is None:
                d = self._search(t, rgb, mapper)
            if d is not None and keep(d):
                d["strikes"] = 0
                continue
            if d is not None:
                why = "on another track's object"
            # One bad pass is forgiven: SAM2 now and then returns a partial
            # mask (the tray's floor instead of the tray when a hand crosses
            # its rim) that fails the checks for a single pass; the track is
            # shown where it was and judged again 0.2 s later.
            if not t.get("strikes") and keep(t):
                t["strikes"] = 1
                continue
            print(f"track {t['id']}:{t['cls']} lost ({why}), searching")
            t["gone"] = 1
            pend.append(t)
            lost += 1
        for t in self.pending:
            d = self._search(t, rgb, mapper)
            if d is not None and keep(d):
                print(f"track {t['id']}:{t['cls']} found again")
            elif t["gone"] < self.patience:           # (a duplicate stays pending)
                t["gone"] += 1
                pend.append(t)
            else:
                print(f"track {t['id']}:{t['cls']} dropped")
        self.tracks, self.pending = kept, pend

        if job_done:
            _, hold = self._job
            self._job = None
            self.vlm_ms, self.err = hold.get("ms"), hold.get("err", "")
            if "objs" in hold:
                self._merge(hold["objs"], rgb, mapper)
                # Whatever still stands on the table unaccounted for right
                # after the answer is something the VLM does not call an
                # object (a cable, a mark): remember it, or it would trigger
                # a request at every check.
                self._clutter = self._unexplained(rgb, mapper, [])
                print(f"track: vlm {self.vlm_ms:.0f}ms, {len(hold['objs'])} boxes -> "
                      f"{len(self.tracks)} tracks: "
                      + " ".join(f"{t['id']}:{t['cls']}" for t in self.tracks)
                      + (f"  ({len(self._clutter)} clutter)" if self._clutter else ""))

        # Ask the VLM again only when the scene gives a reason: a track lost,
        # or a blob on the table (table-colour segmentation of the BEV, ~24
        # ms, so on every other pass) that no track, pending track or known
        # clutter accounts for twice in a row (~0.8 s: not a hand passing).
        # The period is only a safety net (default 30 s; 0 = none).
        self._n_full += 1
        if self._job is None and self._n_full % 2 == 0:
            self._unexplained_n = (self._unexplained_n + 1
                                   if self._unexplained(rgb, mapper, self._clutter) else 0)
        if lost or self._unexplained_n >= 2 or (
                self._period > 0 and time.time() - self._t_vlm >= self._period):
            self.want = True
        # Not while a fresh seed awaits its confirming pass: a request now
        # would hold SAM2 off for the whole answer and keep it hidden that long.
        if self.want and self._job is None and not any(t.get("fresh") for t in self.tracks):
            self._unexplained_n = 0
            self._kick(rgb, q_torso, q_head)

    def _unexplained(self, rgb: np.ndarray, mapper, clutter) -> list[tuple[float, float]]:
        """Base (X, Y) of blobs on the table that no track accounts for.
        The frame warped to the BEV canvas, table colour = its median in Lab;
        a pixel is 'something' if its hue differs (|da|+|db| > 14) or it is
        much darker (the black tray) — a shadow keeps the table's hue and is
        ignored, which is what made this usable next to a hand. Blobs
        touching the canvas border are the arm (and whatever it holds or
        covers), blobs under 300 px (~3x2 cm) are specks. A blob is explained
        by a track or pending track whose OBB is within 3 cm of its centroid,
        or by a ``clutter`` point within 5 cm."""
        canvas = mapper.warp(rgb)
        lab = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB).astype(np.int16)
        valid = canvas.max(2) > 0
        if not valid.any():
            return []
        med = np.median(lab[valid].reshape(-1, 3), axis=0)
        mask = (((np.abs(lab[..., 1] - med[1]) + np.abs(lab[..., 2] - med[2]) > 14)
                 | (lab[..., 0] < med[0] - 90)) & valid).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        H, W = mask.shape
        polys = [t["rect_base"].astype(np.float32) for t in self.tracks + self.pending]
        out = []
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            if a < 300 or x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                continue
            X, Y = mapper.bev_px_to_base(float(cent[k][0]), float(cent[k][1]))
            if any(cv2.pointPolygonTest(p, (float(X), float(Y)), True) > -0.03 for p in polys):
                continue
            if any(abs(X - cx) < 0.05 and abs(Y - cy) < 0.05 for cx, cy in clutter):
                continue
            out.append((float(X), float(Y)))
        return out

    @property
    def visible(self) -> list[dict]:
        """The tracks to draw: all active ones except fresh seeds."""
        return [t for t in self.tracks if not t.get("fresh")]

    def _follow(self, t: dict, rgb: np.ndarray, mapper, cnts, logits
                ) -> tuple[dict | None, str]:
        """The track's object on this frame from where it was on the last one
        — ``cnts``/``logits`` are SAM2's answer to its box + mask prior, from
        the batched outline_many — or (None, why) if that is not it."""
        d = _det_from(t["cls"], t["bbox_px"], cnts, logits, mapper)
        if d is None:
            return None, "no mask on the table"
        mb = _mask_box(d["contours_px"])
        iou, ar = _iou(mb, t["bbox_px"]), _area(mb) / _area(t["bbox_px"])
        if iou < 0.3 or not 0.5 <= ar <= 2.0:
            return None, f"box iou {iou:.2f} area x{ar:.2f}"
        why = self._check(d, t, rgb, mb)
        return (None, why) if why else (self._carry(d, t, mb), "")

    def _search(self, t: dict, rgb: np.ndarray, mapper) -> dict | None:
        """The track's object anywhere on this frame by its template, then
        SAM2 prompted there with box + center (no prior: the prior is where
        it WAS). None if nothing matches or what does is not it."""
        # Other tracks' boxes, grown by half: a placement centred just outside
        # a tracked neighbour still lands the mask on that neighbour (four
        # identical cylinders), and the duplicate was then dropped while the
        # real object got a new id.
        H, W = rgb.shape[:2]
        others = [u["bbox_px"] for u in self.tracks if u is not t]
        (x, y), score = _find(rgb, t["tmpl"], [_grow(b, 0.5, W, H) for b in others])
        if score < 0.6:
            return None
        th, tw = t["tmpl"].shape[:2]
        d = _to_det(dict(label=t["cls"], bbox_px=np.array([x, y, x + tw, y + th], np.float64)),
                    self._seg, mapper)
        if d is None:
            return None
        mb = _mask_box(d["contours_px"])
        if self._check(d, t, rgb, mb) or any(_iou(mb, b) >= 0.5 for b in others):
            return None
        d = self._carry(d, t, mb)
        d.update(still=False, miss=0)         # it moved: the VLM's old box says nothing
        return d

    @staticmethod
    def _check(d: dict, t: dict, rgb: np.ndarray, mb) -> str:
        """Why ``d`` is not track ``t``'s object: footprint off the seed's, or
        no longer looking like it. Empty string if it passes."""
        rl = d["dims_m"][0] / t["seed"][0]
        ra = d["dims_m"][0] * d["dims_m"][1] / (t["seed"][0] * t["seed"][1])
        if not 0.6 <= rl <= 1.6 or not 0.5 <= ra <= 1.8:
            return f"size vs seed: long x{rl:.2f} area x{ra:.2f}"
        if not _stands_out(rgb, mb) or not _has_edges(rgb, d["contours_px"]):
            return "bare table"
        ncc = _ncc(rgb, t["tmpl"], mb)
        if ncc < 0.4:
            return f"looks different (ncc {ncc:.2f})"
        return ""

    @staticmethod
    def _carry(d: dict, t: dict, mb) -> dict:
        d.update(bbox_px=mb, id=t["id"], miss=t["miss"], seed=t["seed"],
                 tmpl=t["tmpl"], gone=0, still=t.get("still", True))
        return d

    def _merge(self, objs: list[dict], rgb: np.ndarray, mapper) -> None:
        """Reconcile the active tracks with a VLM answer: match by bbox IoU
        (against the tracks' boxes on the frame the VLM saw), count misses,
        seed tracks for unmatched boxes on the current frame."""
        # Boxes the VLM put on the robot or a hand: no object is seeded inside
        # them. A cylinder being carried across the table got a track of its
        # own (its old track pending meanwhile), which died on the next pass.
        hands = [o["bbox_px"] for o in objs if o["ignore"]]
        objs = [o for o in objs if not o["ignore"]]
        used: set[int] = set()
        for t in self.tracks:
            then = self._job_boxes.get(t["id"], t["bbox_px"])
            ious = [(-1.0 if j in used else _iou(o["bbox_px"], then))
                    for j, o in enumerate(objs)]
            j = int(np.argmax(ious)) if ious else -1
            if j >= 0 and ious[j] >= 0.4:
                used.add(j)
                t["miss"] = 0
            elif t.get("still", True):
                # A miss counts only for a track that sat still in view the
                # whole time: one carried by a hand across two VLM cycles was
                # otherwise dropped as "missed twice" and re-seeded under a
                # new id the moment it was put down.
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] < 2]
        for j, o in enumerate(objs):
            if j in used:
                continue
            d = _to_det(o, self._seg, mapper)
            if d is None:
                continue
            mb = _mask_box(d["contours_px"])
            # The box is seconds old. If the object is no longer in it, SAM2
            # returns a box-shaped mask of table (see the class docstring):
            # seed only where something stands out from the table AND the
            # outline runs along real edges (a hand's shadow passes the first).
            if not _stands_out(rgb, mb) or not _has_edges(rgb, d["contours_px"]):
                continue
            if any(_iou(mb, hb) > 0 or _iou(o["bbox_px"], hb) > 0.1 for hb in hands):
                continue                                  # in or at a hand
            # The VLM box may frame an object a track already holds with a
            # different extent (its box vs our mask box): same thing, skip.
            if any(_iou(mb, t["bbox_px"]) >= 0.4 for t in self.tracks):
                continue
            # A PENDING track's object, seen by the VLM where it is now (the
            # search has not caught it yet): same footprint as that track's
            # seed and its template matches here -> the old id comes back,
            # instead of a new id that would then fight the old one.
            for p in self.pending:
                if (0.6 <= d["dims_m"][0] / p["seed"][0] <= 1.6
                        and 0.5 <= d["dims_m"][0] * d["dims_m"][1]
                        / (p["seed"][0] * p["seed"][1]) <= 1.8
                        and _ncc(rgb, p["tmpl"], mb) >= 0.5):
                    self.pending.remove(p)
                    d.update(cls=p["cls"], color=p["color"])   # keep its name too
                    self.tracks.append(self._carry(d, p, mb))
                    print(f"track {p['id']}:{p['cls']} found again (by the VLM)")
                    break
            else:
                b = mb.astype(int)
                tmpl = rgb[b[1]:b[3] + 1, b[0]:b[2] + 1].copy()
                # A template without contrast cannot be matched or checked.
                if min(tmpl.shape[:2]) < 4 or float(tmpl.std()) < 8.0:
                    continue
                # fresh: not shown until the next full pass has confirmed it
                # (a box the VLM drew on a neighbour's shadow, or on an
                # object already tracked with a different extent, is gone by
                # then and never flickers on screen).
                d.update(bbox_px=mb, id=self._next_id, miss=0, seed=d["dims_m"],
                         tmpl=tmpl, gone=0, fresh=True)
                self._next_id += 1
                self.tracks.append(d)


def _px(base_xy, z: float, q_torso, q_head, S: float) -> tuple[int, int]:
    u, v = dp.base_to_pixel((base_xy[0], base_xy[1], z), q_torso, q_head)
    return int(round(u * S)), int(round(v * S))


def _draw_bev(bgr, dets, q_torso, q_head, plane_z: float, height: int,
              th: int, fs: float) -> np.ndarray:
    """The frame warped at ``plane_z`` (the plane the quads were cast onto),
    with the fitted OBB of each detection drawn from its BASE-frame corners,
    its center as a cross, and live_detect_bev's 10 cm grid.
    Same vertical flip as live_detect_bev.draw (see ld._flip_v): forward is
    RIGHT, the robot's left is UP, a real top-down view. Resized to ``height``
    so it sits beside the raw frame."""
    canvas = bev.build_mapper(q_torso, q_head, plane_z).warp(bgr)
    S = height / canvas.shape[0]
    disp = cv2.flip(cv2.resize(canvas, None, fx=S, fy=S,
                               interpolation=cv2.INTER_LINEAR), 0)
    ld._grid(disp, S)
    h = disp.shape[0]
    for k, d in enumerate(dets):
        col = d["color"]
        poly = np.array([ld._base_to_px(x, y, S, h) for x, y in d["rect_base"]],
                        dtype=np.int32)
        cv2.polylines(disp, [poly], True, col, th)
        ru, rv = ld._base_to_px(*d["base_xy"], S, h)
        cv2.drawMarker(disp, (ru, rv), col, cv2.MARKER_CROSS, int(18 * S), th)
        ld._text(disp, f"{d.get('id', k)}:{d['cls']}",
                 (int(np.clip(poly[:, 0].min(), 2, disp.shape[1] - 90 * S)),
                  int(np.clip(poly[:, 1].min() - 5 * S, 12 * S, h - 4))),
                 col, fs)
    ld._text(disp, f"BEV @ z={plane_z:.3f}   [+x fwd ->right,  +y left ->UP]",
             (8, int(18 * S)), _COLOR, fs)
    return disp


_KEYS = "c|Space capture  t track  l live  s save  q quit"


def draw(bgr, dets, q_torso, q_head, plane_z: float, title: str,
         footer: str, scale: float = 1.0, err: str = "") -> np.ndarray:
    """Left: the frame, with each VLM box (thin), the SAM2 outline (thick),
    the OBB center (cross, the base-frame point projected back onto the
    table plane) and one legend row per detection. Right: the BEV of the same
    frame at the same plane (_draw_bev), where the fitted OBB is a plain
    rectangle and its yaw can be read off. With no detections (the live
    preview) it is just the two pictures and the grid."""
    S = float(scale)
    disp = (cv2.resize(bgr, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
            if S != 1.0 else bgr.copy())
    th = max(1, int(round(2 * S)))
    fs, fh = 0.5 * S, int(round(16 * S))
    right = _draw_bev(bgr, dets, q_torso, q_head, plane_z, disp.shape[0], th, fs)
    ld._text(disp, title, (8, int(18 * S)), _COLOR, fs)
    if err:
        ld._text(disp, err, (8, int(18 * S) + fh), (60, 60, 255), 0.44 * S)

    for k, d in enumerate(dets):
        col = d["color"]
        tag = f"{d.get('id', k)}:{d['cls']}"
        row = (8, int(38 * S) + fh * (k + (1 if err else 0)))
        b = (d["bbox_px"] * S).astype(int)
        cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), col, 1)
        cv2.polylines(disp, [(c * S).astype(np.int32) for c in d["contours_px"]],
                      True, col, th)
        X, Y = d["base_xy"]
        cv2.drawMarker(disp, _px((X, Y), plane_z, q_torso, q_head, S), col,
                       cv2.MARKER_CROSS, int(18 * S), th)
        ld._text(disp, tag,
                 (int(np.clip(b[0], 2, disp.shape[1] - 90 * S)),
                  int(np.clip(b[1] - 5 * S, 12 * S, disp.shape[0] - 4))),
                 col, fs)
        ld._text(disp, f"{tag} ({X:.3f},{Y:+.3f}) {d['yaw']:5.1f}deg "
                 f"{d['dims_m'][0]:.2f}x{d['dims_m'][1]:.2f}m", row, col, 0.44 * S)

    ld._text(disp, footer, (8, disp.shape[0] - int(8 * S)), (255, 255, 255), 0.45 * S)
    return np.hstack([disp, right])


# live_detect_bev's HTTP viewer, with a page for this script's keys. The
# handler reads its module's _PAGE at request time, so swapping it is enough.
_PAGE = b"""<!doctype html><meta charset=utf-8><title>VLM detector</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{background:#111;color:#ddd;font:14px system-ui;margin:0;padding:10px;text-align:center}
 img{width:100%;max-width:1900px;height:auto;border:1px solid #333}
 button{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;
        padding:6px 14px;margin:2px;font:14px system-ui;cursor:pointer}
 button:hover{background:#333}
 #h{color:#777;font-size:12px;margin-top:6px}
</style>
<div>
 <button onclick="k('c')"><b>c capture</b></button>
 <button onclick="k('t')">t track</button>
 <button onclick="k('l')">l live</button>
 <button onclick="k('s')">s save</button>
</div>
<img id=v alt="waiting for the first frame...">
<div id=h>c (or Space) captures the current frame and sends it to the VLM; the
result stays until the next capture, l returns to the live view. t toggles
TRACK mode: one VLM answer, then SAM2 follows the objects live (c there asks
the VLM again). s writes the shown picture on the robot. Ctrl-C in the
terminal to quit.</div>
<script>
function k(a){fetch('/cmd?k='+encodeURIComponent(a))}
addEventListener('keydown',e=>{
  const m={'c':'c',' ':'c','t':'t','l':'l','s':'s'};
  if(m[e.key]!==undefined){e.preventDefault();k(m[e.key])}
});
// Poll the latest frame instead of an MJPEG <img>: a plain JPEG response is
// rendered by every browser as soon as it arrives, whereas Chrome held the
// last MJPEG part (the RESULT) back until another part followed. ?since=
// makes an unchanged frame a 204, so a still result costs nothing.
let seq=-1, old=null;
async function poll(){
  try{
    const r=await fetch('/frame.jpg?since='+seq,{cache:'no-store'});
    if(r.status===200){
      seq=+r.headers.get('X-Seq');
      const u=URL.createObjectURL(await r.blob());
      document.getElementById('v').src=u;
      if(old)URL.revokeObjectURL(old); old=u;
    }
    setTimeout(poll,100);
  }catch(e){setTimeout(poll,1000)}
}
poll();
</script>"""


def _replay_frames(path: str):
    """(rgb, q_torso, q_head) from a recorded sequence (record_track.py or
    capture_bev.py frame_*.npz), paced by the recorded timestamps and looping
    — a camera stand-in for --replay, so the tracker can be judged on the
    same recording again and again without the robot."""
    files = sorted(Path(path).glob("frame_*.npz"))
    if not files:
        raise SystemExit(f"no frame_*.npz in {path}")
    data = [np.load(f) for f in files]
    ts = np.array([float(d["timestamp"]) for d in data])
    print(f"replay: {len(files)} frames, {ts[-1] - ts[0]:.1f}s "
          f"({(len(files) - 1) / max(ts[-1] - ts[0], 1e-9):.1f} fps) from {path}, looping")
    while True:
        t0 = time.time()
        for d, t in zip(data, ts):
            time.sleep(max(0.0, t0 + (t - ts[0]) - time.time()))
            yield d["rgb"], np.asarray(d["q_torso"], np.float64), np.asarray(d["q_head"], np.float64)
        print("replay: restarting from the first frame")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--vlm-model", default="qwen3.5",
                    help="served model id (vllm serve --served-model-name)")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-call timeout (s)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--plane", type=float, default=TABLE_Z_M,
                    help="base z of the table top (m)")
    ap.add_argument("--angle", type=float, default=30.0, help="head-down align angle")
    ap.add_argument("--redetect", type=float, default=30.0, metavar="SEC",
                    help="track mode: safety-net period for asking the VLM again; it is "
                         "normally asked only when a track is lost or an unknown blob "
                         "appears on the table (0 = no period)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="display magnification of the 960x600 frame + overlay; "
                         "the BEV panel is scaled to the same height, so the "
                         "composite is ~1890 px wide at 1.0")
    ap.add_argument("--serve", type=int, metavar="PORT", default=None,
                    help="MJPEG over HTTP instead of a cv2 window (cv2 here is headless)")
    ap.add_argument("--sam2", choices=sorted(Segmenter.SIZES), default="tiny",
                    help="SAM2 encoder size (checkpoints/); smaller = faster tracking")
    ap.add_argument("--replay", metavar="DIR", default=None,
                    help="no robot: play a recorded sequence (record_track.py, "
                         "frame_*.npz) at its recorded pace, looping, as the camera")
    args = ap.parse_args()

    vlm = VlmDetector(args.base_url, args.vlm_model, args.timeout, args.max_tokens)
    try:
        vlm.check()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - connection refused etc.
        raise SystemExit(f"no vLLM at {args.base_url}: {e}\n"
                         f"start it first (docker command in this file's docstring)")
    print(f"VLM {args.vlm_model} at {args.base_url}: every object on the table "
          f"at z={args.plane:.3f}")
    t0 = time.time()
    seg = Segmenter(args.sam2)
    print(f"SAM2 {args.sam2} ready ({time.time() - t0:.1f}s)")
    table_z = float(args.plane)

    out = _HERE / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if args.replay:
        robot_ctx = contextlib.nullcontext(None)
        frames = _replay_frames(args.replay)
    else:
        configs = get_robot_config()
        configs.enable_sensor("head_camera")
        configs.sensors["head_camera"].transport = "zenoh"
        robot_ctx = Robot(configs=configs)
        frames = None

    with robot_ctx as robot:
        if robot is not None:
            if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
                print("Warning: camera streams may not be active")
            # The head motors are DISABLED whenever no dexcontrol client is
            # connected (measured 2026-09-07: the head holds while a Robot
            # session lives and drops to its -65 deg rest within ~10 s of the
            # session closing), so EVERY run starts with a limp head and must
            # enable it. The enable takes effect with a delay: a pitch command
            # sent right after set_mode was ignored, the same command 1 s
            # later moved the head. Hence the wait, and a check with retry —
            # there is no mode query for the head to confirm it any other way.
            target = np.deg2rad(np.rad2deg(float(robot.torso.pitch_angle)) - args.angle)
            for attempt in range(3):
                robot.head.set_mode("enable")
                time.sleep(1.0)
                set_head_pitch(robot, angle=args.angle)
                pitch = float(np.asarray(robot.head.get_state()["pos"], float)[0])
                if abs(pitch - target) < np.deg2rad(3.0):
                    break
                print(f"head pitch {np.rad2deg(pitch):.1f} deg, target {np.rad2deg(target):.1f}: "
                      f"not reached (attempt {attempt + 1}/3), re-enabling")
            else:
                print("WARNING: head did not reach the target pitch; check the head motors")

        win = "VLM detector"
        ctl = ld._Control() if args.serve else None
        if ctl is not None:
            ld._PAGE = _PAGE
            ld._KEYMAP.update({"c": ord("c"), "t": ord("t"), "l": ord("l")})
            ld.serve(args.serve, ctl)
            print(f"VLM detector on http://0.0.0.0:{args.serve}  (Ctrl-C to quit)")
        else:
            try:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            except cv2.error as e:
                raise SystemExit(
                    f"cv2 cannot open a window ({e.err.splitlines()[0].strip()}).\n"
                    f"This build of cv2 is headless — rerun with --serve 8088") from None
            print(f"VLM detector — {_KEYS}")

        def show(img) -> None:
            if ctl is not None:
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    ctl.publish(jpg.tobytes())
            else:
                cv2.imshow(win, img)

        # The last capture's result, shown instead of the live view until the
        # next capture (or l). ``dets`` goes with it for s.
        #
        # A frame is published only when the picture changes: the live preview
        # at most LIVE_FPS (the camera runs faster, and every frame is ~90 KB
        # of JPEG — at camera rate that is ~20 Mbps, enough over Wi-Fi to
        # queue a second of frames in the socket), CAPTURED and RESULT once
        # each. The page polls /frame.jpg?since=, so a still frame is fetched
        # once and then answered with 204s; a browser opening the page later
        # gets the latest frame on its first poll.
        #
        # In TRACK mode (t) ``tracker`` is set and every frame is processed
        # and published: the loop then runs at SAM2's pace (~7 fps with three
        # objects), which is also the publish rate.
        result, dets, shown = None, [], None
        tracker: Tracker | None = None
        next_live = 0.0
        while True:
            if frames is not None:                        # --replay
                rgb, q_torso, q_head = next(frames)
            else:
                rgb = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"]).get("left_rgb")
                rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
                if rgb is None:
                    continue
                q_torso, q_head = ld._joints(robot)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if tracker is not None:
                # Every frame is a cheap CPU pass; the Tracker itself runs SAM2
                # only every sam2_period, and not at all while the VLM answers
                # (Tracker.step). Stepping and publishing are capped at
                # TRACK_FPS: the camera is faster and identical frames need not
                # be redrawn.
                if time.time() >= next_live:
                    next_live = time.time() + 1.0 / TRACK_FPS
                    t0 = time.time()
                    dets = tracker.step(rgb, q_torso, q_head)
                    ms = (time.time() - t0) * 1e3
                    state = ("vlm running (cpu follow only)" if tracker.busy else
                             f"vlm {tracker.vlm_ms:.0f}ms" if tracker.vlm_ms else "")
                    state += f"   vlm calls {tracker.n_calls}"
                    shown = draw(bgr, dets, q_torso, q_head, table_z,
                                 f"TRACK   table z={table_z:.3f}   {len(dets)} obj   "
                                 f"step {ms:.0f}ms  sam2 {tracker.full_ms:.0f}ms"
                                 f"@{1 / tracker.sam2_period:.0f}Hz   {state}",
                                 _KEYS, args.scale, tracker.err)
                    show(shown)
            elif result is None:
                if time.time() >= next_live:
                    show(draw(bgr, [], q_torso, q_head, table_z,
                              f"LIVE   table z={table_z:.3f}", _KEYS, args.scale))
                    next_live = time.time() + 1.0 / LIVE_FPS
            elif shown is not result:
                show(result)
                shown = result

            key = ctl.take_key() if ctl is not None else cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("t"):
                if tracker is None:
                    tracker = Tracker(vlm, seg, table_z, args.redetect)
                    result = None
                    print("track mode on")
                else:
                    tracker, shown = None, None
                    print("track mode off")
            if key in (ord("c"), 32) and tracker is not None:
                tracker.request()                         # ask the VLM again now
            elif key in (ord("c"), 32):                   # capture THIS frame
                show(draw(bgr, [], q_torso, q_head, table_z,
                          f"CAPTURED   table z={table_z:.3f}   waiting for the VLM...",
                          _KEYS, args.scale))
                t0, err = time.time(), ""

                def partial(dets, _t0=t0, _bgr=bgr, _q=(q_torso, q_head)):
                    # Called as each object closes in the streamed answer:
                    # the picture fills in while the model is still writing.
                    show(draw(_bgr, dets, *_q, table_z,
                              f"RESULT (streaming)   {len(dets)} det so far   "
                              f"{(time.time() - _t0) * 1e3:.0f}ms", _KEYS, args.scale))

                try:
                    dets = detect(rgb, q_torso, q_head, vlm, seg, table_z, progress=partial)
                except Exception as e:  # noqa: BLE001 - a viewer must keep going
                    dets, err = [], f"vlm error: {str(e).splitlines()[0][:90]}"
                    print(err)
                ms = (time.time() - t0) * 1e3
                if (not dets and not err and vlm.last_raw
                        and '"o":[]' not in vlm.last_raw.replace(" ", "")):
                    err = f"unparsed: {vlm.last_raw.strip()[:90]}"
                result = draw(bgr, dets, q_torso, q_head, table_z,
                              f"RESULT   table z={table_z:.3f}   {len(dets)} det   "
                              f"vlm {ms:.0f}ms", _KEYS, args.scale, err)
                print(f"capture: {len(dets)} det in {ms:.0f}ms  raw: {vlm.last_raw.strip()}")
                for d in dets:
                    print(f"  {d['cls']:9s} xy=({d['base_xy'][0]:.3f},"
                          f"{d['base_xy'][1]:+.3f}) yaw={d['yaw']:5.1f} "
                          f"size={d['dims_m'][0]:.3f}x{d['dims_m'][1]:.3f}")
            if key == ord("l"):
                result, tracker, shown = None, None, None
            if key == ord("s") and shown is not None:      # the result or the track frame
                p = out / f"vlm_capture_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(p), shown)
                print(f"saved {p}  (table z {table_z:.3f}, {len(dets)} det)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
