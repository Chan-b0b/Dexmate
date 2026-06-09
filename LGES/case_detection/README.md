# Transparent-case detection

Detect the position + yaw of the **clear plastic case** that sits inside the
**yellow source bin**, so the arm can grasp it after it's moved.

## What we learned from real frames (don't re-litigate this)

We captured real frames and tested the obvious approaches. Results:

| signal | verdict |
|---|---|
| **Stereo depth hole** | ❌ dead — stereo sees *through* the clear case to the bin floor; no case-shaped void. Depth is smooth/valid where the case is. |
| **Yellow-bin ROI** | ✅ reliable (20/20 frames) — bright opaque bin, trivial HSV segmentation. Used to crop a clean search region. |
| **Classical RGB (Canny / edge-density / texture)** | ❌ too entangled — the case's outer boundary is low-contrast on yellow, while the strongest edges are internal ribs + specular reflections + bin walls. No clean isolation. |
| **Learned (YOLO-OBB)** | ✅ chosen approach — the case is clearly visible to a human, so it's learnable; the model runs on the clean bin-ROI crop. |

So: **depth is out, classical is out, we detect the case with a trained
oriented-box model inside the bin ROI.** The earlier depth-hole detector
(`detect_case.py`) is kept only as a reference / negative result.

## Architecture

```
full RGB frame
   │  bin ROI:  cfg.BIN_ROI  >  detect_bin (learned)  >  find_bin (HSV)
   ▼
bin-ROI crop  ──►  YOLO-OBB  ──►  oriented box (cx,cy,w,h,angle)   detect_case_obb.py
                                      │  map crop→full px, deproject @ known z
                                      ▼
                          CaseDetection(center_px, angle, xyz_cam)
```

Two detectors, both YOLO, both auto-labeled by SAM2 (run the labeler once per
target):

- **bin** (`--target bin`): axis-aligned box on the **full frame** → the ROI.
  Use when the bin's image position varies (HSV `find_bin` then breaks).
- **case** (`--target case`): oriented box (OBB) on the **bin-ROI crop** → pose.

## Files

| file | role |
|---|---|
| `config.py` | intrinsics, bin HSV/ROI, known z, `OBB_MODEL_PATH`, `BIN_MODEL_PATH`, `USE_BIN_MODEL` |
| `bin_roi.py` | HSV `find_bin()` (fallback ROI) / crop helpers |
| `capture.py` | robot-side capture (`--keyboard` for SPACE-to-grab) → `data/<timestamp>/` |
| `sam2_autolabel.py` | **auto-label** `--target {case,bin}`: click keyframes → SAM2 propagates → labels |
| `obb_label.py` | mask → OBB line (case) or axis-aligned bbox line (bin); sanity flags |
| `review.py` | contact-sheet montage of labels; flags drift / lost-track in red |
| `prepare_dataset.py` | labeled crops → YOLO train/val + `data.yaml` (`--name case|bin`) |
| `train.py` | `--target {case,bin}` → `runs/{obb,detect}/<target>/weights/best.pt` |
| `detect_bin.py` | **runtime bin detector** (learned ROI) — replaces `find_bin` |
| `detect_case_obb.py` | **runtime case backend**: bin ROI → YOLO-OBB → pose |
| `diagnose.py` / `run_test.py` | overlays + detection-rate / repeatability (`run_test.py --obb`) |
| `detect_case.py` / `test_synthetic.py` | depth-hole detector — reference only, **does not work** here |

## Setup

Two environments. **`capture.py` runs on the robot** (needs `dexcontrol` + ZED).
**Everything else runs on a local / GPU box** — copy this `case_detection/`
folder and the captured `data/` over; no robot stack needed there.

Local box (Python ≥ 3.10):

```bash
# 1. torch FIRST, matched to your CUDA (SAM2 needs torch >= 2.5.1)
#    pick the right command from https://pytorch.org/get-started/

# 2. the rest
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"

# 3. SAM2 checkpoint
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# 4. system deps for the interactive clicker (matplotlib TkAgg) + a DISPLAY
sudo apt-get install -y python3-tk          # and X11 / VNC if remote

# 5. copy captures from the robot
#    robot:  data/case/<ts>/ , data/bin/<ts>/   ->   local: case_detection/data/...

# sanity check
python -c "import torch, ultralytics, cv2; from sam2.build_sam import build_sam2_video_predictor; print('env OK, cuda', torch.cuda.is_available())"
```

Note: the SAM2 **clicking** step needs a display (X11 / VNC if the box is
remote); **training and inference** don't.

## Workflow

Capture on the robot; **label + train on a GPU box** (copy `data/` over).

```bash
# --- on the robot: collect a smooth sequence, head held STILL ---
# slowly slide/rotate the case through the bin while it captures
python capture.py --target case --n 200 --interval 1.0   # -> data/case/<timestamp>/
#   or grab frames by hand in a preview window (SPACE = save, q = quit):
python capture.py --target case --n 200 --keyboard

# --- on the GPU box (copy data/ over) ---
pip install ultralytics
pip install sam2          # + download sam2.1_hiera_small.pt and its cfg;
                          #   set cfg.SAM2_CHECKPOINT / SAM2_MODEL_CFG

# 1. auto-label: click the case on a keyframe, SAM2 propagates over the crop
python sam2_autolabel.py --target case --keyframes 0
#    -> labeled_case/{images,labels} + out/overlay_*.png

# 2. review; drift/lost-track (RED) -> re-run with those as extra keyframes
python review.py --images labeled_case/images --labels labeled_case/labels
python sam2_autolabel.py --target case --keyframes 0,47,118

# 3. split, train, point config at the weights
python prepare_dataset.py --src labeled_case --name case
python train.py --target case --epochs 150
#    -> set cfg.OBB_MODEL_PATH = runs/obb/case/weights/best.pt

# 4. evaluate on held-out captures
python run_test.py --obb            # detection rate + pose repeatability + out/*.png
```

### Bin detector (when the bin position varies at runtime)

Same tools, `--target bin`. Capture frames with the bin at **varied positions**
(pan the head / move the bin) so the detector generalizes. SAM2 labels the
opaque bin trivially.

```bash
python capture.py --target bin --keyboard                # -> data/bin/<timestamp>/
python sam2_autolabel.py --target bin --keyframes 0      # axis-aligned box, full frame
python review.py --images labeled_bin/images --labels labeled_bin/labels
python prepare_dataset.py --src labeled_bin --name bin
python train.py --target bin --epochs 100
#    -> set cfg.BIN_MODEL_PATH and cfg.USE_BIN_MODEL = True
```

At runtime the ROI source is a cascade: `cfg.BIN_ROI` (fixed) → learned
`detect_bin` (if `USE_BIN_MODEL`) → HSV `find_bin`.

## How much data

One object class on a fairly fixed backdrop: aim for **~150–300 frames**
spanning the full range of case positions and rotations in the bin (plus any
lighting / bin-shift you expect at runtime). With SAM2 propagation that's a
handful of clicks. `train.py` adds heavy rotation/flip augmentation so it learns
angle from fewer frames.

## Why SAM2 only for labeling

SAM2 segments the clear case well from a click, but it's too heavy / prompt-
dependent for robot runtime. So we use it **offline to manufacture labels**, then
distill that into a fast YOLO-OBB the Orin runs in real time. The capture must be
a smooth sequence (head still) so propagation tracks; the median bin ROI keeps
the crop stable frame-to-frame so apparent motion is the case, not ROI wobble.

## Reading the verdict

`run_test.py --obb` reports **detection rate** and **pose repeatability** (std of
x/y/z/yaw across frames of a static case). Grasp-grade bar: rate ≳ 95 %, xy std
< ~5 mm, yaw std < ~3°.

## Notes

- Poses are camera-frame. `capture.py` stores `q_torso`/`q_head` per frame, so
  base-frame conversion can reuse `perception/perception.py:transform_zed_point_to_base`.
- `z` comes from `cfg.FLOOR_Z_M` (measure it) or the median bin-floor depth.
- Opaque **batteries** are a separate, easier problem (SAM + `minAreaRect`),
  not handled here.
