"""Configuration for the transparent-case detection test framework.

The case is fully clear, unmarked, sits on an opaque box floor under fixed
lighting, and arrives at an unknown pose. We cannot segment it directly, but
its FOOTPRINT SHAPE and the FLOOR Z are known. So detection is reframed as:

    find where stereo depth FAILS (the case-shaped "hole" on the floor),
    then keep only blobs whose size matches the known footprint at the known z.

Two numbers below (CASE_FOOTPRINT_M, FLOOR_Z_M) are the only things you must
measure for your real setup. Everything else has a working default.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Camera intrinsics — head ZED left camera.
# Copied from perception/perception.py:get_3d_zed_point (same camera/stream).
# ----------------------------------------------------------------------------
FX: float = 366.21429443359375
FY: float = 366.21429443359375
CX: float = 497.73809814453125
CY: float = 315.53277587890625

# Image size used ONLY by the synthetic self-test. Real detection reads the
# shape from the captured depth array, so this need not match exactly.
IMG_W: int = 960
IMG_H: int = 600

# ----------------------------------------------------------------------------
# Known geometry — MEASURE THESE for your case.
# ----------------------------------------------------------------------------
# Case footprint (the two outer in-plane dimensions, metres). Order doesn't
# matter; the detector compares against long/short automatically.
CASE_FOOTPRINT_M: tuple[float, float] = (0.20, 0.12)  # <-- TODO measure

# Distance (m) from the camera to the floor the case rests on, i.e. the depth
# the floor reports. If None, the detector estimates it as the median valid
# depth inside the search ROI each frame (works if the floor dominates the ROI).
FLOOR_Z_M: float | None = None  # <-- set to a measured value for best results

# ----------------------------------------------------------------------------
# Hole-detection thresholds.
# ----------------------------------------------------------------------------
# A pixel is part of the "case hole" if its depth is invalid (no stereo match)
# OR it deviates from the floor depth by more than this (case refracts/blocks
# the floor, so it reads as missing or wrong depth).
FLOOR_DEV_TOL_M: float = 0.03

# Valid metric-depth range; anything outside is treated as invalid.
DEPTH_MIN_M: float = 0.15
DEPTH_MAX_M: float = 2.0

# Morphology kernel (px) to close speckle inside the hole and drop noise.
MORPH_KERNEL_PX: int = 7

# Restrict the search to a region of interest (x, y, w, h) in pixels, or None
# for the whole frame. Narrow this to the box floor to reject clutter.
CASE_SEARCH_ROI: tuple[int, int, int, int] | None = None

# ----------------------------------------------------------------------------
# Acceptance thresholds.
# ----------------------------------------------------------------------------
# Max relative error per side (detected px vs expected px) to call it a match.
SIZE_TOL: float = 0.40
# Minimum size_score in [0,1] to report found=True.
MIN_SCORE: float = 0.55
# Ignore blobs smaller than this fraction of the expected footprint area.
MIN_AREA_FRAC: float = 0.30

# ----------------------------------------------------------------------------
# Bin ROI — the clear case lives inside the opaque YELLOW source bin, so we
# localise the bin first (HSV colour) and only search for the case inside it.
# ----------------------------------------------------------------------------
# Yellow range in HSV (OpenCV H is 0-179). High saturation floor rejects dull
# tan/wood/cardboard background that shares the hue. Widen if the bin isn't found.
BIN_HSV_LO: tuple[int, int, int] = (10, 120, 90)
BIN_HSV_HI: tuple[int, int, int] = (40, 255, 255)
# Reject a "bin" smaller than this fraction of the frame (rejects yellow clutter).
BIN_MIN_AREA_FRAC: float = 0.05
# Shrink the bin bbox inward by this fraction so the search excludes the rim/walls.
BIN_INSET_FRAC: float = 0.10

# Optional FIXED crop ROI (x, y, w, h), already inset. If set, it is used as THE
# crop by BOTH sam2_autolabel.py (labeling) and detect_case_obb.py (runtime) so
# train/run framing is identical. Leave None to compute per-run (see --roi).
# Tip: run sam2_autolabel.py --roi first, copy the printed ROI here to lock it.
BIN_ROI: tuple[int, int, int, int] | None = None

# ----------------------------------------------------------------------------
# YOLO-OBB case detector (learned backend, detect_case_obb.py).
# ----------------------------------------------------------------------------
# Trained weights — set after running train.py.
OBB_MODEL_PATH: str = "runs/detect/case/weight/case_detector.pt"
# Min detection confidence.
OBB_CONF: float = 0.40

# ----------------------------------------------------------------------------
# Learned BIN detector (detect_bin.py) — replaces HSV find_bin when the bin's
# image position varies. Regular axis-aligned YOLO detection on the full frame.
# ----------------------------------------------------------------------------
BIN_MODEL_PATH: str = "runs/detect/bin/weights/bin_detector.pt"
BIN_MODEL_CONF: float = 0.40
# When True, detect_case_obb crops using the learned bin model (else: cfg.BIN_ROI
# if set, else HSV find_bin). Priority at runtime: BIN_ROI > bin model > HSV.
USE_BIN_MODEL: bool = True

# ----------------------------------------------------------------------------
# SAM2 auto-labeling (sam2_autolabel.py) — runs offline on a GPU box, not at
# robot runtime. Set these to your downloaded checkpoint + its model cfg.
# ----------------------------------------------------------------------------
SAM2_CHECKPOINT: str = "checkpoints/sam2.1_hiera_small.pt"
SAM2_MODEL_CFG: str = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Label sanity thresholds (fraction of the crop area), used by review.py to
# flag likely-bad auto-labels (mask too small/large) and by obb_label.py.
LABEL_MIN_AREA_FRAC: float = 0.01
LABEL_MAX_AREA_FRAC: float = 0.95

# Auto-labeling output dirs (relative to this folder).
STAGED_DIR: str = "staged"     # numbered JPEG crops fed to the SAM2 video predictor
LABELED_DIR: str = "labeled"   # images/ + labels/ produced for training

# ----------------------------------------------------------------------------
# Paths (relative to this folder).
# ----------------------------------------------------------------------------
DATA_DIR: str = "data"
OUT_DIR: str = "out"
