#!/usr/bin/env bash
# 0909 dataset gate + FiLM recalibration (2026-09-09). Run after the rsync of
# lges_case_pick_0909{,_val} finishes, BEFORE launching the round.
#
# WHY IT RECALIBRATES: the 0909 collection remounted the F/T sensor, shifting the |F|
# baseline +5.6N (pre-touch 5.1 -> 10.7N). 0816's F0=6 now sits BELOW hover, so the
# contact channel clip((|F|-F0)/tau,0,1) reads ~0.8 while merely hovering and saturates
# at contact — no discrimination left. F0/TAU therefore cannot be inherited.
#
# WHY TOUCH-ANCHORED: the statistics MUST be measured just before and just after touch.
# Pooling "first 15 frames" as hover and "all sealed frames" as press mixes in the
# descent onset and the LOADED LIFT (film_contact.py's header: a held payload raises the
# baseline to ~16-24N). On 0909 that inflated press p95 to 18.1N and pushed tau to 9.0,
# and it faked a separability collapse (AUC 0.86 vs 0816's 0.93). Touch-anchored, the two
# rounds are equally separable (0.890 vs 0.899) and tau lands at 5.6.
#
# Touch is found by walking FORWARD from frame 0 to the seal onset for the first |F| above
# the episode's own baseline + 3*sigma; seal onset is the reliable landmark (0909 p10/50/90
# = 68/80/87) while the old |dF|>=2 rule fired on start-up noise (0909 median t0 = 7).
# Windows: pre = [t0-10, t0)  transient = [t0, t0+5)  settled = [t0+5, seal_onset).
#
# TRANSFER RULES (constants measured on 0816's touch windows; feeding 0816 back through
# them returns exactly the 6 / 4 / 1.40 / 5.00 that round trained with — the rules are the
# identity on the round that worked, not a free fit):
#   F0      = pre_med + GAP_POS*(settled_med - pre_med)     GAP_POS  = 0.6636
#   TAU     = (settled_p95 - F0) / ST_P95_C                 ST_P95_C = 0.5081
#   FZ_TAU  = (fz_settled_med - fz_pre_med) / FZ_SPAN       FZ_SPAN  = 0.3145
#   FZ_OFF  = fz_pre_med - FZ_PRE_C*FZ_TAU                  FZ_PRE_C = -0.1710
#   C1      = settled-distribution (contact p83, fz p16) — the battery's 'realistic
#             contact' c-hat, which is calibration-dependent and so moves with them.
#
# The press-sim / state-authority probes carry the SAME class of 0816-era hardcoded force
# constants, all of which the remount invalidated, so they are derived here too:
#   FZ_DELTA0 = fz jump at touch (settled fz median - pre-touch fz median). Default 1.7
#               came from an even earlier round; 0816's own value is 1.57, 0909's is 1.17.
#   F_BASE    = |F| at first touch (p=0); only the 'pattern' force model reads it.
#   F_CAP     = 1.3*max(episode contact peak) - pre-touch fz median. Applied to 0816 this
#               returns 24.9, i.e. exactly the 25.0 default — the rule is the identity on
#               the round the default was set for.
#   FC_FZ     = the 'first force rise' trigger for the pc_fc cell's swap wrench
#               (fz > FC_FZ for 2 frames). Kept at the same offset above the pre-touch fz
#               baseline that 0816's 3.0 default sat at (+2.46N). At the 3.0 default 0909's
#               trigger is true from frame 0 (fz baseline is 7.54N), so pc_fc pooled
#               early-descent frames and injected only |F|=8.6N — BELOW F0, i.e. c-hat=0.
#   DESCEND_N = |F| below which a frame counts as pre-contact. UNUSABLE on 0909: pre-touch
#               p95 (11.47N) overlaps settled press p50 (11.64N), so no force threshold
#               separates them. Emitted as 0 to mean "use the temporal --pre-contact filter
#               instead" — without it the ramp cells matched n=3 unrelated frames.
#
# Emits CAL_* lines that run_case_pick_0909_all.sh parses — never hardcode these.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maverick/vla_venv/bin/python
RT="$DIR/datasets/lges_case_pick_0909"
RV="$DIR/datasets/lges_case_pick_0909_val"
mkdir -p "$DIR/logs"

for R in "$RT" "$RV"; do
  [[ -f "$R/meta/info.json" ]] || { echo "[0909] $R/meta/info.json missing — rsync still running?" >&2; exit 1; }
done

"$PY" - <<EOF | tee "$DIR/logs/validate_0909.out"
import glob, json
import numpy as np, pandas as pd

GAP_POS, ST_P95_C, FZ_PRE_C, FZ_SPAN = 0.6636, 0.5081, -0.1710, 0.3145   # 0816 anchors

for tag, root in [("train", "$RT"), ("val", "$RV")]:
    info = json.load(open(f"{root}/meta/info.json"))
    f = info["features"]
    st_, act = list(f["observation.state"]["shape"]), list(f["action"]["shape"])
    cams = sorted(k for k in f if k.startswith("observation.images."))
    print(f"[0909] {tag}: eps={info['total_episodes']} frames={info['total_frames']} "
          f"state={st_} action={act} cams={cams}")
    assert st_ == [15], f"{tag} state {st_} != [15]"
    assert act == [7], f"{tag} action {act} != [7]"
    assert cams == ["observation.images.head", "observation.images.head_depth"], f"{tag} cams {cams}"

# ---- touch-anchored windows ------------------------------------------------------
pre, tr, st, prefz, stfz, cz = [], [], [], [], [], []
t0f, peaks = [], []
n_ep = n_ok = 0
for p in sorted(glob.glob("$RT/data/*/*.parquet")):
    df = pd.read_parquet(p, columns=["observation.state", "episode_index"])
    stt = np.stack(df["observation.state"].to_numpy())
    for ep in df["episode_index"].unique():
        s = stt[(df["episode_index"] == ep).to_numpy()]
        n_ep += 1
        fm = np.linalg.norm(s[:, 9:12], axis=1)
        js = np.flatnonzero(s[:, 8] > 0.5)
        if not len(js) or js[0] < 20: continue
        s0 = int(js[0])
        base, sig = np.median(fm[:10]), fm[:10].std() + 1e-6
        cand = np.flatnonzero(fm[:s0] > base + max(3.0 * sig, 1.0))
        if not len(cand): continue
        t0 = int(cand[0])
        if t0 - 10 < 0 or t0 + 5 >= s0: continue
        n_ok += 1
        pre.append(fm[t0-10:t0]); prefz.append(s[t0-10:t0, 11])
        tr.append(fm[t0:t0+5])
        st.append(fm[t0+5:s0]); stfz.append(s[t0+5:s0, 11])
        cz.append(s[t0, 2]); t0f.append(fm[t0]); peaks.append(fm[t0:s0].max())
pre, tr, st = np.concatenate(pre), np.concatenate(tr), np.concatenate(st)
prefz, stfz, cz = np.concatenate(prefz), np.concatenate(stfz), np.array(cz)

print(f"[0909] touch detected in {n_ok}/{n_ep} episodes "
      f"(pre {len(pre)}, transient {len(tr)}, settled {len(st)} frames)")
assert n_ok >= 0.9 * n_ep, f"touch detection only {n_ok}/{n_ep} — inspect the force traces"
for lbl, a in (("pre-touch", pre), ("transient", tr), ("settled", st)):
    print(f"[0909] |F| {lbl:9s} p5/25/50/75/95 = {np.percentile(a,[5,25,50,75,95]).round(2)}")
r = pd.Series(np.concatenate([pre, st])).rank().to_numpy(); n1, n2 = len(pre), len(st)
auc = (r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)
print(f"[0909] AUC(settled>pre-touch) = {auc:.4f}   (0816 touch-anchored: 0.8988)")
print(f"[0909] touch z p10/50/90 = {np.percentile(cz,[10,50,90]).round(3)}   (0816 same detector: [0.885 0.934 0.939])")
assert auc > 0.75, f"pre-touch vs settled not separable (AUC {auc:.3f}) — contact channel cannot work"

# ---- derive the calibration -------------------------------------------------------
pm, sm = np.median(pre), np.median(st)
F0 = pm + GAP_POS * (sm - pm)
TAU = (np.percentile(st, 95) - F0) / ST_P95_C
FZ_TAU = (np.median(stfz) - np.median(prefz)) / FZ_SPAN
FZ_OFF = np.median(prefz) - FZ_PRE_C * FZ_TAU
assert pm < F0 < sm, f"F0 {F0:.2f} outside (pre {pm:.2f}, settled {sm:.2f})"
assert TAU > 0 and FZ_TAU > 0, f"non-positive scale (TAU {TAU:.2f}, FZ_TAU {FZ_TAU:.2f})"
c = lambda a: np.clip((a - F0) / TAU, 0.0, 1.0)
cch, czch = c(st), (stfz - FZ_OFF) / FZ_TAU
C1 = f"{np.percentile(cch,83):.2f},{np.percentile(czch,16):.2f},0"
print(f"[0909] c-hat @ derived cal: pre_mean={c(pre).mean():.4f} settled_mean={cch.mean():.4f} "
      f"settled_p95={np.percentile(cch,95):.4f} transient_mean={c(tr).mean():.4f}")
print(f"[0909]   (0816 @ 6/4/1.40/5.00:  pre_mean=0.0129 settled_mean=0.1626 "
      f"settled_p95=0.5081 transient_mean=0.8794)")
print(f"CAL_F0={F0:.2f}")
print(f"CAL_TAU={TAU:.2f}")
print(f"CAL_FZ_OFF={FZ_OFF:.2f}")
print(f"CAL_FZ_TAU={FZ_TAU:.2f}")
FZ_DELTA0 = np.median(stfz) - np.median(prefz)
F_BASE = float(np.median(t0f))
ep_peak = np.array(peaks)
F_CAP = 1.3 * ep_peak.max() - np.median(prefz)
# a force-only pre-contact filter needs pre-touch p95 < settled p50; else it cannot work
DESCEND_N = 0.0 if np.percentile(pre, 95) >= np.median(st) else np.percentile(pre, 95)
print(f"[0909] probe params: fz_delta0={FZ_DELTA0:.2f} f_base={F_BASE:.2f} f_cap={F_CAP:.2f} "
      f"descend_n={DESCEND_N:.2f}")
print(f"[0909]   (0816 by the same rules: 1.57 / 8.27 / 24.90 / 6.26; the f_cap rule returns "
      f"the probe's own 25.0 default on 0816)")
if DESCEND_N == 0.0:
    print(f"[0909] NOTE pre-touch |F| p95={np.percentile(pre,95):.2f} >= settled p50={np.median(st):.2f}"
          f" -> force-threshold frame filter is unusable; probes MUST pass --pre-contact")
print(f"CAL_C1={C1}")
print(f"CAL_FZ_DELTA0={FZ_DELTA0:.2f}")
print(f"CAL_F_BASE={F_BASE:.2f}")
print(f"CAL_F_CAP={F_CAP:.2f}")
FC_FZ = np.median(prefz) + 2.46      # 0816: 0.54 + 2.46 = 3.00, the probe's own default
print(f"[0909]   fc_fz_thresh={FC_FZ:.2f} (0816 rule check: 0.54+2.46=3.00 = probe default)")
print(f"CAL_DESCEND_N={DESCEND_N:.2f}")
print(f"CAL_FC_FZ={FC_FZ:.2f}")
print(f"CAL_PRE_CONTACT=10")
print("[0909] validation OK")
EOF
[[ ${PIPESTATUS[0]} == 0 ]] || { echo "[0909] VALIDATION FAILED — do not train" >&2; exit 1; }
echo "[0909] calibration: $(grep -h '^CAL_' "$DIR/logs/validate_0909.out" | tr '\n' ' ')"
