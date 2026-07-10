#!/usr/bin/env python3
"""Train a FiLM contact-conditioned SmolVLA (V1/V2): apply the film_contact patches,
then run lerobot's standard training loop unchanged.

  FILM_VARIANT=v2 python train_film.py --policy.path=<init ckpt> --dataset.root=... ...

Variant + dataset (for wrench-normalization stats) come from env:
  FILM_VARIANT       v2 (method) | v1 (decorrelated control)   [default v2]
  FILM_DATASET_ROOT  lerobot dataset dir with meta/stats.json  [default datasets/lges_suction]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_contact  # noqa: E402

_variant = os.environ.get("FILM_VARIANT", "v2")
_mask_force = os.environ.get("FILM_MASK_FORCE", "0") not in ("0", "false", "False")
_cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
_inject = os.environ.get("FILM_INJECT", "suffix")
_f0 = float(os.environ.get("FILM_F0", "12"))
_tau = float(os.environ.get("FILM_TAU", "10"))       # contact-DROP scale (graded)
_fz_tau = float(os.environ.get("FILM_FZ_TAU", "30"))  # continuous fz scale (fz_raw/30)
_root = os.environ.get(
    "FILM_DATASET_ROOT", str(Path(__file__).resolve().parent / "datasets/lges_suction"))
_wm, _ws = film_contact.load_wrench_stats(_root)
_sm, _ss = film_contact.load_seal_stats(_root)
film_contact.apply(_variant, _wm, _ws, seal_mean=_sm, seal_std=_ss, cond=_cond,
                   contact_F0=_f0, contact_tau=_tau, fz_tau=_fz_tau,
                   mask_force=_mask_force, inject=_inject)
print(f"[film] patched VLAFlowMatching: variant={_variant} cond={_cond} inject={_inject} "
      f"mask_force={_mask_force} contact=clip(({_f0:.0f}-|F|)/{_tau:.0f}) fz=fz/{_fz_tau:.0f} "
      f"wrench_mean={[round(x,2) for x in _wm.tolist()]}", file=sys.stderr)

from lerobot.scripts.lerobot_train import train  # noqa: E402

if __name__ == "__main__":
    train()  # @parser.wrap() parses sys.argv — same CLI as lerobot-train


#FILM_VARIANT=v2 RUN_NAME=smoke ./train_film.sh --steps=4 --save_freq=2

# # 둘 다 (현재 기본)
# FILM_VARIANT=v2 RUN_NAME=film_v2 ./train_film.sh
# # contact만
# FILM_COND=contact FILM_VARIANT=v2 RUN_NAME=film_v2_contact ./train_film.sh
# # seal만
# FILM_COND=seal    FILM_VARIANT=v2 RUN_NAME=film_v2_seal    ./train_film.sh

# # 평가 — 반드시 학습과 같은 FILM_COND (+ FILM_MASK_FORCE)
# FILM_COND=contact,seal FILM_MASK_FORCE=0 python run_policy.py --go --task case_pick \
#   --goto-start <take> --checkpoint outputs/film_v2/checkpoints/last --film ... --loop