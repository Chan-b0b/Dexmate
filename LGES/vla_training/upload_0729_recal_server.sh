#!/usr/bin/env bash
# Run on the SERVER (maverick). Uploads the 0729 recal-round checkpoints to HF —
# they are NOT on the Hub yet (verified 404 on 2026-08-04) and the robot-side eval
# commands (robot_eval_0729_recal.sh) pull from HF.
# Layout per the 2026-07-30 policy: main = val-best, branch 'last' = final step.
set -euo pipefail
~/vla_venv/bin/python - <<'EOF'
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
VLA = Path("/home/maverick/Dexmate/LGES/vla_training")
RUNS = {
    "smolvla_film_0729_prefix_mask1_recal_fromnaive":
        "FiLM v2 cond=contact,fmag,fz,seal inject=prefix mask1, recal calib "
        "(F0=5.5/tau=1, fmag=5.5/1, fz=(fz-3.0)/0.7), warm-start from naive_0729 best, 20k",
    "smolvla_film_0729_prefix_mask1_recal":
        "FiLM v2 cond=contact,fmag,fz,seal inject=prefix mask1, recal calib, from base, 50k",
}
DEPLOY_ENV = ("FILM_COND=contact,fmag,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 "
              "FILM_F0=5.5 FILM_TAU=1 FILM_FMAG_OFF=5.5 FILM_FMAG_TAU=1 "
              "FILM_FZ_OFF=3.0 FILM_FZ_TAU=0.7")

for run, setting in RUNS.items():
    repo = f"Chanho-Lee/{run}"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    for rev, ck in ((None, "best"), ("last", "last")):
        src = VLA / "outputs" / run / "checkpoints" / ck / "pretrained_model"
        assert src.is_dir(), f"missing {src}"
        if rev:
            api.create_branch(repo, branch=rev, exist_ok=True)
        api.upload_folder(folder_path=str(src), repo_id=repo, repo_type="model",
                          revision=rev, commit_message=f"upload {run} '{ck}'")
    card = (f"# {run}\n\nSmolVLA (lerobot 0.5.1) checkpoint, LGES case_pick demo.\n\n"
            f"- setting: {setting}\n"
            f"- dataset: Chanho-Lee/lges_case_pick_0729 (val: _0729_val)\n"
            f"- main = val-best, branch `last` = final step\n\n"
            f"Deploy env (calibration buffers are persistent=False — env is the only "
            f"source):\n```\n{DEPLOY_ENV}\n```\n")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model", commit_message="model card")
    print("[upload] OK", repo)
EOF
