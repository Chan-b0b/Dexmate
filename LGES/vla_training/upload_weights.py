#!/usr/bin/env python3
"""Upload each run's last pretrained_model to HF hub as Chanho-Lee/<run_name>,
with a small model card recording the training setting. Usage:
  python upload_weights.py <run_name> [<run_name> ...]
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

VLA = Path(__file__).resolve().parent
# run -> (dataset repo, one-line setting)
META = {
    "smolvla_naive_0708":          ("Chanho-Lee/lges_case_pick_0708",     "vanilla SmolVLA finetune, 30k steps"),
    "smolvla_naive_0708_abs":      ("Chanho-Lee/lges_case_pick_0708_abs", "vanilla SmolVLA finetune, 30k steps"),
    "smolvla_film_0708":           ("Chanho-Lee/lges_case_pick_0708",     "FiLM v2 from base, cond=contact,fz,seal inject=suffix mask_force=0, 30k"),
    "smolvla_film_0708_abs":       ("Chanho-Lee/lges_case_pick_0708_abs", "FiLM v2 from base, cond=contact,fz,seal inject=suffix mask_force=0, 30k"),
    "smolvla_film_0708_prefix":    ("Chanho-Lee/lges_case_pick_0708",     "FiLM v2 from base, cond=contact,fz,seal inject=prefix mask_force=0, 30k"),
    "smolvla_film_0708_abs_prefix":("Chanho-Lee/lges_case_pick_0708_abs", "FiLM v2 from base, cond=contact,fz,seal inject=prefix mask_force=0, 30k"),
    "film_on_naive_0708":          ("Chanho-Lee/lges_case_pick_0708",     "FiLM v2 from naive-30k ckpt, inject=suffix mask_force=0, 10k"),
    "film_on_naive_0708_abs":      ("Chanho-Lee/lges_case_pick_0708_abs", "FiLM v2 from naive-30k ckpt, inject=suffix mask_force=0, 10k"),
    "smolvla_film_0708_mask1":     ("Chanho-Lee/lges_case_pick_0708",     "FiLM v2 from base, cond=contact,fz,seal inject=suffix mask_force=1, 30k"),
    "smolvla_film_0708_abs_mask1": ("Chanho-Lee/lges_case_pick_0708_abs", "FiLM v2 from base, cond=contact,fz,seal inject=suffix mask_force=1, 30k"),
    "smolvla_film_0708_prefix_mask1":     ("Chanho-Lee/lges_case_pick_0708",     "FiLM v2 from base, cond=contact,fz,seal inject=prefix mask_force=1, 30k"),
    "smolvla_film_0708_abs_prefix_mask1": ("Chanho-Lee/lges_case_pick_0708_abs", "FiLM v2 from base, cond=contact,fz,seal inject=prefix mask_force=1, 30k"),
    "smolvla_film_0708_dF_prefix_mask1":     ("Chanho-Lee/lges_case_pick_0708_dF",     "FiLM v2 from base, cond=contact,fz,seal,dfmag inject=prefix mask_force=1 dfmag_tau=5, 30k, state 16"),
    "smolvla_film_0708_abs_dF_prefix_mask1": ("Chanho-Lee/lges_case_pick_0708_abs_dF", "FiLM v2 from base, cond=contact,fz,seal,dfmag inject=prefix mask_force=1 dfmag_tau=5, 30k, state 16"),
    "smolvla_film_0708_dF_prefix_mask1_os10":     ("Chanho-Lee/lges_case_pick_0708_dF",     "FiLM v2 from base, cond=contact,fz,seal,dfmag inject=prefix mask_force=1, contact-transition oversampling x10, 30k, state 16"),
    "smolvla_film_0708_abs_dF_prefix_mask1_os10": ("Chanho-Lee/lges_case_pick_0708_abs_dF", "FiLM v2 from base, cond=contact,fz,seal,dfmag inject=prefix mask_force=1, contact-transition oversampling x10, 30k, state 16"),
    "smolvla_naive_0721":                    ("Chanho-Lee/lges_case_pick_0721",    "vanilla SmolVLA finetune, NEW robot, rel actions, 30k"),
    "smolvla_film_0721_prefix_mask1":        ("Chanho-Lee/lges_case_pick_0721",    "FiLM v2 from base, cond=contact,fz,seal inject=prefix mask_force=1 F0=6 tau=4 fz_tau=5 (new-robot calib), 30k"),
    "smolvla_film_0721_dF_prefix_mask1":     ("Chanho-Lee/lges_case_pick_0721_dF", "FiLM v2 from base, cond=contact,fz,seal,dfmag inject=prefix mask_force=1 F0=6 tau=4 fz_tau=5, 30k, state 16"),
    "smolvla_film_0721_dF_prefix_mask1_os3": ("Chanho-Lee/lges_case_pick_0721_dF", "FiLM v2 from base, cond=contact,fz,seal,dfmag inject=prefix mask_force=1 F0=6 tau=4 fz_tau=5, transition oversampling x3, 30k, state 16"),
    "smolvla_film_0721_prefix_mask1_os3": ("Chanho-Lee/lges_case_pick_0721", "FiLM v2 from base, cond=contact,fz,seal inject=prefix mask_force=1 F0=6 tau=4 fz_tau=5, transition oversampling x3, 30k"),
}

api = HfApi()
for run in sys.argv[1:]:
    src = VLA / "outputs" / run / "checkpoints" / "last" / "pretrained_model"
    if not src.is_dir():
        print(f"[upload] SKIP {run}: {src} not found"); continue
    ds, setting = META.get(run, ("?", "?"))
    repo = f"Chanho-Lee/{run}"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    card = (f"# {run}\n\nSmolVLA (lerobot 0.5.1) checkpoint for the LGES case_pick demo.\n\n"
            f"- setting: {setting}\n- dataset: [{ds}](https://huggingface.co/datasets/{ds})\n"
            f"- base: [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)\n\n"
            "FiLM checkpoints must be loaded with `film_contact.apply()` using the SAME "
            "cond/inject/mask_force as in the setting line (see LGES/vla_training).\n")
    api.upload_folder(folder_path=str(src), repo_id=repo, repo_type="model",
                      commit_message=f"upload {run} ({setting})")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model", commit_message="model card")
    print(f"[upload] OK {repo}")
print("[upload] done")
