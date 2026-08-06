#!/usr/bin/env python3
"""Upload each run's pretrained_model to HF hub as Chanho-Lee/<run_name>,
with a small model card recording the training setting. Usage:
  python upload_weights.py <run_name> [<run_name> ...]              # val-best -> main
  python upload_weights.py --with-last <run_name> ...               # + last -> 'last' branch
  python upload_weights.py --last-only <run_name> ...               # only the 'last' branch

Branch convention (one repo per run, two revisions):
  main   val-best checkpoint (2026-07-30 selection policy) — the deploy pick
  last   final-step checkpoint
Load a specific one with the `revision` argument, which PreTrainedPolicy.from_pretrained
forwards to the hub:
  PI05Policy.from_pretrained("Chanho-Lee/pi05_naive_0729", revision="last")
This is why `last` is a branch and not a subfolder or a <run>_last repo: from_pretrained
takes `revision` but has no `subfolder` argument.
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
    "smolvla_naive_0721_0727":                  ("Chanho-Lee/lges_case_pick_0721_0727", "vanilla SmolVLA finetune, new robot, rel, 50k"),
    "smolvla_film_0721_0727_prefix_mask1":      ("Chanho-Lee/lges_case_pick_0721_0727", "FiLM v2, cond=contact,fz,seal inject=prefix mask_force=1 F0=6 tau=4 fz=(fz-2.6)/5, 50k"),
    "smolvla_film_0721_0727_prefix_mask1_os3":  ("Chanho-Lee/lges_case_pick_0721_0727", "FiLM v2, prefix mask1 + transition oversampling x3, 50k"),
    "smolvla_film_0721_0727_suffix_mask1":      ("Chanho-Lee/lges_case_pick_0721_0727", "FiLM v2, cond=contact,fz,seal inject=suffix mask_force=1, 50k"),
    "xvla_0721_0727":                           ("Chanho-Lee/lges_case_pick_0721_0727", "X-VLA (2toINF/X-VLA-Pt init via train_xvla.py shims), 224x224 pad, 50k"),
    "pi05_naive_0721_0727":                     ("Chanho-Lee/lges_case_pick_0721_0727", "pi0.5 (lerobot/pi05_base) finetune, bs8 grad-ckpt, 50k"),
    "act_0721_0727":                            ("Chanho-Lee/lges_case_pick_0721_0727", "ACT from scratch, bs32, 50k"),
    "smolvla_film_0721_0727_prefix_mask1_cs": ("Chanho-Lee/lges_case_pick_0721_0727", "FiLM v2, cond=contact,seal (no fz) inject=prefix mask_force=1, 50k"),
    "smolvla_naive_0729":               ("Chanho-Lee/lges_case_pick_0729", "vanilla SmolVLA, 50k, val-best selected"),
    "smolvla_film_0729_prefix_mask1":   ("Chanho-Lee/lges_case_pick_0729", "FiLM v2 cond=contact,fz,seal prefix mask1 FZ_OFF=2.1, 50k, val-best"),
    "smolvla_film_0729_prefix_mask0":   ("Chanho-Lee/lges_case_pick_0729", "FiLM v2 cond=contact,fz,seal prefix mask0 FZ_OFF=2.1 (ablation), 50k, val-best"),
    "smolvla_film_0729_suffix_mask1":   ("Chanho-Lee/lges_case_pick_0729", "FiLM v2 cond=contact,fz,seal suffix mask1 FZ_OFF=2.1, 50k, val-best"),
    "act_0729":                         ("Chanho-Lee/lges_case_pick_0729", "ACT from scratch, 50k, val-best"),
    "smolvla_film_0729_prefix_mask1_recal": ("Chanho-Lee/lges_case_pick_0729",
        "FiLM v2 cond=contact,fmag,fz,seal prefix mask1, RECAL calib F0=5.5 tau=1 "
        "fmag=(|F|-5.5)/1 fz=(fz-3.0)/0.7, 50k from base, val-best@5k. "
        "DEPLOY: override FILM_F0 = field hover baseline + 1.5 (~7) — real baseline "
        "drifted to 5.5-5.8N vs train 4.6N (rollout analysis 2026-08-03)"),
    "smolvla_film_0729_prefix_mask0_recal": ("Chanho-Lee/lges_case_pick_0729",
        "FiLM v2 cond=contact,fmag,fz,seal prefix mask0 (ablation — FiLM stays inert), "
        "RECAL calib, 50k from base, val-best@5k"),
    "smolvla_film_0729_prefix_mask1_recal_fromnaive": ("Chanho-Lee/lges_case_pick_0729",
        "FiLM v2 cond=contact,fmag,fz,seal prefix mask1, RECAL calib F0=5.5 tau=1 "
        "fmag=(|F|-5.5)/1 fz=(fz-3.0)/0.7, warm-start from smolvla_naive_0729 best@10k, "
        "20k steps, val-best@2500. Offline deploy pick 2026-08-04 (stops at 6.8N "
        "first-contact + monotone force ramp; press-sim 0.6mm nominal / 3.4mm early). "
        "DEPLOY: override FILM_F0 = field hover baseline + 1.5 (~7)"),
    "smolvla_film_0729_prefix_mask1_recal_fromnaive_v1": ("Chanho-Lee/lges_case_pick_0729",
        "FiLM V1 = DECORRELATED CONTROL for the fromnaive run (c-hat shuffled across the "
        "batch at train time: same capacity and mechanism as v2, grounding removed), "
        "cond=contact,fmag,fz,seal prefix mask1, RECAL calib F0=5.5 tau=1 "
        "fmag=(|F|-5.5)/1 fz=(fz-3.0)/0.7, warm-start from smolvla_naive_0729 best@10k, "
        "20k steps, val-best@5000 (val loss 0.1715; monotone rise after, same overfitting "
        "shape as the rest of the 0729 round)"),
    "smolvla_film_0729_prefix_mask0_recal_fromnaive": ("Chanho-Lee/lges_case_pick_0729",
        "FiLM v2 cond=contact,fmag,fz,seal prefix mask0 — NEGATIVE CONTROL for the fromnaive "
        "run (raw wrench left in the state, so the policy can read force directly and FiLM "
        "stays inert), RECAL calib F0=5.5 tau=1 fmag=(|F|-5.5)/1 fz=(fz-3.0)/0.7, "
        "warm-start from smolvla_naive_0729 best@10k, 20k steps, val-best"),
    # 0729 pi0.5 round (B300 box, 2026-08-05/06). film_contact_pi05 injects at SUFFIX only
    # (action-token embeddings at the expert input, all layers) — there is no prefix variant.
    "pi05_naive_0729":            ("Chanho-Lee/lges_case_pick_0729",
        "pi0.5 (lerobot/pi05_base) finetune, bs8 grad-ckpt, 50k, val-best@10k (val 0.0313; "
        "rises monotonically to 0.0404 at 50k — 10k is the earliest eval, so the optimum "
        "may be earlier)"),
    "pi05_film_frombase_0729":    ("Chanho-Lee/lges_case_pick_0729",
        "pi0.5 FiLM v2 from lerobot/pi05_base, cond=contact,fz,seal inject=suffix "
        "mask_force=1 FZ_OFF=2.1, bs8 grad-ckpt, 50k, val-best@10k (val 0.0611)"),
    "pi05_film_onnaive_0729":     ("Chanho-Lee/lges_case_pick_0729",
        "pi0.5 FiLM v2 warm-started from pi05_naive_0729 best@10k, cond=contact,fz,seal "
        "inject=suffix mask_force=1 FZ_OFF=2.1, bs8 grad-ckpt, 50k, val-best@10k"),
}

api = HfApi()
FLAGS = {a for a in sys.argv[1:] if a.startswith("--")}
RUNS = [a for a in sys.argv[1:] if not a.startswith("--")]
if FLAGS - {"--with-last", "--last-only"}:
    sys.exit(f"unknown flag(s): {' '.join(sorted(FLAGS - {'--with-last', '--last-only'}))}")


def family(run):
    """Card wording per model family — the 0729 round is no longer SmolVLA-only."""
    if run.startswith("pi05"):
        return "pi0.5", "lerobot/pi05_base"
    if run.startswith("xvla"):
        return "X-VLA", "2toINF/X-VLA-Pt"
    if run.startswith("act"):
        return "ACT", None                      # from scratch
    return "SmolVLA", "lerobot/smolvla_base"


def push(run, which, branch):
    """which: 'best'|'last' checkpoint dir; branch: 'main'|'last' revision to push it to."""
    repo = f"Chanho-Lee/{run}"
    ck = VLA / "outputs" / run / "checkpoints"
    src = ck / which / "pretrained_model"
    if not src.is_dir():
        print(f"[upload] SKIP {repo}@{branch}: {src} not found", flush=True); return
    step = (ck / which).resolve().name
    model, base = family(run)
    ds, setting = META.get(run, ("?", "?"))
    print(f"[upload] {repo}@{branch}: '{which}' checkpoint ({step})", flush=True)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    if branch != "main":
        api.create_branch(repo, branch=branch, repo_type="model", exist_ok=True)
    card = (f"# {run}\n\n{model} (lerobot 0.5.1) checkpoint for the LGES "
            f"case_pick demo.\n\n"
            f"- setting: {setting}\n"
            f"- checkpoint: {step} ({'val-best' if which == 'best' else 'final step'})\n"
            f"- dataset: [{ds}](https://huggingface.co/datasets/{ds})\n"
            + (f"- base: [{base}](https://huggingface.co/{base})\n" if base
               else "- base: trained from scratch\n")
            + "\nFiLM checkpoints must be loaded with `film_contact.apply()` (SmolVLA) or "
              "`film_contact_pi05.apply()` (pi0.5) using the SAME cond/inject/mask_force as "
              "in the setting line (see LGES/vla_training).\n")
    api.upload_folder(folder_path=str(src), repo_id=repo, repo_type="model",
                      revision=branch, commit_message=f"upload {run} {which}@{step}")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model", revision=branch,
                    commit_message="model card")
    print(f"[upload] OK {repo}@{branch}", flush=True)


for run in RUNS:
    ck = VLA / "outputs" / run / "checkpoints"
    has_best = (ck / "best").exists()   # val-best if selected (2026-07-30 policy)
    if "--last-only" not in FLAGS:
        push(run, "best" if has_best else "last", "main")
    if FLAGS & {"--with-last", "--last-only"} and has_best:
        if (ck / "best").resolve() == (ck / "last").resolve():
            print(f"[upload] {run}: best == last, skipping the 'last' branch", flush=True)
        else:
            push(run, "last", "last")
print("[upload] done")
