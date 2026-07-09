#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Upload a trained lerobot checkpoint (smolvla or smolvla_meanflow) to the HF Hub.

Uploads the checkpoint's pretrained_model folder as-is (config.json, model.safetensors,
pre/post-processor files, train_config.json), so the robot can load it with
`from_pretrained("<repo_id>")` exactly like lerobot/smolvla_base.

Auth: run `hf auth login` once, or pass HF_TOKEN in the environment.
NOTE: on this server use HF_HOME=$HOME/.cache/huggingface (the /data token is unreadable).

  python push_checkpoint.py --run smolvla_meanflow_0708                  # -> Chanho-Lee/smolvla_meanflow_0708
  python push_checkpoint.py --run smolvla_meanflow_0708 --checkpoint 020000
  python push_checkpoint.py --path <any pretrained_model dir> --repo user/name
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

VLA_OUTPUTS = Path.home() / "Dexmate/LGES/vla_training/outputs"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run name under vla_training/outputs/")
    ap.add_argument("--checkpoint", default="last", help="checkpoint step dir (default: last)")
    ap.add_argument("--path", type=Path, help="explicit pretrained_model dir (overrides --run)")
    ap.add_argument("--repo", help="target repo id (default: Chanho-Lee/<run>)")
    ap.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    args = ap.parse_args()

    if args.path:
        model_dir = args.path
        default_name = model_dir.parent.parent.parent.name  # outputs/<run>/checkpoints/<ckpt>/pretrained_model
    elif args.run:
        model_dir = VLA_OUTPUTS / args.run / "checkpoints" / args.checkpoint / "pretrained_model"
        default_name = args.run
    else:
        raise SystemExit("pass --run or --path")

    if not (model_dir / "model.safetensors").exists():
        raise SystemExit(f"no model.safetensors in {model_dir}")

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = args.repo or f"{user}/{default_name}"

    print(f"uploading {model_dir}  ->  {repo_id} (private={not args.public})")
    api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=f"upload {default_name} ({args.checkpoint})",
    )
    print(f"done: https://huggingface.co/{repo_id}")
    print(f"\nOn the robot:")
    print(f"  policy = SmolVLAMeanFlowPolicy.from_pretrained('{repo_id}')   # needs `import smolvla_meanflow`")


if __name__ == "__main__":
    main()
