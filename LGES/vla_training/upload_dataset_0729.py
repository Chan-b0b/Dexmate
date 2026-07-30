#!/usr/bin/env python3
"""Upload the built lges_case_pick_0729 splits to the HF Hub.

Same single-worker strategy as upload_dataset_0727.py (this machine's slow
uplink starves parallel workers below the per-request timeout). Resumable —
re-run after any interruption and already-committed files are skipped.

Roots live in datasets_local/ (NOT datasets/ — that symlink points at the
GPU server's /data volume, which is not mounted on the recorder).

  python upload_dataset_0729.py
"""
import json
from pathlib import Path

from huggingface_hub import HfApi

VLA_DIR = Path(__file__).resolve().parent
SPLITS = ["lges_case_pick_0729", "lges_case_pick_0729_val"]
ORG = "Chanho-Lee"

api = HfApi()
for name in SPLITS:
    root = VLA_DIR / "datasets_local" / name
    repo_id = f"{ORG}/{name}"
    version = json.loads((root / "meta" / "info.json").read_text())["codebase_version"]
    print(f"== {repo_id}  ({root}, codebase {version}) ==")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=root, repo_type="dataset",
                            num_workers=1)
    # LeRobotDataset refuses to load a hub repo without its codebase-version tag.
    api.create_tag(repo_id, tag=version, repo_type="dataset", exist_ok=True)
    print(f"  done + tagged {version} -> https://huggingface.co/datasets/{repo_id}")
print("all splits uploaded.")
