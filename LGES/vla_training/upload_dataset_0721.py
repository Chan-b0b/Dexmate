#!/usr/bin/env python3
"""Upload the already-built lges_case_pick_0721 splits to the HF Hub.

Standalone replacement for convert_to_lerobot.py --push on THIS machine's slow
uplink (~250 KB/s to HF's CDN): push_to_hub's default 7 parallel workers starve
each other below the per-request timeout and retry forever (observed: 0/60
files after 2.5 h). One worker gets the full bandwidth, so every request
finishes and progress is monotonic. Resumable — re-run after any interruption
and already-committed files are skipped.

  /home/dexmate/miniconda3/bin/python upload_dataset_0721.py
"""
import json
from pathlib import Path

from huggingface_hub import HfApi

VLA_DIR = Path(__file__).resolve().parent
SPLITS = ["lges_case_pick_0721", "lges_case_pick_0721_val"]
ORG = "Chanho-Lee"

api = HfApi()
for name in SPLITS:
    root = VLA_DIR / "datasets" / name
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
