#!/usr/bin/env python3
"""Merge multiple HF-hub LeRobotDatasets into one local dataset for combined training.

Downloads each source repo_id (train + its _val split) and aggregates them with
lerobot's built-in lerobot.datasets.aggregate.aggregate_datasets — validates fps/
robot_type/features match, then concatenates episodes/frames into one dataset with
re-indexed episode/frame/task indices and merged stats.json.

Run on the TRAINING server (needs lerobot installed), not on the recorder:
  python aggregate_datasets.py \
      --repo-ids Chanho-Lee/lges_case_pick_0721 Chanho-Lee/lges_case_pick_0727 \
      --name lges_case_pick_0721_0727

Then point training at the merged root, e.g.:
  DATASET_REPO=local/lges_case_pick_0721_0727 \
  DATASET_ROOT=datasets/lges_case_pick_0721_0727 \
  FILM_DATASET_ROOT=datasets/lges_case_pick_0721_0727 \
  ./train_film.sh --dataset.repo_id=local/lges_case_pick_0721_0727 \
      --dataset.root=datasets/lges_case_pick_0721_0727 ...
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.datasets.aggregate import aggregate_datasets

VLA_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = VLA_DIR / "datasets"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-ids", nargs="+", required=True,
                    help="HF hub dataset repo_ids to merge, e.g. Chanho-Lee/lges_case_pick_0721 "
                         "Chanho-Lee/lges_case_pick_0727 (their _val counterparts are merged too)")
    ap.add_argument("--name", required=True,
                    help="output dataset name, written to --out/<name> and <name>_val")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for suffix, out_suffix in [("", ""), ("_val", "_val")]:
        repo_ids = [f"{r}{suffix}" for r in args.repo_ids]
        roots = []
        for repo_id in repo_ids:
            root = args.out / repo_id.split("/")[-1]
            print(f"downloading {repo_id} -> {root}")
            snapshot_download(repo_id, repo_type="dataset", local_dir=root)
            roots.append(root)

        aggr_name = f"{args.name}{out_suffix}"
        aggr_root = args.out / aggr_name
        if aggr_root.exists():
            shutil.rmtree(aggr_root)
        print(f"aggregating {repo_ids} -> {aggr_root}")
        aggregate_datasets(repo_ids=repo_ids, aggr_repo_id=f"local/{aggr_name}",
                           roots=roots, aggr_root=aggr_root)
        print(f"  done -> {aggr_root}")
    print("all splits aggregated.")


if __name__ == "__main__":
    main()
