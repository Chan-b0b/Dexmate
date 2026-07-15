#!/usr/bin/env bash
# Copy only the files needed to run ./run.sh (Zenoh → ROS2 bridge launcher).
# Usage: ./copy_minimal_runtime.sh [DEST_DIR]
# Default DEST: ../situation_bundle_runtime (next to this folder)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-${ROOT}/../situation_bundle_runtime}"

# Paths relative to situation_bundle/ — keep in sync with situation_stack/main.py
# (bridge + manipulation_manager)
FILES=(
  run.sh
  situation_stack/run.sh
  situation_stack/main.py
  bridge/zenoh_robot_state_to_ros2.py
  manipulation_bridge/manipulation_manager.py
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp -a "$ROOT/$f" "$DEST/$f"
done
chmod +x "$DEST/run.sh" "$DEST/situation_stack/run.sh"

echo "Copied ${#FILES[@]} paths → $DEST"
echo "Run: cd \"$DEST\" && ./run.sh"
