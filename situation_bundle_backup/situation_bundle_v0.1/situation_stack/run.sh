#!/usr/bin/env bash
# situation_bundle: optional Zenoh→ROS2 bridge + manipulation manager (default).
# Bundle root = parent of this directory.
#
# Bridge:  export ROBOT_PREFIX=dm/vg144e604acd-1p
# Skip bridge only: SITUATION_SKIP_ZENOH_ROS_BRIDGE=1 (manipulation_manager still runs)
# Skip manipulation: SITUATION_SKIP_MANIPULATION_BRIDGE=1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "$HERE/.." && pwd)"
ROS_SETUP="${SITUATION_ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  # ROS setup.bash references vars that may be unset; nounset breaks it.
  set +u
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  set -u
else
  echo "[situation_stack] WARN: ROS setup not found at $ROS_SETUP" >&2
fi
export PYTHONPATH="${BUNDLE}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${DEXCONTROL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${DEXCONTROL_PYTHONPATH}:${PYTHONPATH}"
fi
exec python3 "$HERE/main.py" "$@"
