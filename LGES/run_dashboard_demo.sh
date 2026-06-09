#!/usr/bin/env bash
# One-shot launcher for the case + battery dashboard demo.
#
# Brings up, in order:
#   1. the head-camera dexsensor on the nano (over SSH, best-effort)
#   2. the dashboard web server          (background)
#   3. the bin-detection overlay         (background)
#   4. the demo itself, with --dashboard (foreground — interactive)
#
# The demo runs in the foreground so you can answer its safety / E-stop
# prompts and watch its logs; Ctrl-C (or the demo finishing) tears the
# background services back down. Run it from anywhere — it cd's to its own
# directory (LGES/) so `python -m case_battery_demo…` resolves.
#
#   ./run_dashboard_demo.sh                 # forward only
#   ./run_dashboard_demo.sh --loop          # pass extra demo flags through
#   PORT=9090 SPOOL=/tmp/foo ./run_dashboard_demo.sh
#
# Then open http://<robot-ip>:8080/ in a browser.
set -uo pipefail

cd "$(dirname "$0")"                       # LGES/ — package import root
SPOOL="${SPOOL:-/tmp/cns_dashboard}"
PORT="${PORT:-8080}"

pids=()
cleanup() {
  echo
  echo "[run_all] shutting down dashboard services…"
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) Head camera on the nano (idempotent; non-fatal if the nano is unreachable).
echo "[run_all] ensuring head camera on the nano…"
python -m case_battery_demo.dashboard.camera_launch \
  || echo "[run_all] camera not confirmed — continuing (image may stay blank)."

# 2) Dashboard web server (camera already handled above, so skip its launch).
echo "[run_all] starting dashboard server on :${PORT} (spool ${SPOOL})…"
python -m case_battery_demo.dashboard.server \
  --no-launch-camera --spool "$SPOOL" --port "$PORT" &
pids+=($!)

# 3) Bin-detection overlay.
echo "[run_all] starting bin detector…"
python -m case_battery_demo.dashboard.detector --spool "$SPOOL" &
pids+=($!)

# 4) The demo, in the foreground. Any extra args ("$@") pass straight through
#    (e.g. --loop, --undo, --skip-confirmation).
echo "[run_all] starting demo — answer the safety prompt below (Ctrl-C to stop all)."
python -m case_battery_demo.run_demo --dashboard "$@"
