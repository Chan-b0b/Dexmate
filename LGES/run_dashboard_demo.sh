#!/usr/bin/env bash
# Start the dashboard background services (camera, web server, detector).
# The demo is run separately in another terminal:
#
#   Terminal 1:  ./run_dashboard_demo.sh
#   Terminal 2:  cd LGES && python -m case_battery_demo.run_demo [--dashboard] [flags]
#
# Pass --dashboard to the demo if you want live joints/EE/wrench/camera in
# the viewer; omit it if you just want the robot to run without data spooling.
#
#   PORT=9090 SPOOL=/tmp/foo REVIEW_PORT=9091 RECORD_DIR=recordings ./run_dashboard_demo.sh
#
# Then open http://<robot-ip>:8080/ (live) and :8081/ (take review) in a browser.
# Ctrl-C here tears down the background services.
set -uo pipefail

cd "$(dirname "$0")"                       # LGES/ — package import root
SPOOL="${SPOOL:-/tmp/cns_dashboard}"
PORT="${PORT:-8080}"
REVIEW_PORT="${REVIEW_PORT:-8081}"
RECORD_DIR="${RECORD_DIR:-recordings}"     # must match the demo's --record-dir

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

# 3) Case-detection (BEV) overlay.
echo "[run_all] starting case detector…"
python -m case_battery_demo.dashboard.detector --spool "$SPOOL" &
pids+=($!)

# 4) Barcode reader image feed (IMAGE.SEND at 1 Hz — never triggers a read).
echo "[run_all] starting barcode image publisher…"
python -m case_battery_demo.dashboard.barcode --spool "$SPOOL" &
pids+=($!)

# 5) Recorded-take review dashboard (gallery + frame scrubber).
echo "[run_all] starting take review server on :${REVIEW_PORT} (root ${RECORD_DIR})…"
python -m case_battery_demo.dashboard.review_server \
  --root "$RECORD_DIR" --port "$REVIEW_PORT" &
pids+=($!)

echo "[run_all] Dashboard services running. Start the demo in another terminal:"
echo "  cd $(pwd) && python -m case_battery_demo.run_demo [--dashboard] [flags]"
echo "Ctrl-C to stop services."
wait
