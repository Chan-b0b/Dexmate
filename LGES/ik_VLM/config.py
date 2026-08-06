"""Configuration for the ik_VLM supervised-execution layer.

All thresholds/paths for the signal tap, envelope monitor, Tier-0/1 recovery,
and the Tier-2 local-VLM advisor live here. ik_demo's own config is untouched
(imported read-only where geometry is needed).
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Signal tap (signals.py) — background sampling of the tared wrench.
#
# Features are computed in the SENSOR frame (no FK from the tap thread — the
# pinocchio model is not thread-safe against the main IK stream):
#   f_ax   |tared force along the tube axis| (sensor z), N
#   f_lat  hypot of the tared lateral force (sensor xy), N — magnitude, so
#          wrist-yaw invariant
#   t_mag  |tared torque|, Nm
#   df_mag |d(force vector)/dt|, N/s — rate-normalized so 10-15 Hz recordings
#          and the 50 Hz live tap share one envelope
# ---------------------------------------------------------------------------
SIGNAL_HZ: float = 50.0
SIGNAL_BUFFER_S: float = 10.0            # ring buffer depth (trip context / logging)
# Per-run signal log (jsonl, one tick per line) — the preferred envelope-build
# source (exact phases, live rate). None disables.
SIGNAL_LOG_DIR: str | None = str(_HERE / "logs")
# A descent tick (tick_cb) marks the phase "descend"; it decays back to the
# enclosing phase when no descent tick has arrived for this long.
DESCEND_PHASE_DECAY_S: float = 0.3

# ---------------------------------------------------------------------------
# Envelope monitor (monitor.py) — open-set: per-(phase, feature) mean + k*sigma
# UPPER bands from nominal runs only. Anomaly = signal too large; "expected
# contact never came" is already covered by the script's max_descent.
# ---------------------------------------------------------------------------
ENVELOPE_PATH: str = str(_HERE / "envelope.json")
ENVELOPE_K_SIGMA: float = 4.0
# Force features are heavy-tailed (contact/seal impulses), so mean+k*sigma
# under-covers the nominal tail: the bound is raised to at least the MAXIMUM
# ever observed in nominal data times this margin ("never seen in nominal").
ENVELOPE_QMAX_MARGIN: float = 1.2
TRIGGER_CONSECUTIVE: int = 5             # consecutive out-of-band ticks to trip
# Absolute band floors (added to the mean) so a tiny nominal sigma can't make
# a band hair-trigger. Units match the feature.
ENVELOPE_MIN_BAND: dict[str, float] = {
    "f_ax": 3.0,      # N   — well under FORCE_CONTACT_THRESHOLD_N (10 N)
    "f_lat": 2.5,     # N   — under the recovery's own 3 N force-guide threshold
    "t_mag": 0.4,     # Nm
    "df_mag": 40.0,   # N/s — a 10 N step in ~0.25 s
    "q_err_max": 0.05,  # rad — streamed IK tracks within ~0.01-0.02 nominally;
                        # the live qmax margin sets the real bound. Recordings
                        # lack this signal, so a recordings-built envelope has
                        # no q_err_max bound and the monitor skips the feature.
}
# Phases the monitor is ARMED in. Contact-adjacent phases (creep-seal, misseat
# probing) are spiky by design and stay unarmed in v1.
MONITORED_PHASES: tuple[str, ...] = ("descend", "transport", "place", "pick")

# ---------------------------------------------------------------------------
# Tier 0 — safe hold (recovery.py). The in-loop halt already froze the arm;
# this lifts a small, always-safe amount off whatever was being touched.
# Automatic; everything after it is operator-approved.
# ---------------------------------------------------------------------------
HOLD_LIFT_M: float = 0.03

# ---------------------------------------------------------------------------
# Tier 1 — re-entry (world_state.py / resume_matrix.py)
# ---------------------------------------------------------------------------
SEAL_CHECK_TIMEOUT_S: float = 2.0        # one-shot DI0 read window
# A re-detected case farther than this from where the script expected it is
# "unexpected position" -> Tier 2 (dropped / shifted part, disturbed stack).
CASE_XY_UNEXPECTED_M: float = 0.10
REDETECT_TRIES: int = 2                  # fresh detections before "not found"

# ---------------------------------------------------------------------------
# Tier 2 — local VLM advisor (vlm_advisor.py). Any OpenAI-compatible
# /chat/completions endpoint with vision (Ollama, llama.cpp server, vLLM).
# The advisor only PROPOSES a skill from ALLOWED_SKILLS; execution is gated
# behind the operator. Endpoint unreachable / unparseable -> call_operator.
# ---------------------------------------------------------------------------
VLM_BASE_URL: str = "http://localhost:11434/v1"
VLM_MODEL: str = "qwen2.5vl:7b"
VLM_TIMEOUT_S: float = 120.0
VLM_MAX_TOKENS: int = 512
VLM_IMAGE_MAX_W: int = 960               # downscale the head frame before sending
ALLOWED_SKILLS: tuple[str, ...] = (
    "hold",               # stay put (part held), wait for the operator
    "lift_to_transport",  # lift to SAFE_TRANSPORT_Z at the current xy
    "redetect",           # fresh BEV detection, report what is seen
    "retry_place",        # re-detect -> re-plan -> place again
    "retry_pick",         # part not held: re-detect -> pick again
    "release_blowoff",    # blow-off release at the current pose (operator only)
    "park_arm",           # view-park joints (clear the head camera)
    "call_operator",      # default / fallback
)
