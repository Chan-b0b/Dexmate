"""Online FiLM authority probe (vla_training/probe_film_authority_live.py).

A non-contact height sweep above the detected case top; the policy's
predictions are logged only, never commanded.

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

# Online FiLM authority probe: safe, non-contact height sweep above the detected
# case top. Values are cup-tip clearances; the probe converts them to EE z with
# SUCTION_LENGTH_M. Policy predictions are logged only and never commanded.
VLA_FILM_PROBE_CLEARANCES_M: tuple[float, ...] = (0.25, 0.10, 0.03)
VLA_FILM_PROBE_SETTLE_S: float = 0.5
# Counterfactual fz sweep in raw-force units. Converted to FiLM input units with
# the checkpoint runtime _fz_tau, so this stays interpretable across calibrations.
VLA_FILM_PROBE_FZ_DELTAS_N: tuple[float, ...] = (-3.0, 3.0)
