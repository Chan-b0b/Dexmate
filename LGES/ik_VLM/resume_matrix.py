"""Tier-1 re-entry decision: WorldState -> where to resume the script.

The matrix is deliberately cause-agnostic: it maps what the world LOOKS LIKE
now (seal x detection x phase) to a re-entry point, so situations we never
enumerated still resolve as long as re-reading the world is enough. What it
can't resolve escalates to the VLM advisor (Tier 2) or the operator.

    sealed  case detected        interpretation            resume
    ------  -------------------  -------------------------  -------------------
    yes     at expected pose     disturbed mid-place        retry_place
    yes     elsewhere / not      stale detection / moved    retry_place (fresh
            found                stack                       detect is the fix;
                                                             not found -> operator)
    no      near expected        pick never happened /      retry_pick
            (holding was          part fell back onto
             expected)            the stack
    no      unexpected position  part dropped somewhere     vlm (auto re-pick of
                                                             a dropped part is
                                                             NOT safe blind)
    no      not found            unknown                    operator
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config as cfg
from .world_state import WorldState


@dataclass
class ResumeAction:
    kind: str        # retry_place | retry_pick | vlm | operator
    why: str


def decide(w: WorldState) -> ResumeAction:
    if w.sealed is None and w.holding_expected:
        return ResumeAction("operator", "seal state unreadable while a part may be held")

    if w.sealed:
        if w.case_found:
            return ResumeAction("retry_place",
                                "part still held and a case is in view — re-plan the "
                                "place from the fresh detection")
        return ResumeAction("operator",
                            "part held but no case detected — reposition/inspect")

    # not sealed
    if w.holding_expected:
        # we thought we were carrying something and we aren't
        if w.case_found and (w.xy_err_m is None or w.xy_err_m <= cfg.CASE_XY_UNEXPECTED_M):
            return ResumeAction("retry_pick",
                                "part is not on the cup but sits near the expected "
                                "pose — re-pick it")
        if w.case_found:
            return ResumeAction("vlm",
                                f"part left the cup and lies {w.xy_err_m * 1000:.0f}mm "
                                "from the expected pose — blind re-pick unsafe")
        return ResumeAction("operator", "part left the cup and is not detected")

    # nothing was held (e.g. pick-phase anomaly)
    if w.case_found:
        return ResumeAction("retry_pick", "nothing held; case in view — retry the pick")
    return ResumeAction("vlm", "nothing held and no case detected — needs scene "
                               "understanding")
