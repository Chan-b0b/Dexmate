"""Tier-0/1/2 recovery executor.

Automation contract (agreed 0806): Tier 0 (halt + small lift) is AUTOMATIC;
everything after it — lift to transport, re-detect, re-entry, any VLM-proposed
skill — runs only after an operator Enter. The chassis is never moved here;
if the world needs the base repositioned, that is the operator's call.

The actual re-entry actions (re-plan + place / pick) are CLOSURES supplied by
the caller (supervisor.py), so this module stays free of pose-resolution
details and the caller keeps its own trims/expected-z policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from loguru import logger

from . import config as cfg
from . import resume_matrix, world_state
from .vlm_advisor import SkillProposal, VLMAdvisor, grab_head_rgb


@dataclass
class RecoveryContext:
    label: str
    station: str
    layers: int
    phase: str
    holding_expected: bool
    trip_reason: str
    plane_z: float | None = None
    expected_xy: tuple | None = None
    # (case_center | None) -> bool(success). Supplied by the supervisor.
    retry_place: Callable | None = None
    retry_pick: Callable | None = None
    history: list = field(default_factory=list)


def _approve(prompt: str) -> bool:
    return input(f"[ik_VLM] {prompt} — Enter=proceed / q=stop: ").strip().lower() != "q"


class RecoveryRunner:
    def __init__(self, bot, mover, advisor: "VLMAdvisor | None" = None) -> None:
        self._bot = bot
        self._mover = mover
        self._advisor = advisor

    # ------------------------------------------------------------------
    # Tier 0 — automatic. The descent hook already halted the stream; get
    # a small, always-safe clearance off whatever was being touched.
    # ------------------------------------------------------------------
    def hold_and_lift(self) -> None:
        pos, rpy = self._mover.current_ee_pose()
        logger.warning("[ik_VLM] Tier 0: holding, lifting {:+.0f}mm (ee_z {:.4f})",
                       cfg.HOLD_LIFT_M * 1000, float(pos[2]))
        self._mover.move_ee_vertical(float(pos[2]) + cfg.HOLD_LIFT_M, rpy)

    # ------------------------------------------------------------------
    # Tier 1/2 — operator-gated.
    # ------------------------------------------------------------------
    def run(self, ctx: RecoveryContext) -> str:
        """Full recovery after hold_and_lift(). Returns "resumed" (a re-entry
        action succeeded — the caller may continue the sequence) or "stopped"
        (operator ended it; the arm is left safe, part held if sealed)."""
        logger.warning("[ik_VLM] anomaly: {} (item {}@{}, phase {})",
                       ctx.trip_reason, ctx.label, ctx.station, ctx.phase)
        if not _approve("lift to transport + re-detect + re-entry"):
            logger.warning("[ik_VLM] operator stopped at hold (part kept as-is)")
            return "stopped"

        from ..ik_demo import chassis_sequence as cs
        _, rpy = self._mover.current_ee_pose()
        self._mover._lift_to_transport(rpy)
        cs._view_park(self._mover, ctx.label)   # clear the head-camera view

        w = world_state.classify(
            self._bot, phase=ctx.phase, label=ctx.label, station=ctx.station,
            holding_expected=ctx.holding_expected, layers=ctx.layers,
            plane_z=ctx.plane_z, expected_xy=ctx.expected_xy)
        action = resume_matrix.decide(w)
        logger.info("[ik_VLM] resume decision: {} ({})", action.kind, action.why)
        ctx.history.append((action.kind, action.why))
        return self._execute(action.kind, action.why, w, ctx)

    def _execute(self, kind: str, why: str, w, ctx: RecoveryContext) -> str:
        if kind == "retry_place":
            if ctx.retry_place is not None and _approve(f"retry place ({why})"):
                if ctx.retry_place(w.case_center):
                    logger.info("[ik_VLM] re-entry place SUCCEEDED — resuming")
                    return "resumed"
                logger.error("[ik_VLM] re-entry place failed")
            return self._operator(w, ctx)
        if kind == "retry_pick":
            if ctx.retry_pick is not None and _approve(f"retry pick ({why})"):
                if ctx.retry_pick(w.case_center):
                    logger.info("[ik_VLM] re-entry pick SUCCEEDED — resuming")
                    return "resumed"
                logger.error("[ik_VLM] re-entry pick failed")
            return self._operator(w, ctx)
        if kind == "vlm":
            return self._consult_vlm(why, w, ctx)
        return self._operator(w, ctx)

    # ------------------------------------------------------------------
    def _consult_vlm(self, why: str, w, ctx: RecoveryContext) -> str:
        if self._advisor is None:
            return self._operator(w, ctx)
        context = (f"The anomaly monitor tripped: {ctx.trip_reason}.\n"
                   f"Script phase: {ctx.phase}, item {ctx.label} at the {ctx.station} "
                   f"station.\nWorld state: {w.summary()}.\n"
                   f"Rule-based re-entry escalated because: {why}.\n"
                   "What single skill should run next?")
        prop = self._advisor.consult(grab_head_rgb(self._bot), context)
        ctx.history.append(("vlm", prop.skill, prop.situation))
        if prop.skill == "call_operator":
            return self._operator(w, ctx)
        if not _approve(f"VLM proposes '{prop.skill}' — {prop.situation} "
                        f"(conf {prop.confidence:.2f})"):
            return self._operator(w, ctx)
        return self._run_skill(prop.skill, w, ctx)

    def _run_skill(self, skill: str, w, ctx: RecoveryContext) -> str:
        from ..ik_demo import chassis_sequence as cs
        from ..ik_demo.drivers import suction_io
        if skill == "hold":
            return self._operator(w, ctx)
        if skill == "lift_to_transport":
            _, rpy = self._mover.current_ee_pose()
            self._mover._lift_to_transport(rpy)
            return self._operator(w, ctx)
        if skill == "redetect":
            w = world_state.classify(
                self._bot, phase=ctx.phase, label=ctx.label, station=ctx.station,
                holding_expected=ctx.holding_expected, layers=ctx.layers,
                plane_z=ctx.plane_z, expected_xy=ctx.expected_xy)
            return self._execute(resume_matrix.decide(w).kind, "after redetect", w, ctx)
        if skill == "retry_place":
            return self._execute("retry_place", "VLM proposal", w, ctx)
        if skill == "retry_pick":
            return self._execute("retry_pick", "VLM proposal", w, ctx)
        if skill == "release_blowoff":
            suction_io.release()
            logger.warning("[ik_VLM] released the part at the current pose")
            return self._operator(w, ctx)
        if skill == "park_arm":
            cs._view_park(self._mover, ctx.label)
            return self._operator(w, ctx)
        return self._operator(w, ctx)

    # ------------------------------------------------------------------
    def _operator(self, w, ctx: RecoveryContext) -> str:
        """Interactive last resort: the operator drives the skill library."""
        skills = ", ".join(cfg.ALLOWED_SKILLS)
        logger.warning("[ik_VLM] OPERATOR: world = {}", w.summary())
        while True:
            cmd = input(f"[ik_VLM] skill ({skills}) / r=resume-anyway / q=stop: ").strip().lower()
            if cmd == "q":
                return "stopped"
            if cmd == "r":
                return "resumed"
            if cmd in cfg.ALLOWED_SKILLS and cmd != "call_operator":
                out = self._run_skill(cmd, w, ctx)
                if out in ("resumed", "stopped"):
                    return out
            else:
                logger.info("unknown skill {!r}", cmd)
