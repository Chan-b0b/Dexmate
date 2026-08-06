"""Tier-2 advisor: a local VLM proposes ONE skill from a bounded library.

The VLM never commands torques. It sees the head camera + a text context
(phase, trip reason, world state, resume-matrix escalation reason) and must
answer with strict JSON choosing from ALLOWED_SKILLS. Execution is gated
behind the operator (recovery.py). Endpoint unreachable, timeout, or
unparseable output all degrade to "call_operator" — the advisor can only
ever ADD options, never remove the operator fallback.

Backend: any OpenAI-compatible /chat/completions endpoint with vision
(Ollama `ollama pull qwen2.5vl:7b`, llama.cpp server, vLLM). Configure
VLM_BASE_URL / VLM_MODEL in config.py.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import numpy as np
import requests
from loguru import logger

from . import config as cfg

_SYSTEM = """You are the recovery advisor of a stationary industrial robot arm.
The robot moves battery cases between two stacks with a suction cup, following
a fixed scripted sequence. An anomaly monitor has PAUSED the script and the arm
is holding still at a safe height. You are shown the robot's head-camera image
and a text summary of the situation.

Choose exactly ONE next skill from this list (these are the only actions that
exist; each runs with built-in force/speed limits and operator approval):
{skills}

Rules:
- Prefer the least-committal safe skill. If the scene is unclear, occluded, or
  a human is in the workspace, choose "call_operator".
- "release_blowoff" drops the held part where it is — only when it is already
  resting in a valid seat.
- Answer with STRICT JSON only, no prose:
  {{"situation": "<one sentence>", "skill": "<name>", "confidence": <0..1>}}"""


@dataclass
class SkillProposal:
    skill: str
    situation: str
    confidence: float
    raw: str = ""


def grab_head_rgb(bot) -> "np.ndarray | None":
    obs = bot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
    rgb = obs.get("left_rgb")
    if isinstance(rgb, dict):
        rgb = rgb.get("data")
    return rgb


def _encode_jpeg_b64(rgb: np.ndarray) -> str:
    import cv2
    h, w = rgb.shape[:2]
    if w > cfg.VLM_IMAGE_MAX_W:
        s = cfg.VLM_IMAGE_MAX_W / w
        rgb = cv2.resize(rgb, (cfg.VLM_IMAGE_MAX_W, int(h * s)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


class VLMAdvisor:
    def __init__(self, base_url: str = cfg.VLM_BASE_URL,
                 model: str = cfg.VLM_MODEL) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model

    def consult(self, rgb: "np.ndarray | None", context: str) -> SkillProposal:
        fallback = SkillProposal("call_operator", "advisor unavailable", 0.0)
        if rgb is None:
            logger.warning("[ik_VLM] advisor: no camera frame — operator fallback")
            return fallback
        content = [{"type": "text", "text": context},
                   {"type": "image_url", "image_url": {
                       "url": f"data:image/jpeg;base64,{_encode_jpeg_b64(rgb)}"}}]
        body = {
            "model": self._model,
            "max_tokens": cfg.VLM_MAX_TOKENS,
            "temperature": 0.0,
            "messages": [
                {"role": "system",
                 "content": _SYSTEM.format(skills="\n".join(f"- {s}" for s in cfg.ALLOWED_SKILLS))},
                {"role": "user", "content": content},
            ],
        }
        try:
            r = requests.post(self._url, json=body, timeout=cfg.VLM_TIMEOUT_S)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            logger.warning("[ik_VLM] advisor request failed ({}) — operator fallback", e)
            return fallback
        prop = self._parse(text)
        logger.info("[ik_VLM] advisor: skill={} conf={:.2f} — {}",
                    prop.skill, prop.confidence, prop.situation)
        return prop

    @staticmethod
    def _parse(text: str) -> SkillProposal:
        fallback = SkillProposal("call_operator", "unparseable advisor output", 0.0, text)
        # tolerate code fences / prose around the JSON object
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return fallback
        try:
            d = json.loads(text[start:end + 1])
            skill = str(d.get("skill", ""))
            if skill not in cfg.ALLOWED_SKILLS:
                logger.warning("[ik_VLM] advisor proposed unknown skill {!r}", skill)
                return fallback
            return SkillProposal(skill, str(d.get("situation", "")),
                                 float(d.get("confidence", 0.0)), text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return fallback
