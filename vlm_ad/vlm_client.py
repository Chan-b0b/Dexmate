"""로컬 vLLM 서버(OpenAI 호환 API)에 올라간 Qwen3.5-35B-A3B 호출 클라이언트.

핵심 설계 포인트:
  1. 구조화된 JSON 출력을 강제해서 파싱을 안정화한다.
  2. 정상 동작 컨텍스트를 프롬프트에 명시해서 오탐을 줄인다.
  3. 타임아웃/파싱 실패 시 예외를 던지지 않고 "판단 불가"를 명시적으로
     반환한다 — 호출부(safety_supervisor)가 이를 fail-safe(정지)로
     처리할 수 있게 한다.
  4. Qwen3 계열의 thinking 모드를 서버 쪽에서 끄고(chat_template_kwargs),
     그래도 <think> 블록이 새어 나오는 경우를 파서에서 한 번 더 막는다.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
from openai import APITimeoutError, OpenAI

from config import VLMConfig

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _chat_extra_body(cfg: VLMConfig) -> dict:
    """Qwen3 계열 thinking 모드를 끄기 위한 vLLM 확장 파라미터.

    vLLM의 OpenAI 호환 서버는 chat_template_kwargs를 그대로 chat template에
    넘겨주고, Qwen3 계열 template은 enable_thinking=False를 받으면 <think>
    블록을 생성하지 않는다. Qwen2.5-VL 시절에는 없던 파라미터다.
    """
    if not cfg.disable_thinking:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _strip_thinking(raw_text: str) -> str:
    """모델이 남긴 <think> 블록을 제거한다.

    disable_thinking=True로 요청해도 방어적으로 한 번 더 처리한다: 서버의
    chat template 버전에 따라 이 옵션이 무시될 수 있고, 그러면 <think> 안의
    중괄호가 아래 JSON 추출 정규식(greedy `\\{.*\\}`)에 섞여 들어가 판정이
    통째로 "파싱 실패"(= fail-safe 정지)로 떨어진다. max_tokens에서 잘려
    </think>가 아예 없는 경우도 같이 처리한다.
    """
    text = _THINK_BLOCK_RE.sub("", raw_text)
    unclosed = text.find("<think>")
    if unclosed != -1:
        text = text[:unclosed]
    return text


_SYSTEM_PROMPT = """당신은 산업/서비스 현장에서 동작하는 로봇의 안전 감시 시스템입니다.
카메라 이미지 한 장과 로봇의 현재 작업 설명을 보고, 로봇이 즉시 멈춰야 할
비정상적이거나 위험한 상황인지 판단하세요.

반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 추가하지 마세요.
{
  "is_anomaly": true 또는 false,
  "confidence": 0.0에서 1.0 사이 숫자,
  "reason": "판단 근거를 한국어로 20단어 이내로 간결하게"
}

판단 기준:
- 사람이 로봇의 작업 반경 안에 위험하게 근접했는가
- 로봇이 계획된 작업과 무관하게 충돌, 낙하, 걸림 등의 상태에 있는가
- 장면에 화재, 파손, 누출 등 명백한 위험 신호가 있는가
- 단순히 조명 변화, 그림자, 정상적인 사람의 통행은 이상으로 판단하지 마세요
"""


@dataclass
class VLMVerdict:
    is_anomaly: bool
    confidence: float
    reason: str
    ok: bool  # False면 호출 실패/파싱 실패 (판단 불가 상태)


class VLMAnomalyVerifier:
    def __init__(self, cfg: VLMConfig) -> None:
        self._cfg = cfg
        # max_retries=0으로 끄는 이유: openai SDK 기본값은 2라서 타임아웃 1회가
        # 실제로는 "3회 시도 + 백오프"로 번져 request_timeout_s의 6배 넘게
        # 걸린다 (실측: 0.5s 설정 -> 3.07s에 반환, 4.0s 설정이면 약 13s).
        # 그러면 (1) 판단 불가 판정이 그만큼 늦어지고 (2) worker_deadline_s
        # (10s)에 걸려 요청이 포기된 뒤에도 좀비 스레드가 서버를 계속
        # 두드린다. 안전 판정에서 13초 지난 답은 쓸모없다. 재시도 정책은
        # 상위(vlm_cooldown_s + 연속 실패 카운터)가 이미 갖고 있다.
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.request_timeout_s,
            max_retries=0,
        )

    @staticmethod
    def _encode_frame(frame_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("이미지 인코딩 실패")
        return base64.b64encode(buf).decode("utf-8")

    def verify(
        self, frame_bgr: np.ndarray, task_context: str, timeout_s: float | None = None,
    ) -> VLMVerdict:
        """이상 후보 프레임을 VLM에게 보내 정밀 판단을 받는다.

        Args:
            frame_bgr: 이상 후보로 걸린 시점의 카메라 프레임.
            task_context: 로봇이 현재 수행 중인 작업에 대한 짧은 설명.
                (예: "테이블 위 물체를 오른팔로 집는 중")
            timeout_s: 이 호출만 다른 타임아웃을 쓰고 싶을 때 지정한다.
                None이면 VLMConfig.request_timeout_s(클라이언트 기본값).
                콜드 스타트를 감안해야 하는 워밍업 호출만 길게 준다.
        """
        try:
            b64_image = self._encode_frame(frame_bgr)
        except ValueError as e:
            logger.error("frame encode failed: %s", e)
            return VLMVerdict(False, 0.0, "인코딩 실패", ok=False)

        user_text = f"현재 작업: {task_context}\n이 이미지가 비정상 상황인지 판단하세요."

        client = self._client if timeout_s is None else self._client.with_options(timeout=timeout_s)

        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                            },
                        ],
                    },
                ],
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                extra_body=_chat_extra_body(self._cfg),
            )
        except APITimeoutError:
            logger.warning("VLM 호출 타임아웃 (%.1fs)", self._cfg.request_timeout_s)
            return VLMVerdict(False, 0.0, "타임아웃", ok=False)
        except Exception as e:  # noqa: BLE001 - 안전계층이므로 어떤 예외든 정지 판단으로 흡수
            logger.error("VLM 호출 실패: %s", e)
            return VLMVerdict(False, 0.0, f"호출 오류: {e}", ok=False)

        raw_text = response.choices[0].message.content or ""
        return self._parse_verdict(raw_text)

    @staticmethod
    def _parse_verdict(raw_text: str) -> VLMVerdict:
        # 모델이 코드블록(```json ... ```)으로 감싸거나 <think> 블록을 남기는
        # 경우를 대비해 thinking을 걷어내고 JSON 부분만 추출
        match = re.search(r"\{.*\}", _strip_thinking(raw_text), re.DOTALL)
        if not match:
            logger.error("VLM 응답에서 JSON을 찾을 수 없음: %r", raw_text)
            return VLMVerdict(False, 0.0, "파싱 실패", ok=False)

        try:
            data = json.loads(match.group(0))
            is_anomaly = bool(data["is_anomaly"])
            confidence = float(data["confidence"])
            reason = str(data.get("reason", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("VLM JSON 파싱 오류: %s / raw=%r", e, raw_text)
            return VLMVerdict(False, 0.0, "파싱 실패", ok=False)

        confidence = max(0.0, min(1.0, confidence))
        return VLMVerdict(is_anomaly, confidence, reason, ok=True)


# ----------------------------------------------------------------------------
# 작업 차단(task feasibility) 판단 - anomaly와는 다른 질문을 던진다.
# "위험한가?"가 아니라 "이 작업을 계속할 수 있는가?"를 묻는다.
# ----------------------------------------------------------------------------

_FEASIBILITY_SYSTEM_PROMPT = """당신은 로봇이 맡은 작업을 계속 수행할 수 있는지
판단하는 감시 시스템입니다. 로봇은 위험하지는 않지만, 목표를 향해 진행하지
못하고 같은 자리에서 힘을 계속 가하고 있는 상태로 관찰되었습니다.

반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 추가하지 마세요.
{
  "is_blocked": true 또는 false,
  "confidence": 0.0에서 1.0 사이 숫자,
  "blocking_reason": "무엇이 막고 있는지 한국어로 20단어 이내로 간결하게",
  "suggested_action": "wait_and_retry" 또는 "needs_replanning" 또는 "needs_human_intervention" 중 하나
}

판단 기준:
- 목표 물체나 경로에 예상치 못한 장애물(다른 물체, 사람, 구조물)이 있는가
- 잡으려는 물체의 위치/자세가 계획과 달라 파지가 불가능한 상태인가
- 단순히 정상적인 파지/삽입 과정에서 발생하는 일시적 저항인지, 아니면 진짜로
  막혀 있는지 이미지로 구분하세요. 확신이 낮으면 confidence를 낮게 주세요.
- suggested_action:
    wait_and_retry: 장애물이 곧 사라질 수 있는 일시적 상황 (예: 사람이 지나가는 중)
    needs_replanning: 장애물이 고정적이라 다른 경로/파지 방식이 필요
    needs_human_intervention: 로봇이 스스로 해결할 수 없는 상황 (예: 물체 파손,
        예상 밖의 구조 변경)
"""


@dataclass
class TaskFeasibilityVerdict:
    is_blocked: bool
    confidence: float
    blocking_reason: str
    suggested_action: str  # "wait_and_retry" | "needs_replanning" | "needs_human_intervention"
    ok: bool  # False면 호출 실패/파싱 실패


class TaskFeasibilityVerifier:
    """작업이 장애물 등으로 차단되었는지 VLM에게 확인하는 클라이언트.

    VLMAnomalyVerifier와 별도 클래스로 분리한 이유: 시스템 프롬프트와 판단
    기준, 출력 스키마가 완전히 다르다 (안전 위험 여부가 아니라 작업 수행
    가능 여부를 묻는다). 같은 vLLM 서버/모델을 재사용하되 호출 방식만 다르다.
    """

    def __init__(self, cfg: VLMConfig) -> None:
        self._cfg = cfg
        # max_retries=0으로 끄는 이유: openai SDK 기본값은 2라서 타임아웃 1회가
        # 실제로는 "3회 시도 + 백오프"로 번져 request_timeout_s의 6배 넘게
        # 걸린다 (실측: 0.5s 설정 -> 3.07s에 반환, 4.0s 설정이면 약 13s).
        # 그러면 (1) 판단 불가 판정이 그만큼 늦어지고 (2) worker_deadline_s
        # (10s)에 걸려 요청이 포기된 뒤에도 좀비 스레드가 서버를 계속
        # 두드린다. 안전 판정에서 13초 지난 답은 쓸모없다. 재시도 정책은
        # 상위(vlm_cooldown_s + 연속 실패 카운터)가 이미 갖고 있다.
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.request_timeout_s,
            max_retries=0,
        )

    def verify(
        self, frame_bgr: np.ndarray, task_context: str, timeout_s: float | None = None,
    ) -> TaskFeasibilityVerdict:
        """작업 차단 여부를 VLM에게 확인한다.

        Args:
            timeout_s: VLMAnomalyVerifier.verify와 같은 의미 (워밍업용).
        """
        try:
            b64_image = VLMAnomalyVerifier._encode_frame(frame_bgr)
        except ValueError as e:
            logger.error("frame encode failed: %s", e)
            return TaskFeasibilityVerdict(False, 0.0, "인코딩 실패", "needs_human_intervention", ok=False)

        user_text = (
            f"현재 작업: {task_context}\n"
            "로봇이 진행하지 못하고 있습니다. 이 상황이 작업 차단인지 판단하세요."
        )

        client = self._client if timeout_s is None else self._client.with_options(timeout=timeout_s)

        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": _FEASIBILITY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                            },
                        ],
                    },
                ],
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                extra_body=_chat_extra_body(self._cfg),
            )
        except APITimeoutError:
            logger.warning("VLM(feasibility) 호출 타임아웃 (%.1fs)", self._cfg.request_timeout_s)
            return TaskFeasibilityVerdict(False, 0.0, "타임아웃", "needs_human_intervention", ok=False)
        except Exception as e:  # noqa: BLE001
            logger.error("VLM(feasibility) 호출 실패: %s", e)
            return TaskFeasibilityVerdict(False, 0.0, f"호출 오류: {e}", "needs_human_intervention", ok=False)

        raw_text = response.choices[0].message.content or ""
        return self._parse_verdict(raw_text)

    @staticmethod
    def _parse_verdict(raw_text: str) -> TaskFeasibilityVerdict:
        match = re.search(r"\{.*\}", _strip_thinking(raw_text), re.DOTALL)
        if not match:
            logger.error("VLM(feasibility) 응답에서 JSON을 찾을 수 없음: %r", raw_text)
            return TaskFeasibilityVerdict(False, 0.0, "파싱 실패", "needs_human_intervention", ok=False)

        try:
            data = json.loads(match.group(0))
            is_blocked = bool(data["is_blocked"])
            confidence = float(data["confidence"])
            blocking_reason = str(data.get("blocking_reason", ""))
            suggested_action = str(data.get("suggested_action", "needs_human_intervention"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("VLM(feasibility) JSON 파싱 오류: %s / raw=%r", e, raw_text)
            return TaskFeasibilityVerdict(False, 0.0, "파싱 실패", "needs_human_intervention", ok=False)

        if suggested_action not in ("wait_and_retry", "needs_replanning", "needs_human_intervention"):
            suggested_action = "needs_human_intervention"

        confidence = max(0.0, min(1.0, confidence))
        return TaskFeasibilityVerdict(is_blocked, confidence, blocking_reason, suggested_action, ok=True)
