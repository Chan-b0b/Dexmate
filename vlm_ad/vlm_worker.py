"""VLM 호출을 감시 루프에서 떼어내는 워커.

왜 필요한가:
    VLM 판정은 로컬 vLLM 서버에 HTTP로 물어보는 일이라 응답까지
    request_timeout_s(4초)가 걸릴 수 있다. 이 호출을 감시 루프 틱 안에서
    동기로 하면 그 시간 동안 감시 루프 전체가 멈춘다 - 관절 전류 이상탐지,
    힘 스파이크 감지, 정지 미확인 상태 복구 재시도가 다 같이 멈춘다.
    Qwen3.5-35B-A3B로 올리면서 이미지 프리필이 무거워져 이 구간이 더 길어졌다.

    더 위험한 부작용: dexcontrol의 E-Stop 서비스 콜은 클라이언트 타임아웃이
    50ms로 매우 짧게 하드코딩되어 있다 (config.py의 EmergencyStopRetryConfig
    docstring 참고). 같은 스레드가 무거운 작업에 붙잡혀 있으면 서비스는
    정상인데도 정지 호출이 타임아웃날 수 있다. HTTP 대기는 GIL을 놓기 때문에
    호출을 워커 스레드로 옮기면 감시 루프와 정지 경로가 계속 살아 있는다.

설계:
    - 동시에 하나만 in-flight. 처리 중에 들어온 요청은 버린다 (몇 초 전
      프레임에 대한 판정은 이미 쓸모없고, 큐를 쌓으면 판정이 계속 뒤처진다).
      호출 주기는 원래도 상위의 쿨다운이 제한하고 있었다.
    - poll()은 절대 블록하지 않는다. 결과가 없으면 None.
    - 워커가 deadline_s 안에 안 돌아오면 그 요청을 포기하고 "판단 불가"
      판정을 돌려준다 (호출부가 fail-safe로 처리). 포기된 요청이 뒤늦게
      끝내도 그 결과는 버린다 - generation 번호로 구분한다. 오래된 판정이
      지금 상태를 덮어쓰면 안 되기 때문이다.

    이 클래스는 판정의 내용을 전혀 해석하지 않는다. 정책 판단은 전부
    safety_supervisor에 남겨두고, 여기서는 "언제 부르고, 언제 포기하고,
    어떤 결과를 신뢰할지"만 다룬다.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Generic, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

V = TypeVar("V")


class AsyncVerifier(Generic[V]):
    """VLM 판정 함수 하나를 워커 스레드에서 실행하는 래퍼.

    Args:
        name: 로그 구분용 이름 ("anomaly" / "feasibility").
        verify_fn: (frame_bgr, task_context) -> 판정. VLMAnomalyVerifier.verify
            또는 TaskFeasibilityVerifier.verify. 이 함수들은 자체적으로 모든
            예외를 흡수하고 ok=False 판정을 돌려주도록 되어 있다.
        failure_factory: (reason) -> ok=False 판정. 워커가 예외로 죽거나
            deadline을 넘겼을 때 호출부에 돌려줄 "판단 불가" 객체를 만든다.
            판정 타입이 경로마다 다르므로 생성 책임을 호출부에 둔다.
        deadline_s: 이 시간을 넘기면 요청을 포기한다. verify_fn 내부의 HTTP
            타임아웃이 어떤 이유로든 안 먹었을 때를 위한 마지막 안전망이므로
            request_timeout_s보다 넉넉하게 잡는다.
    """

    def __init__(
        self,
        name: str,
        verify_fn: Callable[[np.ndarray, str], V],
        failure_factory: Callable[[str], V],
        deadline_s: float,
    ) -> None:
        self._name = name
        self._verify_fn = verify_fn
        self._failure_factory = failure_factory
        self._deadline_s = deadline_s

        self._lock = threading.Lock()
        self._in_flight = False
        self._submitted_at = 0.0
        self._generation = 0
        self._result: V | None = None

    def busy(self) -> bool:
        """판정을 기다리는 중인지. 상위의 쿨다운 판단과 함께 쓰인다."""
        with self._lock:
            return self._in_flight

    def submit(self, frame_bgr: np.ndarray, task_context: str) -> bool:
        """판정 요청을 워커에 넘긴다.

        Returns:
            True면 요청이 시작됐다. False면 이미 처리 중이라 이번 요청을
            버렸다는 뜻이므로, 호출부는 쿨다운 타임스탬프를 갱신하면 안 된다.
        """
        with self._lock:
            if self._in_flight:
                return False
            self._generation += 1
            generation = self._generation
            self._in_flight = True
            self._submitted_at = time.monotonic()

        # frame_bgr을 복사하지 않는 이유: DexcontrolAdapter.get_camera_frame()이
        # cv2.cvtColor로 매 틱 새 배열을 만들어 돌려주므로, 워커가 들고 있는
        # 동안 다음 틱이 그 버퍼를 덮어쓸 일이 없다 (파이썬 refcount가 잡아둔다).
        # 프레임이 7MB급이라 감시 루프 안에서 불필요한 memcpy를 하지 않는다.
        threading.Thread(
            target=self._work,
            args=(generation, frame_bgr, task_context),
            name=f"vlm-{self._name}",
            daemon=True,
        ).start()
        return True

    def _work(self, generation: int, frame_bgr: np.ndarray, task_context: str) -> None:
        try:
            verdict = self._verify_fn(frame_bgr, task_context)
        except Exception as e:  # noqa: BLE001 - 워커가 조용히 죽으면 판정이 영원히 안 온다
            logger.exception("VLM 워커(%s) 예외", self._name)
            verdict = self._failure_factory(f"워커 예외: {e}")

        with self._lock:
            discarded = generation != self._generation
            if not discarded:
                self._result = verdict
                self._in_flight = False

        if discarded:
            # deadline 초과로 이미 포기된 요청. 그 사이 상위는 판단 불가로
            # 처리했으므로 이 결과를 반영하면 오래된 판정으로 덮어쓰게 된다.
            logger.warning(
                "VLM 워커(%s) 뒤늦은 응답 폐기 (요청 gen=%d)", self._name, generation,
            )

    def poll(self) -> V | None:
        """완료된 판정을 꺼낸다. 없으면 None. 절대 블록하지 않는다.

        deadline을 넘긴 in-flight 요청은 여기서 포기 처리되고 "판단 불가"
        판정으로 반환된다.
        """
        with self._lock:
            if self._result is not None:
                verdict = self._result
                self._result = None
                return verdict

            if not self._in_flight:
                return None

            if time.monotonic() - self._submitted_at < self._deadline_s:
                return None

            # deadline 초과: 요청을 포기한다. generation을 올려서 뒤늦게
            # 끝나는 워커의 결과를 무효화한다 (_work의 discarded 분기).
            self._generation += 1
            self._in_flight = False

        logger.error(
            "VLM 워커(%s)가 %.1fs 안에 응답하지 않아 요청을 포기합니다 (판단 불가 처리)",
            self._name, self._deadline_s,
        )
        return self._failure_factory(f"워커 응답 없음 ({self._deadline_s:.1f}s 초과)")
