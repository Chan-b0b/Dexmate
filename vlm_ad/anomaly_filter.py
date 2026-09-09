"""경량 1차 필터.

VLM은 느리고 비싸므로 매 프레임 호출하지 않는다. 이 모듈은 값싼 연산(광류,
힘/토크 변화율)만으로 "이상 후보"를 빠르게 골라내고, 후보가 나왔을 때만
VLM 검증 단계로 넘긴다. 오탐이 있어도 괜찮다 (VLM이 다시 걸러줌) — 하지만
미탐(진짜 이상을 놓치는 것)은 최소화하는 방향으로 임계값을 보수적으로 잡는다.

실제 배포 시에는 이 필터를 더 정교한 모델(예: 정상 동작 영상으로 학습한
경량 오토인코더의 재구성 오차)로 교체할 수 있다. 인터페이스만 유지하면 된다.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from config import AnomalyFilterConfig


class LightweightAnomalyFilter:
    """광류 급변 + 힘/토크 급변 기반 1차 이상 후보 탐지기.

    시작 직후 startup_warmup_s 동안은 광류 트리거를 무시한다 (카메라 스트림
    안정화 전 과도구간 - config.AnomalyFilterConfig의 실측값 주석 참고).
    """

    def __init__(self, cfg: AnomalyFilterConfig) -> None:
        self._cfg = cfg
        self._prev_gray: np.ndarray | None = None
        self._prev_force: np.ndarray | None = None
        self._flow_history: deque[float] = deque(maxlen=30)
        self._warmup_frames = max(0, round(cfg.startup_warmup_s * cfg.filter_rate_hz))
        self._frames_seen = 0

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_force = None
        self._flow_history.clear()
        # 과도구간 카운터도 되돌린다. reset()은 스트림이 끊겼다 다시 붙는
        # 상황에서 호출되므로, 그때도 같은 과도구간이 다시 발생한다.
        self._frames_seen = 0

    def check(self, frame_bgr: np.ndarray, force_torque: np.ndarray) -> tuple[bool, dict]:
        """한 프레임/센서 샘플을 검사해 이상 후보 여부를 반환한다.

        Args:
            frame_bgr: 카메라 프레임 (H, W, 3), BGR.
            force_torque: 힘/토크 센서 벡터 (예: [fx, fy, fz, tx, ty, tz]).

        Returns:
            (is_candidate, debug_info) - debug_info["flow_magnitude"]는
            StallDetector가 재사용하므로 항상 포함되어야 한다.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 120))  # 연산량 절감

        flow_mag = 0.0
        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray_small, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            flow_mag = float(np.mean(np.linalg.norm(flow, axis=2)))
        self._prev_gray = gray_small
        self._flow_history.append(flow_mag)

        force_delta = 0.0
        if self._prev_force is not None:
            force_delta = float(np.max(np.abs(force_torque - self._prev_force)))
        self._prev_force = force_torque.copy()

        self._frames_seen += 1
        warming_up = self._frames_seen <= self._warmup_frames

        motion_spike = flow_mag > self._cfg.flow_magnitude_threshold
        force_spike = force_delta > self._cfg.force_delta_threshold_n

        # 과도구간에는 광류 트리거만 무시한다 (config.startup_warmup_s 주석의
        # 실측값 참고). 힘 채널에는 이런 과도구간이 관측되지 않았으므로
        # (정지 상태 force_delta max 0.54N) 힘 스파이크는 그대로 살려둔다 -
        # 여기서 힘까지 막으면 기동 직후 1초 동안 충돌을 못 보는 구멍이 생긴다.
        #
        # motion_spike/flow_magnitude는 억제하지 않고 원래 측정값을 그대로
        # 내보낸다. calibrate_filter.py가 임계값을 산출할 때 필요하고,
        # StallDetector도 flow_magnitude를 계속 받아야 한다.
        is_candidate = (motion_spike and not warming_up) or force_spike
        debug_info = {
            "flow_magnitude": flow_mag,
            "force_delta_n": force_delta,
            "motion_spike": motion_spike,
            "force_spike": force_spike,
            "warming_up": warming_up,
            "timestamp": time.time(),
        }
        return is_candidate, debug_info
