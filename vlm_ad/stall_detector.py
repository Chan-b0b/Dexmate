"""작업 진행이 막혔는지(stall) 감지하는 경량 모듈.

anomaly_filter와는 완전히 다른 신호를 본다:
  - anomaly_filter: "위험하거나 비정상적인 상황" (충돌, 사람 근접 등)
        -> 순간적인 스파이크(광류 급변, 힘 급변)를 본다 -> 즉시 정지 후보
  - StallDetector: "위험하지는 않지만 목표를 달성할 수 없는 상황"
        (장애물에 막힘, 파지 반복 실패 등)
        -> 오히려 "변화가 없음"(움직임 정지)과 "힘이 계속 일정하게 유지됨
           (plateau)"의 조합을 본다 -> 작업 차단 후보

왜 두 신호를 구분해야 하는가:
  단순히 "힘이 크다"만 보면 정상적으로 물건을 꽉 쥐는 것과 장애물에 막혀
  계속 미는 것을 구분할 수 없다. 핵심 구분자는 "지속 시간"과 "변화 패턴"이다.
    - 충돌/충격: 힘이 짧은 시간에 급격히 튀었다가 사라짐 (스파이크)
    - 장애물에 막힘: 로봇이 움직이지 않는데 힘이 낮지 않은 수준에서
      오랫동안 거의 일정하게 유지됨 (plateau)

이 모듈은 "차단 후보"만 판단한다. 실제로 장애물 때문인지, 왜 막혔는지는
VLM(TaskFeasibilityVerifier)이 이미지를 보고 판단한다.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from config import StallDetectorConfig


class StallDetector:
    def __init__(self, cfg: StallDetectorConfig, filter_rate_hz: float) -> None:
        self._cfg = cfg
        window_len = max(1, int(cfg.stall_time_s * filter_rate_hz))
        self._flow_window: deque[float] = deque(maxlen=window_len)
        self._force_window: deque[float] = deque(maxlen=window_len)
        self._last_stall_check_ts = 0.0

    def reset(self) -> None:
        self._flow_window.clear()
        self._force_window.clear()

    def update(self, flow_magnitude: float, force_torque: np.ndarray) -> tuple[bool, dict]:
        """한 스텝의 광류/힘 데이터를 넣고 "작업 차단 후보" 여부를 반환한다.

        Args:
            flow_magnitude: anomaly_filter가 이미 계산한 광류 평균 크기
                (재사용 - 별도로 다시 계산하지 않음).
            force_torque: 현재 힘/토크 벡터.

        Returns:
            (is_stall_candidate, debug_info)
        """
        force_mag = float(np.linalg.norm(force_torque))
        self._flow_window.append(flow_magnitude)
        self._force_window.append(force_mag)

        debug_info = {
            "flow_mean": 0.0,
            "force_mean": 0.0,
            "force_std": 0.0,
            "window_full": False,
        }

        if len(self._flow_window) < self._flow_window.maxlen:
            return False, debug_info  # 아직 판단할 만큼 데이터가 쌓이지 않음

        flow_arr = np.array(self._flow_window)
        force_arr = np.array(self._force_window)

        motion_is_stopped = bool(np.max(flow_arr) < self._cfg.motion_stopped_flow_threshold)
        has_contact = bool(np.mean(force_arr) > self._cfg.min_force_floor_n)
        force_is_flat = bool(np.std(force_arr) < self._cfg.force_plateau_std_threshold)

        debug_info.update({
            "flow_mean": float(np.mean(flow_arr)),
            "force_mean": float(np.mean(force_arr)),
            "force_std": float(np.std(force_arr)),
            "window_full": True,
        })

        is_stall_candidate = motion_is_stopped and has_contact and force_is_flat
        return is_stall_candidate, debug_info

    def cooldown_ok(self) -> bool:
        return (time.monotonic() - self._last_stall_check_ts) >= self._cfg.check_cooldown_s

    def mark_checked(self) -> None:
        self._last_stall_check_ts = time.monotonic()
