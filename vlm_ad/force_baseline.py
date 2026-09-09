"""작업(task)마다 다른 "정상적인 힘 수준"을 온라인으로 추정하는 모듈.

문제: 고정된 하나의 힘 임계값(예: 40N)으로는 가벼운 파지 작업과 무거운
물체를 다루는 작업을 동시에 다룰 수 없다. 가벼운 작업에서는 40N이면
명백한 충돌이지만, 무거운 물체를 드는 작업에서는 30N이 정상 범위일 수 있다.

해결: 현재 작업이 시작된 이후 "이상하지 않다고 판단된 순간들"의 힘 값
분포를 계속 관찰해서, 이번 작업에서 지금까지 정상으로 보였던 힘 수준을
기준선으로 삼는다. 새 값이 이 기준선에서 얼마나 벗어났는지(비율)를 보고
판단하면, 작업마다 다른 "정상 힘"에 자동으로 적응한다.

통계량으로 평균/표준편차 대신 중앙값(median)과 MAD(median absolute
deviation)를 쓰는 이유: 평균/표준편차는 이상치(충돌 등) 몇 개가 섞여
들어오면 바로 왜곡된다. 중앙값/MAD는 이상치에 훨씬 덜 민감해서, 혹시
기준선에 충돌 데이터가 몇 개 섞여도 기준 자체가 크게 흔들리지 않는다.

주의: 이 기준선은 작업이 바뀌면(task_context 문자열이 바뀌면) 반드시
리셋되어야 한다. 그러지 않으면 이전 작업의 힘 수준이 새 작업에 잘못
적용된다 (예: 가벼운 파지 작업 기준선을 무거운 물체 작업에 그대로 쓰면
정상 힘도 계속 "이상"으로 오탐하게 됨).

이 파일은 두 가지를 제공한다:
  - AdaptiveForceBaseline: 스칼라 값(예: 손목 F/T 센서의 힘 크기) 하나에
    대한 기준선.
  - VectorAdaptiveBaseline: 벡터(예: 관절별 모터 전류 7개)에 대한 기준선.
    각 관절마다 독립적으로 median/MAD를 추적한다 - 관절별로 정상 전류
    수준이 다르기 때문이다 (어깨 관절과 손목 관절은 원래 부하가 다르다).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from config import ForceBaselineConfig


class AdaptiveForceBaseline:
    def __init__(self, cfg: ForceBaselineConfig) -> None:
        self._cfg = cfg
        self._window: deque[float] = deque(maxlen=cfg.window_size)
        self._current_task_key: str | None = None

    def update(self, task_key: str, force_magnitude: float) -> None:
        """정상으로 판단된 힘 샘플을 기준선에 반영한다.

        이상 후보로 이미 플래그된 샘플은 호출하는 쪽(safety_supervisor)에서
        걸러내고 넘겨야 한다 - 그러지 않으면 기준선이 이상 상황 자체를
        "정상"으로 학습해버린다.
        """
        if task_key != self._current_task_key:
            # 작업이 바뀌었으니 이전 작업의 힘 수준을 리셋
            self._current_task_key = task_key
            self._window.clear()
        self._window.append(force_magnitude)

    def is_ready(self) -> bool:
        return len(self._window) >= self._cfg.min_samples

    def deviation_ratio(self, force_magnitude: float) -> float | None:
        """현재 힘 값이 기준선에서 얼마나 벗어났는지 비율로 반환.

        기준선이 아직 충분히 쌓이지 않았으면 None을 반환한다 (호출부에서
        이 경우 fallback 절대 임계값을 쓰도록 처리해야 함).
        """
        if not self.is_ready():
            return None

        arr = np.array(self._window)
        baseline = float(np.median(arr))
        mad = float(np.median(np.abs(arr - baseline)))
        # MAD가 0에 가까우면(힘이 거의 변화가 없던 작업) 나누기 폭주를 막기 위해
        # 최소값을 둔다.
        mad = max(mad, self._cfg.min_mad_n)

        return abs(force_magnitude - baseline) / mad

    def get_baseline(self) -> float | None:
        if not self.is_ready():
            return None
        return float(np.median(np.array(self._window)))


class VectorAdaptiveBaseline:
    """AdaptiveForceBaseline을 벡터(관절별 여러 값)로 확장한 버전.

    관절별 모터 전류처럼 "차원마다 정상 수준이 다른" 다변량 신호에 쓴다.
    각 차원(관절)마다 독립적으로 median/MAD 기준선을 추적한다 - 관절마다
    부하가 다르므로 공통 기준값을 쓰면 안 된다 (예: 어깨 관절은 원래
    전류가 크고 손목 관절은 작다).
    """

    def __init__(self, cfg: ForceBaselineConfig, dim: int) -> None:
        self._cfg = cfg
        self._dim = dim
        self._window: deque[np.ndarray] = deque(maxlen=cfg.window_size)
        self._current_task_key: str | None = None

    def update(self, task_key: str, values: np.ndarray) -> None:
        if task_key != self._current_task_key:
            self._current_task_key = task_key
            self._window.clear()
        self._window.append(np.asarray(values, dtype=np.float64))

    def is_ready(self) -> bool:
        return len(self._window) >= self._cfg.min_samples

    def deviation_ratios(self, values: np.ndarray) -> np.ndarray | None:
        """차원(관절)별 편차 비율 벡터를 반환. 기준선 미준비 시 None."""
        if not self.is_ready():
            return None

        arr = np.stack(self._window, axis=0)  # (N, dim)
        baseline = np.median(arr, axis=0)      # (dim,)
        mad = np.median(np.abs(arr - baseline), axis=0)
        mad = np.maximum(mad, self._cfg.min_mad_n)

        return np.abs(np.asarray(values, dtype=np.float64) - baseline) / mad

    def get_baseline(self) -> np.ndarray | None:
        if not self.is_ready():
            return None
        return np.median(np.stack(self._window, axis=0), axis=0)
