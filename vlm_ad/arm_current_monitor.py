"""관절별 모터 전류(cur)를 이용한 이상탐지.

확인된 사실:
  1. get_robot_state.py를 실제 로봇에서 실행한 결과 (dexcontrol 0.5.0 기준,
     당시엔 Arm.get_state()가 아직 public이었음):
        bot.left_arm.get_state() -> {
            'cur': [-3.63, -3.02, -4.802, 0.262, 0.578, 0.114, 0.32],
            'pos': [...], 'vel': [...],
            'receive_time_ns': ..., 'timestamp_ns': ..., 'error': {...},
        }
  2. component.py(v0.4.5, 실제 배포 버전) 원문 확인 결과: get_state()는
     _get_state()로 private화됐고, 대신 get_joint_current(joint_id=None)라는
     전용 public 메서드가 생겼다. robot_interface.py의
     get_arm_joint_currents()는 이제 이 메서드를 직접 호출한다.

왜 momentum_observer.py(운동량 관측기) 대신 이 방식을 쓰는가:
    momentum observer는 전류를 토크(Nm)로 정확히 환산해야 하는데, 그러려면
    모터 토크 상수와 기어비가 필요하다. component.py에 get_joint_torque()가
    있긴 하지만, 이 로봇은 "torque" 필드를 리포트하지 않아(전류만 리포트)
    호출하면 ValueError가 난다. 모터 토크 상수는 URDF에도 dexcontrol
    공개 예제에도 없어 확인되지 않았다. 부정확한 상수로 물리 공식에 억지로
    끼워 넣으면 "그럴듯해 보이지만 스케일이 틀린" 잔차가 나오는데, 이건
    아무 신호가 없는 것보다 더 위험하다 (틀린 확신을 주기 때문).

    대신, 이미 만들어둔 "작업별 적응형 기준선"(force_baseline.py) 패턴을
    전류값에 그대로 적용한다. 물리 상수가 전혀 필요 없고, 손목 F/T
    센서와 똑같은 원리로 "이 작업에서 이 관절의 정상 전류 수준을 벗어났는가"
    를 관절마다 독립적으로 판단할 수 있다. momentum observer보다 정밀하진
    않지만, 훨씬 적은 가정으로 "팔 중간 부분(어깨~손목)의 이상 부하"를
    감지할 수 있다 - 손목 F/T 센서 하나로는 못 보는 영역이다.

한계:
    - 절대 물리 한계(이 관절이 몇 A를 넘으면 위험한지)는 확인되지 않았다.
      URDF의 effort limit은 Nm 단위라 전류(A)와 직접 비교할 수 없다.
      모터 스펙시트나 Dexmate 쪽에 정격 전류를 문의하면 절대 하드리밋을
      추가할 수 있다. 지금은 상대적 편차(기준선 대비)만으로 판단한다.
    - 전류는 위치 제어 시 자세(중력 부하)에 따라서도 크게 달라진다.
      기준선이 "이 작업" 단위로 리셋되긴 하지만, 같은 작업 안에서도 팔의
      자세가 크게 바뀌면(예: 수평 뻗기 vs 수직 들기) 전류의 정상 범위
      자체가 달라질 수 있다. 오탐이 잦으면 stall_detector처럼 "움직임이
      거의 없는 구간"에서만 판단하도록 좁히는 것을 권장.
"""

from __future__ import annotations

import logging

import numpy as np

from config import ForceBaselineConfig
from force_baseline import VectorAdaptiveBaseline

logger = logging.getLogger(__name__)


class ArmCurrentAnomalyDetector:
    """한쪽 팔(7개 관절)의 전류를 감시해 관절별 이상 편차를 판단한다."""

    def __init__(self, cfg: ForceBaselineConfig, num_joints: int = 7) -> None:
        self._cfg = cfg
        self._baseline = VectorAdaptiveBaseline(cfg, dim=num_joints)

    def update_baseline(self, task_key: str, currents: np.ndarray) -> None:
        """정상으로 판단된 전류 샘플을 기준선에 반영한다.

        safety_supervisor에서 다른 이상 신호(카메라/힘)가 아무것도 안 잡힌
        틱에서만 호출해야 한다 - 이상 상황 자체를 정상으로 학습하지 않도록.
        """
        self._baseline.update(task_key, currents)

    def check(self, currents: np.ndarray) -> tuple[bool, np.ndarray | None, dict]:
        """현재 전류 벡터가 기준선 대비 이상인지 판단.

        Returns:
            (is_abnormal, deviation_ratios, debug_info)
            deviation_ratios는 기준선이 아직 준비 안 됐으면 None.
        """
        ratios = self._baseline.deviation_ratios(currents)

        if ratios is None:
            # 기준선 미준비 - fallback 절대값으로 대략적인 판단만 수행
            is_abnormal = bool(np.any(np.abs(currents) >= self._cfg.fallback_absolute_threshold_n))
            debug_info = {"baseline_ready": False}
            return is_abnormal, None, debug_info

        is_abnormal = bool(np.any(ratios >= self._cfg.deviation_ratio_threshold))
        debug_info = {
            "baseline_ready": True,
            "baseline": self._baseline.get_baseline(),
            "max_ratio": float(np.max(ratios)),
            "worst_joint_idx": int(np.argmax(ratios)),
        }
        return is_abnormal, ratios, debug_info
