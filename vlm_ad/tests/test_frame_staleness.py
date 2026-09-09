"""카메라 프레임 신선도 가드 검증 (VLM 서버/로봇 불필요 - 둘 다 가짜).

정책 확인: 프레임이 오래되면 로봇을 정지시키지 않고, 시각 기반 신호만
무효로 하면서(광류 트리거, stall 후보, VLM 판정) 힘/전류 경로로 계속
감시하고 경고를 남긴다.

주의: 합성 프레임의 광류는 1.3 수준이라 실제 임계값 8.0을 못 넘는다
(Farneback은 장면 구조에 크게 의존한다 - 실제 임계값 튜닝은
calibrate_filter.py로 실제 카메라 프레임을 써야 한다). 여기서는 가드
로직만 보는 것이므로 임계값을 낮춰서 광류 트리거가 발동하게 만든다.
"""

import dataclasses
import logging
import sys

from _harness import check, summary

import numpy as np
from config import (AnomalyFilterConfig, ForceBaselineConfig, SafetyPolicyConfig,
                    StallDetectorConfig, TaskFeasibilityPolicyConfig, VLMConfig)
from robot_interface import RobotInterface
from safety_supervisor import SafetySupervisor
from vlm_client import TaskFeasibilityVerdict, VLMVerdict
from vlm_worker import AsyncVerifier

logging.basicConfig(level=logging.WARNING, format="      %(levelname)-8s %(message)s")

FILTER_CFG = dataclasses.replace(AnomalyFilterConfig(), flow_magnitude_threshold=0.5)
BASE = np.random.default_rng(0).integers(0, 255, (600, 960, 3), dtype=np.uint8)


class FakeRobot(RobotInterface):
    """프레임 나이와 힘을 테스트가 직접 조종하는 가짜 로봇."""

    def __init__(self):
        self.age = 0.05          # 정상 스트림 수준 (실측 p50 6.5ms)
        self.force = np.zeros(12)
        self.shift = 0
        self.calls = []

    def get_camera_frame(self):
        self.shift += 40         # 매 틱 큰 움직임 -> 광류 스파이크
        return np.roll(BASE, self.shift, axis=1)

    def get_camera_frame_age_s(self): return self.age
    def get_force_torque(self): return self.force
    def get_current_task_description(self): return "테스트 작업"
    def emergency_stop(self, reason): self.calls.append(("estop", reason))
    def pause_task(self, reason): self.calls.append(("pause", reason))
    def slow_down(self, scale): self.calls.append(("slow", scale))
    def resume(self): pass
    def is_stopped(self): return False
    def shutdown(self): pass
    def get_arm_joint_currents(self, side): return None
    def get_arm_joint_vel(self, side): return None


def build(robot):
    sup = SafetySupervisor(
        robot=robot, filter_cfg=FILTER_CFG, vlm_cfg=VLMConfig(),
        policy_cfg=SafetyPolicyConfig(), stall_cfg=StallDetectorConfig(),
        feasibility_policy_cfg=TaskFeasibilityPolicyConfig(),
        force_baseline_cfg=ForceBaselineConfig(),
    )
    # VLM 서버를 부르지 않도록 두 워커를 즉답 스텁으로 교체
    sup._anomaly_worker = AsyncVerifier(
        "stub", lambda f, c: VLMVerdict(True, 0.9, "스텁 이상판정", ok=True),
        lambda r: VLMVerdict(False, 0.0, r, ok=False), deadline_s=5.0)
    sup._feasibility_worker = AsyncVerifier(
        "stub", lambda f, c: TaskFeasibilityVerdict(False, 0.0, "스텁", "wait_and_retry", ok=True),
        lambda r: TaskFeasibilityVerdict(False, 0.0, r, "needs_human_intervention", ok=False),
        deadline_s=5.0)
    return sup


print(f"설정: max_frame_age_s={FILTER_CFG.max_frame_age_s} "
      f"warn_interval={FILTER_CFG.stale_warn_interval_s}s "
      f"flow_thr={FILTER_CFG.flow_magnitude_threshold}(합성프레임용으로 낮춤)")

print("\n[1] 신선한 프레임: 시각 경로 정상 동작")
r = FakeRobot()
sup = build(r)
for _ in range(20):
    sup._tick()                          # 과도구간 15틱 통과 후 광류 후보 발생
check("VLM 요청이 나갔다 (시각 경로 살아있음)", sup._last_vlm_call_ts > 0)

print("\n[2] 오래된 프레임: 광류 후보 억제 + VLM 요청 안 나감 (경고만)")
r2 = FakeRobot()
sup2 = build(r2)
r2.age = 2.0                             # 2초 전 프레임
for _ in range(30):
    sup2._tick()
check("VLM 요청 0건 (오래된 장면을 판정하지 않음)", sup2._last_vlm_call_ts == 0.0)
check("로봇에 어떤 조치도 안 함 (정지 아님, 경고만)", r2.calls == [])
check("stale 상태로 기록됨", sup2._stale_since is not None)

print("\n[3] 오래된 프레임인데 힘 스파이크: 힘 경로는 살아있는가")
r3 = FakeRobot()
sup3 = build(r3)
r3.age = 2.0
sup3._tick()
r3.force = np.array([0, 0, 25.0, 0, 0, 0] + [0] * 6)   # 25N 급변 (임계 15N)
sup3._tick()
check("힘 스파이크는 후보로 인식되지만 VLM 요청은 억제", sup3._last_vlm_call_ts == 0.0)
check("정지는 없음 (힘 스파이크만으로는 정지 안 함)", r3.calls == [])

print("\n[4] 오래된 프레임 + 절대 힘 한계 초과: 정지하는가 (핵심)")
r4 = FakeRobot()
sup4 = build(r4)
r4.age = 5.0                                            # 카메라 완전 정지
r4.force = np.array([0, 0, 45.0, 0, 0, 0] + [0] * 6)    # 45N > 한계 40N
sup4._tick()
estops = [c for c in r4.calls if c[0] == "estop"]
check(f"카메라가 죽었어도 힘 절대 한계로 정지: {estops[0][1] if estops else None}",
      len(estops) == 1 and "절대 안전 한계" in estops[0][1])

print("\n[5] 정지 요청이 매 틱 반복되지 않는가 (래치)")
for _ in range(20):
    sup4._tick()
check(f"20틱 더 돌려도 estop 요청 1건 유지 (실제 {len([c for c in r4.calls if c[0]=='estop'])}건)",
      len([c for c in r4.calls if c[0] == "estop"]) == 1)
r4.force = np.zeros(12)
sup4._tick()                                            # 힘 정상화 -> 재무장
r4.force = np.array([0, 0, 45.0, 0, 0, 0] + [0] * 6)
sup4._tick()
check("힘이 내려갔다 다시 초과하면 재요청", len([c for c in r4.calls if c[0] == "estop"]) == 2)

print("\n[6] 신선도 복구 시 시각 경로가 재개되는가")
r6 = FakeRobot()
sup6 = build(r6)
r6.age = 2.0
for _ in range(20):
    sup6._tick()
stale_flag = sup6._stale_since is not None
r6.age = 0.05                                           # 스트림 복구
for _ in range(20):
    sup6._tick()
check("복구 전 stale, 복구 후 해제", stale_flag and sup6._stale_since is None)
check("복구 후 VLM 요청 재개", sup6._last_vlm_call_ts > 0)

print("\n[7] age를 모르는 어댑터(None)에서는 가드가 발동하지 않는가")


class NoAgeRobot(FakeRobot):
    def get_camera_frame_age_s(self): return None


r7 = NoAgeRobot()
sup7 = build(r7)
for _ in range(20):
    sup7._tick()
check("None이면 신선한 것으로 취급 -> 시각 경로 유지", sup7._last_vlm_call_ts > 0)
check("stale 경고도 안 남음", sup7._stale_since is None)

sys.exit(summary())
