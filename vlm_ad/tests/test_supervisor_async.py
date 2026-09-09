"""SafetySupervisor의 비동기 배선 검증 (VLM 서버/로봇 불필요 - 둘 다 가짜).

VLM 판정을 워커 스레드로 옮긴 뒤에도 정책이 그대로 동작하는지 본다:
  1. 후보 틱이 즉시 리턴하고, 판정은 몇 틱 뒤에 반영된다.
  2. 판정 대기 중 후보가 계속 잡히면 기존대로 감속을 유지한다.
  3. 연속 실패 카운터가 비동기에서도 동작해 fail-safe 정지로 이어진다.
  4. 작업차단 경로도 같은 구조로 동작한다.
  5. 절대 힘 한계는 VLM을 기다리지 않고 그 틱에서 즉시 정지한다.
"""

import sys
import time

from _harness import check, summary

import numpy as np
from config import (AnomalyFilterConfig, ForceBaselineConfig, SafetyPolicyConfig,
                    StallDetectorConfig, TaskFeasibilityPolicyConfig, VLMConfig)
from robot_interface import RobotInterface
from safety_supervisor import SafetySupervisor
from vlm_client import TaskFeasibilityVerdict, VLMVerdict
from vlm_worker import AsyncVerifier


class FakeRobot(RobotInterface):
    """조치 호출만 기록하는 가짜 로봇."""

    def __init__(self):
        self.calls = []

    def get_camera_frame(self): return np.zeros((8, 8, 3), dtype=np.uint8)
    def get_force_torque(self): return np.zeros(12)
    def get_current_task_description(self): return "테이블 위 물체를 오른팔로 집는 중"
    def emergency_stop(self, reason): self.calls.append(("estop", reason))
    def pause_task(self, reason): self.calls.append(("pause", reason))
    def slow_down(self, scale): self.calls.append(("slow", scale))
    def resume(self): self.calls.append(("resume", None))
    def is_stopped(self): return any(c[0] == "estop" for c in self.calls)
    def shutdown(self): pass
    def get_arm_joint_currents(self, side): return None
    def get_arm_joint_vel(self, side): return None


def build():
    robot = FakeRobot()
    return robot, SafetySupervisor(
        robot=robot,
        filter_cfg=AnomalyFilterConfig(),
        vlm_cfg=VLMConfig(),
        policy_cfg=SafetyPolicyConfig(),
        stall_cfg=StallDetectorConfig(),
        feasibility_policy_cfg=TaskFeasibilityPolicyConfig(),
        force_baseline_cfg=ForceBaselineConfig(),
    )


def anomaly_fail(reason):
    return VLMVerdict(False, 0.0, reason, ok=False)


def feasibility_fail(reason):
    return TaskFeasibilityVerdict(False, 0.0, reason, "needs_human_intervention", ok=False)


FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
FT = np.zeros(12)
DBG = {"flow_magnitude": 12.0}

# --- 1) 판정을 기다리는 동안 루프가 계속 돌고, 판정은 뒤늦게 반영된다 -------
print("[1] 이상 경로: 1.2초 걸리는 판정이 뒤늦게 정지로 이어지는가")
robot, sup = build()
sup._anomaly_worker = AsyncVerifier(
    "test",
    lambda f, c: (time.sleep(1.2), VLMVerdict(True, 0.9, "사람이 작업 반경 안에 있음", ok=True))[1],
    anomaly_fail, deadline_s=10.0,
)

t0 = time.monotonic()
sup._run_anomaly_path(FRAME, FT, is_candidate=True, debug_info=DBG)
submit_tick_ms = (time.monotonic() - t0) * 1000
check(f"후보 틱이 {submit_tick_ms:.1f}ms 만에 리턴 (동기였다면 1200ms)", submit_tick_ms < 20)
check("아직 아무 조치 없음 (판정 대기 중)", robot.calls == [])
check("busy() True", sup._anomaly_worker.busy() is True)

ticks, estop_at = 0, None
while time.monotonic() - t0 < 2.0:
    sup._run_anomaly_path(FRAME, FT, is_candidate=False, debug_info=DBG)
    ticks += 1
    if estop_at is None and any(c[0] == "estop" for c in robot.calls):
        estop_at = time.monotonic() - t0
    time.sleep(1 / 15)

check(f"2초 동안 {ticks}틱 도달 (루프 살아있음)", ticks >= 25)
check(f"판정 도착 후 정지 (t={estop_at:.2f}s)" if estop_at else "정지 안 됨",
      estop_at is not None and 1.1 < estop_at < 1.5)
check(f"정지 이유에 VLM 판단 근거 포함: {robot.calls[0][1]!r}",
      "사람이 작업 반경" in robot.calls[0][1])

# --- 2) 처리 중 후보가 계속 들어오면 기존 semantics(감속 유지)를 지킨다 -----
print("[2] 판정 대기 중 후보가 계속 잡히면 감속을 유지하는가")
robot2, sup2 = build()
sup2._anomaly_worker = AsyncVerifier(
    "test",
    lambda f, c: (time.sleep(1.0), VLMVerdict(False, 0.0, "정상", ok=True))[1],
    anomaly_fail, deadline_s=10.0,
)
sup2._run_anomaly_path(FRAME, FT, True, DBG)          # 제출
for _ in range(3):
    time.sleep(0.05)
    sup2._run_anomaly_path(FRAME, FT, True, DBG)      # 대기 중 재후보
slow_calls = [c for c in robot2.calls if c[0] == "slow"]
check(f"대기 중 재후보 3회 -> 감속 {len(slow_calls)}회", len(slow_calls) == 3)
check("중복 요청은 안 나감 (busy)", sup2._anomaly_worker.busy() is True)

# --- 3) VLM 연속 실패 -> fail-safe 정지 -------------------------------------
print("[3] 워커 실패가 연속 임계치(3회)를 넘으면 fail-safe 정지하는가")
robot3, sup3 = build()
sup3._anomaly_worker = AsyncVerifier(
    "test", lambda f, c: VLMVerdict(False, 0.0, "타임아웃", ok=False),
    anomaly_fail, deadline_s=10.0,
)
for _ in range(3):
    sup3._last_vlm_call_ts = 0.0                       # 쿨다운 무시
    sup3._run_anomaly_path(FRAME, FT, True, DBG)       # 제출
    time.sleep(0.05)
    sup3._run_anomaly_path(FRAME, FT, False, DBG)      # 수거
check(f"연속 실패 카운터 = {sup3._consecutive_vlm_failures}", sup3._consecutive_vlm_failures == 3)
estops = [c for c in robot3.calls if c[0] == "estop"]
check(f"fail-safe 정지 발생: {estops[0][1] if estops else None}",
      len(estops) == 1 and "판단 불가" in estops[0][1])

# --- 4) 작업차단 경로: 제출 성공 시에만 쿨다운(mark_checked)이 갱신된다 -----
print("[4] 작업차단 경로 배선")
robot4, sup4 = build()
sup4._feasibility_worker = AsyncVerifier(
    "test",
    lambda f, c: (time.sleep(0.5), TaskFeasibilityVerdict(
        True, 0.8, "적재함이 경로를 막고 있음", "needs_replanning", ok=True))[1],
    feasibility_fail, deadline_s=10.0,
)
t0 = time.monotonic()
sup4._run_task_feasibility_path(FRAME, 30.0, "물체 삽입 중", is_candidate=True, debug_info={})
tick_ms = (time.monotonic() - t0) * 1000
check(f"후보 틱이 {tick_ms:.1f}ms 만에 리턴", tick_ms < 20)
check("제출 시 stall 쿨다운 갱신됨", sup4._stall_detector.cooldown_ok() is False)

pause_at = None
t0 = time.monotonic()
while pause_at is None and time.monotonic() - t0 < 1.5:
    sup4._run_task_feasibility_path(FRAME, 5.0, "물체 삽입 중", False, {})
    if any(c[0] == "pause" for c in robot4.calls):
        pause_at = time.monotonic() - t0
    time.sleep(1 / 15)
check(f"판정 도착 후 pause_task (t={pause_at:.2f}s)" if pause_at else "pause_task 안 됨",
      pause_at is not None and pause_at < 0.9)
check(f"pause 이유: {robot4.calls[-1][1]!r}",
      "needs_replanning" in robot4.calls[-1][1] and "적재함" in robot4.calls[-1][1])

# --- 5) 절대 힘 한계는 VLM을 기다리지 않고 즉시 정지 -----------------------
print("[5] 절대 힘 한계 초과는 VLM 대기 없이 즉시 정지하는가")
robot5, sup5 = build()
t0 = time.monotonic()
sup5._run_task_feasibility_path(FRAME, 45.0, "물체 삽입 중", is_candidate=True, debug_info={})
check(f"{(time.monotonic()-t0)*1000:.1f}ms 안에 즉시 정지",
      any(c[0] == "estop" for c in robot5.calls) and (time.monotonic() - t0) < 0.02)
check("VLM 요청은 안 나감", sup5._feasibility_worker.busy() is False)

sys.exit(summary())
