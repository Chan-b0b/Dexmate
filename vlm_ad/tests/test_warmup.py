"""SafetySupervisor.warm_up() 검증.

**실제 vLLM 서버가 필요하다** (없으면 SKIP). 로봇은 가짜다.

확인하는 것:
  1. 두 경로 워밍업이 정상 완료되고, 로봇에는 어떤 조치도 하지 않는다.
  2. 워밍업만 긴 타임아웃(warmup_timeout_s)을 쓴다 - 콜드 스타트가
     request_timeout_s를 넘기기 때문에 이게 핵심이다.
  3. 모델 id가 틀리면 기동 시점에 잡아낸다 (자기진단).
  4. 카메라 프레임을 못 받으면 예외를 던지지 않고 False를 돌려준다.
  5. verify()에 timeout_s 인자를 추가한 뒤에도 워커 경로가 동작한다.

주의: 3번 항목은 **의도적으로 존재하지 않는 모델 이름으로 요청을 보낸다.**
그래서 이 테스트를 돌리면 vLLM 서버 콘솔에 아래 로그가 실행 1회당 2건
(이상/작업차단 두 경로) 찍힌다 - 정상이며 무해하다:
    ERROR ... error=ErrorInfo(message='The model `<이름>` does not exist.',
                              type='NotFoundError', param='model', code=404)
    INFO: ... "POST /v1/chat/completions HTTP/1.1" 404 Not Found
그래서 모델 이름을 로그만 보고도 알아볼 수 있게 지어 놨다.
"""

import dataclasses
import logging
import sys
import time

from _harness import check, load_fixture_frame, skip_without_vlm_server, summary

import numpy as np
from config import (AnomalyFilterConfig, ForceBaselineConfig, SafetyPolicyConfig,
                    StallDetectorConfig, TaskFeasibilityPolicyConfig, VLMConfig)
from robot_interface import RobotInterface
from safety_supervisor import SafetySupervisor

logging.basicConfig(level=logging.INFO, format="      %(levelname)-8s %(message)s")
skip_without_vlm_server()

# vLLM 서버 로그에 남을 이름이므로, 로그만 보고 이 테스트가 원인임을 알 수 있게 짓는다.
BOGUS_MODEL = "intentionally-missing-model-warmup-selftest"

SCENE = load_fixture_frame()   # 실제 head_camera 프레임 (960x600)


class FakeRobot(RobotInterface):
    def __init__(self, camera_fails=False):
        self.camera_fails = camera_fails
        self.calls = []

    def get_camera_frame(self):
        if self.camera_fails:
            raise RuntimeError("카메라 스트림 비활성")
        return SCENE

    def get_force_torque(self): return np.zeros(12)
    def get_current_task_description(self): return "배터리 케이스를 빈에서 오른팔로 집는 중"
    def emergency_stop(self, reason): self.calls.append(("estop", reason))
    def pause_task(self, reason): self.calls.append(("pause", reason))
    def slow_down(self, scale): self.calls.append(("slow", scale))
    def resume(self): pass
    def is_stopped(self): return False
    def shutdown(self): pass
    def get_arm_joint_currents(self, side): return None
    def get_arm_joint_vel(self, side): return None


def build(robot, vlm_cfg):
    return SafetySupervisor(
        robot=robot, filter_cfg=AnomalyFilterConfig(), vlm_cfg=vlm_cfg,
        policy_cfg=SafetyPolicyConfig(), stall_cfg=StallDetectorConfig(),
        feasibility_policy_cfg=TaskFeasibilityPolicyConfig(),
        force_baseline_cfg=ForceBaselineConfig(),
    )


# --- 1) 정상 워밍업 ---------------------------------------------------------
print("[1] 실제 서버 대상 워밍업 (두 경로 모두)")
robot = FakeRobot()
sup = build(robot, VLMConfig())
t0 = time.monotonic()
result = sup.warm_up()
elapsed = time.monotonic() - t0
check(f"warm_up() True 반환 ({elapsed:.2f}s, 두 경로 합산)", result is True)
check("워밍업이 로봇에 조치를 취하지 않음 (판정 결과는 버린다)", robot.calls == [])

# --- 2) 타임아웃 오버라이드가 실제로 먹는지 --------------------------------
print("[2] 워밍업만 긴 타임아웃을 쓰는가 (client.with_options 검증)")
# request_timeout_s를 0.1s로 낮추면 일반 판정은 반드시 타임아웃난다.
# 워밍업이 warmup_timeout_s를 쓴다면 같은 조건에서도 성공해야 한다.
tight = dataclasses.replace(VLMConfig(), request_timeout_s=0.1, warmup_timeout_s=30.0)
sup2 = build(FakeRobot(), tight)

normal = sup2._vlm.verify(SCENE, "테스트")                       # 기본 타임아웃 0.1s
check(f"일반 판정은 0.1s에서 타임아웃: ok={normal.ok} reason={normal.reason!r}",
      normal.ok is False and normal.reason == "타임아웃")

t0 = time.monotonic()
warm = sup2._vlm.verify(SCENE, "테스트", timeout_s=tight.warmup_timeout_s)
check(f"timeout_s 지정 시 성공: ok={warm.ok} ({time.monotonic()-t0:.2f}s, 0.1s 예산 초과)",
      warm.ok is True)
check("워밍업 경로 전체도 성공", sup2.warm_up() is True)

# --- 3) 모델 id가 틀리면 기동 시점에 잡아내는가 (자기진단) -----------------
print("[3] 모델 id 오타를 기동 시점에 잡아내는가")
print(f"     (서버 로그에 404 2건이 의도적으로 찍힌다: model={BOGUS_MODEL})")
wrong = dataclasses.replace(VLMConfig(), model=BOGUS_MODEL)
sup3 = build(FakeRobot(), wrong)
check("warm_up() False 반환 (위 CRITICAL 로그 참고)", sup3.warm_up() is False)

# --- 4) 카메라가 죽어 있으면 워밍업을 건너뛰고 기동은 계속 ----------------
print("[4] 카메라 프레임을 못 받으면")
print("     (아래 traceback은 의도된 것 - 예외를 흡수하는지 본다)")
sup4 = build(FakeRobot(camera_fails=True), VLMConfig())
check("예외를 밖으로 던지지 않고 False 반환", sup4.warm_up() is False)

# --- 5) verify 시그니처 변경이 AsyncVerifier와 호환되는가 ------------------
print("[5] 워커 경로가 시그니처 변경 후에도 동작하는가")
robot5 = FakeRobot()
sup5 = build(robot5, VLMConfig())
FT = np.zeros(12)
sup5._run_anomaly_path(SCENE, FT, is_candidate=True, debug_info={"flow_magnitude": 12.0})
check("후보 틱이 즉시 리턴 + 요청 제출됨", sup5._anomaly_worker.busy() is True)

t0, verdict_seen = time.monotonic(), False
while time.monotonic() - t0 < 10.0:
    sup5._run_anomaly_path(SCENE, FT, False, {"flow_magnitude": 0.0})
    if sup5._consecutive_vlm_failures == 0 and not sup5._anomaly_worker.busy():
        verdict_seen = True
        break
    time.sleep(1 / 15)
check(f"판정 수거 완료 ({time.monotonic()-t0:.2f}s), 실패 카운터 0", verdict_seen)
check("정상 장면이므로 정지 안 걸림", robot5.calls == [])

sys.exit(summary())
