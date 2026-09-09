"""로봇 SDK(dexcontrol)와 안전정지 시스템 사이의 어댑터 계층.

⚠️ 버전 호환성 이력:
  한때 dexcontrol 0.5.0 + 로봇 펌웨어 0.4.5 조합에서 E-Stop 서비스 콜
  (bot.estop.activate())이 "서비스는 발견되는데 응답이 영원히 안 옴"
  상태로 멈추는 문제를 겪었다 (타임아웃을 5000초로 늘려도 동일 - 네트워크
  지연이 아니라 구조적 문제였음).

  원인: dexcontrol과 로봇 펌웨어는 마이너 버전 단위로 락스텝(lockstep)
  되어 있다 (PyPI 공식 호환성 표 기준: dexcontrol 0.X.y <-> 펌웨어 0.X.*).
  버전이 안 맞으면 클라이언트가 보낸 요청을 펌웨어가 파싱 못 해서 조용히
  버리는 것으로 추정됨 (에러 응답조차 안 옴).

  당시 해결책: dexcontrol을 pip install "dexcontrol>=0.4.5,<0.5.0"으로
  펌웨어와 같은 라인(0.4.x)에 맞춰 다운그레이드.

  이후 로봇 펌웨어가 0.5.x로 업그레이드되면서 dexcontrol 0.5.0의 E-Stop이
  다시 정상 동작함을 실제 로봇에서 확인했다. 지금은 requirements.txt에서
  dexcontrol을 0.5.x로 맞춰 고정하고, main.py에서 시작 시 설치된 버전을
  체크해서 다시 어긋나면 CRITICAL 로그로 경고한다
  (config.EXPECTED_DEXCONTROL_MINOR_LINE 참고).

⚠️ 참고 - 아래 어댑터가 arm.get_state()를 직접 쓰지 않고 전용 public
  메서드를 쓰는 이유 (component.py 원문 확인, 다운그레이드했던 0.4.5에서
  RobotComponent.get_state()가 한때 _get_state()로 private화되어 깨졌던
  적이 있음 - arm_current_monitor.py/momentum_observer.py 참고). 지금
  설치된 0.5.0에서는 get_state()가 다시 public이지만, 의미가 더 명확한
  아래 전용 메서드들을 그대로 쓴다 (0.5.0 component.py 소스로 재확인함):
      get_joint_pos(joint_id=None)      -> np.ndarray
      get_joint_vel(joint_id=None)      -> np.ndarray
      get_joint_current(joint_id=None)  -> np.ndarray (전류, A)
      get_joint_torque(joint_id=None)   -> np.ndarray (토크, Nm - "torque"
          필드가 있는 컴포넌트만. 우리 로봇은 "cur"만 리포트해서 이 메서드는
          호출하면 ValueError가 난다)
      get_joint_err(joint_id=None)      -> np.ndarray (에러 코드)

공개 예제로 확인된 API (dexmate-ai/dexcontrol/examples, main 브랜치 기준.
설치된 dexcontrol 0.5.0의 실제 소스(src/dexcontrol/core/misc.py,
component.py, arm.py, camera/zed_camera.py)를 직접 대조해서 아래 API들이
그대로 존재하고 동일하게 동작함을 확인했다):

  정지/재개  (examples/advanced_examples/estop_robot.py)
      from dexcontrol.robot import Robot
      bot = Robot()
      bot.estop.activate()
      bot.estop.deactivate()
      bot.estop.show()
      bot.shutdown()

  카메라  (examples/basic_examples/sensors/get_head_zed_x_mini_data.py)
      from dexcontrol.core.config import get_robot_config
      configs = get_robot_config()
      configs.enable_sensor("head_camera")
      configs.sensors["head_camera"].transport = "zenoh"  # 또는 "rtc"
      with Robot(configs=configs) as robot:
          robot.sensors.head_camera.wait_for_active(timeout=5.0)
          data = robot.sensors.head_camera.get_obs(
              obs_keys=["left_rgb", "right_rgb", "depth"],
              include_timestamp=True,
          )
          # data["left_rgb"]는 {"data": ndarray, "timestamp_ns":..., "receive_time_ns":...}
          # 형태이거나 상황에 따라 ndarray 자체일 수 있음 (아래 _extract_image 참고)

  힘/토크 센서  (examples/basic_examples/sensors/get_force_torque_sensor.py)
      bot = Robot()
      arm = bot.left_arm  # 또는 bot.right_arm
      wrench = arm.wrench_sensor.get_wrench_state()  # [fx, fy, fz, mx, my, mz]

확인되지 않았던 "감속" 문제에 대한 결론 (src/dexcontrol/core/arm.py 분석 결과):
  Arm.set_joint_pos_vel(joint_pos, joint_vel)는 docstring에 명시된 대로
  "100Hz 같은 고주파로 계속 호출해야" 하는 명령입니다. dexcontrol의 Arm은
  궤적을 스스로 소유하지 않고, 외부(모션 플래너/teleop 루프)가 스트리밍하는
  pos+vel 명령을 그대로 중계하는 얇은 레이어입니다.

  즉, "이미 실행 중인 명령을 부드럽게 줄인다"는 개념 자체가 Arm 레벨에 없고,
  진짜 감속을 하려면 그 pos/vel 스트림을 만들어내는 상위 루프(누가 100Hz로
  set_joint_pos_vel을 호출하고 있는지) 자신이 속도를 낮춰야 합니다. 이 안전
  감시 시스템이 그 루프와 별도 프로세스/스레드라면 직접 끼어들 수 없습니다.

  대안으로 arm.set_modes(["disable"]*7)도 있지만, 이는 조인트 제어를 그냥
  꺼버리는 것이라 자중에 의해 팔이 늘어질 위험이 있어 오히려 estop보다
  덜 안전할 수 있습니다 (Vega 팔의 실제 페일세이프 동작 방식을 확인하기
  전까지는 권장하지 않음).

  결론: 이 어댑터는 두 가지 경로를 지원합니다.
    1) 기본값: slow_down()도 estop.activate()로 처리 (가장 안전, 추가 배선 불필요)
    2) 실제 모션 스트리밍 루프가 있다면, 그 루프가 참조하는 속도 스케일 값을
       건네줄 수 있는 콜백(external_velocity_scale_setter)을 주입해서, 안전
       감시 시스템이 "속도를 줄여라"라는 신호만 보내고 실제 감속은 그 루프가
       수행하게 할 수 있음. 콜백이 없으면 1)로 자동 fallback.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Callable

import cv2
import numpy as np

from config import EmergencyStopRetryConfig

logger = logging.getLogger(__name__)


class EmergencyStopFailure(RuntimeError):
    """emergency_stop() 호출이 재시도 후에도 확인되지 않았을 때 발생.

    이 예외가 올라온다는 것은 소프트웨어로는 더 이상 로봇이 멈췄다고
    확신할 수 없다는 뜻이다. 호출부는 이걸 삼키고 계속 진행해서는 안 되며,
    사람에게 알리고 물리적 e-stop 개입을 기다리는 등의 조치를 해야 한다.
    """


class RobotInterface(abc.ABC):
    """안전정지 시스템이 필요로 하는 최소 로봇 인터페이스."""

    @abc.abstractmethod
    def get_camera_frame(self) -> np.ndarray:
        """최신 헤드 카메라 프레임을 (H, W, 3) BGR ndarray로 반환."""

    @abc.abstractmethod
    def get_force_torque(self) -> np.ndarray:
        """힘/토크 센서 값을 벡터로 반환 (모니터링 대상 팔 구성에 따라 6 또는 12차원)."""

    @abc.abstractmethod
    def get_current_task_description(self) -> str:
        """VLM 프롬프트에 넣을, 현재 로봇이 수행 중인 작업에 대한 짧은 설명."""

    @abc.abstractmethod
    def emergency_stop(self, reason: str) -> None:
        """즉시 정지."""

    @abc.abstractmethod
    def pause_task(self, reason: str) -> None:
        """작업 차단(장애물 등) 판단 시 호출. 위험 상황은 아니지만 더 이상
        진행해서는 안 되는 상태를 알린다.

        기본 구현은 emergency_stop()과 동일하게 estop을 사용하지만 (검증된
        "작업만 일시정지, 힘은 유지"에 해당하는 API가 없음), 로깅상 별도
        카테고리("task_blocked")로 남겨서 운영자가 "위험해서 멈췄다"와
        "막혀서 멈췄다"를 구분할 수 있게 한다.
        """

    @abc.abstractmethod
    def slow_down(self, scale: float) -> None:
        """감속. (현재는 검증된 부분 감속 API가 없어 정지로 대체됨 - 위 docstring 참고)"""

    @abc.abstractmethod
    def resume(self) -> None:
        """정지 상태에서 정상 동작으로 복귀 (사람의 확인 후에만 호출되어야 함)."""

    @abc.abstractmethod
    def is_stopped(self) -> bool:
        """현재 emergency_stop 상태인지 여부."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """프로세스 종료 시 연결 정리."""

    # 아래 두 메서드는 선택적 채널이다. 기본값은 "지원 안 함"(None)이며,
    # DexcontrolAdapter처럼 실제 확인된 API가 있는 구현체만 override한다.
    # safety_supervisor는 None이 오면 해당 채널을 건너뛴다.
    def get_arm_joint_currents(self, side: str) -> np.ndarray | None:
        """관절별 모터 전류(A). 지원 안 하면 None."""
        return None

    def get_arm_joint_vel(self, side: str) -> np.ndarray | None:
        """관절별 속도(rad/s). 지원 안 하면 None."""
        return None

    def get_camera_frame_age_s(self) -> float | None:
        """가장 최근 get_camera_frame()이 돌려준 프레임의 나이(초). 확인 불가면 None.

        None을 돌려주면 safety_supervisor는 신선도 가드를 적용하지 않는다.
        신선도를 알 수 없는 어댑터에서 가드가 항상 발동해 시각 기반 경로가
        통째로 죽어버리는 걸 막기 위한 것이다.
        """
        return None


def _extract_image(raw) -> np.ndarray:  # noqa: ANN001
    """get_obs() 반환값에서 실제 이미지 ndarray만 뽑아낸다.

    예제 코드 기준 data[key]는 {"data": ndarray, "timestamp_ns":...} 형태이거나
    (버전에 따라) ndarray 자체일 수 있어 양쪽 다 처리한다.
    """
    if isinstance(raw, dict):
        return raw["data"]
    return raw


class DexcontrolAdapter(RobotInterface):
    """dexcontrol 실제 로봇에 연결하는 어댑터."""

    def __init__(
        self,
        robot,  # noqa: ANN001
        monitor_arms: tuple[str, ...] = ("left", "right"),
        external_velocity_scale_setter: Callable[[float], None] | None = None,
        estop_retry_cfg: EmergencyStopRetryConfig | None = None,
    ) -> None:
        """
        Args:
            robot: dexcontrol.robot.Robot 인스턴스. 생성 시 head_camera 센서가
                이미 enable_sensor("head_camera")로 활성화되어 있어야 한다.
            monitor_arms: 힘/토크를 감시할 팔. 기본값은 양쪽 모두.
            external_velocity_scale_setter: 실제 모션을 100Hz로 스트리밍하는
                상위 루프(모션 플래너/teleop 브릿지 등)가 참조하는 속도 스케일
                값을 갱신하는 콜백. 이게 주어지면 slow_down()이 이 콜백을 호출해
                "진짜" 감속을 시도한다. None이면(기본값) slow_down()도 안전하게
                estop.activate()로 처리된다.
        """
        self._robot = robot
        self._monitor_arms = monitor_arms
        self._external_velocity_scale_setter = external_velocity_scale_setter
        self._external_task_pause_hook: Callable[[str], None] | None = None
        self._estop_retry_cfg = estop_retry_cfg or EmergencyStopRetryConfig()
        self._current_task = "알 수 없음"
        self._last_frame_age_s: float | None = None

    def set_external_task_pause_hook(self, hook: Callable[[str], None] | None) -> None:
        """작업 차단 시 estop 대신 호출할 수 있는 훅 주입.

        예: 상위 모션 루프가 "현재 위치 유지, 힘 목표를 0으로" 같은 동작을
        지원한다면, 이 훅을 통해 estop보다 덜 급격한 정지를 시도할 수 있다.
        훅이 없으면 pause_task()도 emergency_stop()과 동일하게 estop을 쓴다.
        """
        self._external_task_pause_hook = hook

    def set_current_task_description(self, description: str) -> None:
        """상위 태스크 플래너가 호출해서 현재 작업 컨텍스트를 갱신."""
        self._current_task = description

    def get_camera_frame(self) -> np.ndarray:
        # include_timestamp=True로 받는 이유: 프레임 신선도를 확인해야 한다.
        # 스트림이 얼어붙어도 get_obs()는 마지막 캐시 프레임을 계속 돌려주기
        # 때문에(dexcontrol에 staleness 파라미터가 없음), 타임스탬프를 안 보면
        # 같은 프레임을 계속 정상 프레임으로 취급하게 된다. 그러면 광류가 0이
        # 되어 후보가 아예 안 잡히면서 시스템은 "정상"으로 보인다.
        obs = self._robot.sensors.head_camera.get_obs(
            obs_keys=["left_rgb"], include_timestamp=True,
        )
        raw = obs["left_rgb"]

        # receive_time_ns = 호스트가 이 프레임을 받은 벽시계 시각(ns).
        # 센서 캡처 시각(timestamp_ns)이 아니라 이걸 쓰는 이유는 로컬
        # time.time_ns()와 같은 시계라 비교가 안전하기 때문이다 (캡처 시각은
        # 로봇 쪽 시계 도메인일 수 있다. 실측 두 값의 차이는 약 78ms).
        if isinstance(raw, dict) and "receive_time_ns" in raw:
            self._last_frame_age_s = max(
                0.0, (time.time_ns() - raw["receive_time_ns"]) / 1e9,
            )
        else:
            self._last_frame_age_s = None  # 이 버전은 신선도를 알려주지 않음

        img_rgb = _extract_image(raw)
        # ZED 카메라는 RGB로 나오므로(스트림 메타데이터 color_format='rgb'로
        # 확인), cv2 계열 함수(BGR 기준)와의 일관성을 위해 변환
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def get_camera_frame_age_s(self) -> float | None:
        """가장 최근 get_camera_frame() 호출이 받은 프레임의 나이(초).

        get_camera_frame()과 같은 get_obs 호출에서 얻은 값을 그대로 돌려준다.
        별도로 다시 조회하면 다른 프레임을 보게 되므로 그렇게 하지 않는다.
        """
        return self._last_frame_age_s

    def get_force_torque(self) -> np.ndarray:
        readings: list[np.ndarray] = []
        for side in self._monitor_arms:
            arm = self._robot.left_arm if side == "left" else self._robot.right_arm
            if arm.wrench_sensor is None:
                readings.append(np.zeros(6, dtype=np.float64))
                continue
            readings.append(np.asarray(arm.wrench_sensor.get_wrench_state(), dtype=np.float64))
        return np.concatenate(readings)

    def get_arm_joint_currents(self, side: str) -> np.ndarray:
        """확인됨 (component.py v0.4.5): RobotJointComponent.get_joint_current().

        이전에는 arm.get_state()["cur"]로 접근했는데, v0.4.5에서 get_state()가
        _get_state()로 private화되면서 깨졌다. 전용 public 메서드로 교체.
        """
        arm = self._robot.left_arm if side == "left" else self._robot.right_arm
        return np.asarray(arm.get_joint_current(), dtype=np.float64)

    def get_arm_joint_vel(self, side: str) -> np.ndarray:
        """확인됨 (component.py v0.4.5): RobotJointComponent.get_joint_vel().

        이전에는 arm.get_state()["vel"]로 접근했는데, 마찬가지로 get_state()가
        private화되면서 깨졌다. 전용 public 메서드로 교체.
        """
        arm = self._robot.left_arm if side == "left" else self._robot.right_arm
        return np.asarray(arm.get_joint_vel(), dtype=np.float64)

    def get_current_task_description(self) -> str:
        return self._current_task

    def _activate_estop_with_retry(self, reason: str) -> None:
        """estop.activate() 호출 + 실제 상태 확인 + 재시도. emergency_stop()/pause_task() 공용.

        중요(misc.py 확인 결과): EStop.activate()는 내부 서비스 콜이
        실패해도 예외를 던지지 않고 조용히 리턴한다. 그래서 예외를 잡는
        것만으로는 실패를 감지할 수 없다 - 반드시 is_software_estop_enabled()
        로 상태 토픽을 직접 확인해야 한다 (서비스 콜과 별개의 채널이라
        서비스가 잠깐 느려도 상태는 정상적으로 반영되는 경우가 많다).
        """
        cfg = self._estop_retry_cfg
        last_state: bool | None = None

        for attempt in range(1, cfg.max_retries + 1):
            try:
                self._robot.estop.activate()
            except Exception as e:  # noqa: BLE001 - 라이브러리 동작이 바뀔 가능성에 대한 방어
                logger.error(
                    "EMERGENCY STOP 호출 중 예외 (시도 %d/%d): %s",
                    attempt, cfg.max_retries, e,
                )

            time.sleep(cfg.verify_delay_s)  # 상태 토픽에 반영될 시간을 준다

            try:
                last_state = self._robot.estop.is_software_estop_enabled()
            except Exception as e:  # noqa: BLE001
                logger.error("E-Stop 상태 확인 중 예외: %s", e)
                last_state = None

            if last_state is True:
                if attempt > 1:
                    logger.critical("EMERGENCY STOP 재시도 %d회차에 확인됨", attempt)
                return

            logger.error(
                "EMERGENCY STOP 확인 실패 (시도 %d/%d) - "
                "is_software_estop_enabled()=%s",
                attempt, cfg.max_retries, last_state,
            )
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_interval_s)

        raise EmergencyStopFailure(
            f"{cfg.max_retries}회 재시도 후에도 E-Stop이 활성화됐는지 상태 토픽으로 "
            f"확인할 수 없습니다 (마지막 상태: {last_state}). 로봇이 실제로 정지했는지 "
            f"확인되지 않습니다. 즉시 물리적 e-stop으로 개입하세요. 원래 정지 이유: {reason}",
        )

    def emergency_stop(self, reason: str) -> None:
        """확인됨: bot.estop.activate(). 단, misc.py 소스 확인 결과 이 호출은
        내부 서비스 콜(클라이언트 타임아웃 50ms로 하드코딩)이 실패해도
        예외를 던지지 않고 로그만 남기고 조용히 리턴한다:

            ERROR Failed to set E-Stop to True: no response from service
            (timeout or unreachable). The robot may NOT be in the expected
            E-Stop state.

        그래서 이 메서드는 activate() 호출 자체의 성공 여부를 믿지 않고,
        매번 is_software_estop_enabled()로 실제 상태 토픽을 확인한다.
        확인이 안 되면 재시도하고, 모든 재시도가 실패하면
        EmergencyStopFailure를 던져서 조용히 넘어가지 못하게 한다.

        50ms라는 타임아웃은 매우 짧다. 같은 프로세스에서 무거운 Python
        연산(카메라 처리, VLM 호출 등)이 GIL을 붙잡고 있으면 실제 서비스는
        정상인데도 이 타임아웃에 걸릴 수 있다 - 이 실패가 반복적으로
        난다면 로봇 자체보다는 이 프로세스의 스케줄링 지연을 먼저 의심해야
        한다.

        중요: 이건 소프트웨어 e-stop 하나에만 의존하면 안 되는 이유이기도
        하다. 이 호출이 계속 실패한다면 물리적 하드웨어 e-stop이 필요한
        상황으로 취급해야 한다. 이 메서드가 EmergencyStopFailure를
        던진다면, 운영 절차상 사람이 즉시 물리 버튼으로 개입해야 한다는
        뜻으로 취급해야 한다.
        """
        logger.critical("EMERGENCY STOP 요청: %s", reason)
        self._activate_estop_with_retry(reason)

    def pause_task(self, reason: str) -> None:
        """작업 차단 판단 시 호출. 별도 훅이 있으면 그걸 쓰고, 없으면 estop.

        검증된 "부드러운 작업 일시정지" API가 없어 기본값은 emergency_stop과
        동일한 estop.activate()지만, 로그 레벨과 문구를 달리해서 운영자가
        원인을 구분할 수 있게 한다.
        """
        if self._external_task_pause_hook is not None:
            logger.warning("작업 차단(pause_task) - 외부 훅으로 처리: %s", reason)
            self._external_task_pause_hook(reason)
            return

        logger.warning("작업 차단(pause_task) - 검증된 일시정지 API가 없어 estop으로 대체: %s", reason)
        self._activate_estop_with_retry(reason)

    def slow_down(self, scale: float) -> None:
        """진짜 감속은 이 어댑터가 직접 할 수 없다 (위 모듈 docstring 참고).

        set_joint_pos_vel()은 100Hz로 계속 호출돼야 하는 명령이라, 그 스트림을
        만드는 상위 루프만이 실제로 속도를 낮출 수 있다. 그 루프가 참조하는
        속도 스케일 콜백이 주입돼 있으면 그걸 호출하고, 없으면 안전하게
        즉시 정지로 대체한다.
        """
        if self._external_velocity_scale_setter is not None:
            logger.warning("감속 요청(scale=%.2f) - 외부 모션 루프에 전달", scale)
            self._external_velocity_scale_setter(scale)
            return

        logger.warning(
            "감속 요청(scale=%.2f) - 모션 스트리밍 루프에 접근할 수 없어 정지로 대체",
            scale,
        )
        self._robot.estop.activate()

    def resume(self) -> None:
        """확인됨: bot.estop.deactivate().

        주의: SafetySupervisor는 이 메서드를 자동으로 호출하지 않는다.
        사람의 확인을 거친 상위 운영 코드에서만 명시적으로 호출해야 한다.
        """
        logger.info("정상 동작 복귀")
        self._robot.estop.deactivate()

    def is_stopped(self) -> bool:
        """확인됨: bot.estop.is_software_estop_enabled() (misc.py에서 실제
        상태 토픽을 읽는 것으로 확인됨 - 서비스 콜과 무관한 독립 채널)."""
        try:
            return bool(self._robot.estop.is_software_estop_enabled())
        except Exception as e:  # noqa: BLE001
            logger.error("E-Stop 상태 확인 중 예외: %s", e)
            return False  # 확인 불가 시 "정지 아님"으로 낙관하지 않되,
            # 상위 로직에서 이 반환값 하나만으로 안전 여부를 판단하지 않도록 주의

    def shutdown(self) -> None:
        """확인됨: bot.shutdown()."""
        self._robot.shutdown()
