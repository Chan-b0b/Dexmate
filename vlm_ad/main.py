"""엔트리 포인트.

사전 준비:
  1. Jetson Thor에서 vLLM 서버 실행 (전체 명령어는 config.py 상단 주석 참고).
     VLM은 컨테이너, 이 감시 로직은 호스트에서 돌린다 - 컨테이너를
     --network host로 띄우므로 localhost:8000으로 그냥 붙는다.
       vllm serve /models/Qwen3.5-35B-A3B --served-model-name qwen3.5 \
           --max-model-len 8192 --gpu-memory-utilization 0.7 \
           --limit-mm-per-prompt '{"image":1,"video":0}'
     서빙된 모델 id가 config.py의 VLMConfig.model과 정확히 같아야 한다
     (`curl localhost:8000/v1/models`로 확인). 다르면 매 호출이 404로
     떨어져 fail-safe 정지가 계속 걸린다.
  2. python main.py

dexcontrol 초기화는 examples/basic_examples/sensors/get_head_zed_x_mini_data.py에서
확인된 방식을 따른다: head_camera 센서를 명시적으로 enable해야 한다.
"""

from __future__ import annotations

import importlib.metadata
import logging
import signal
import sys

from config import AnomalyFilterConfig, DropDetectorConfig, EXPECTED_DEXCONTROL_MINOR_LINE, ForceBaselineConfig, SafetyPolicyConfig, StallDetectorConfig, TaskFeasibilityPolicyConfig, VLMConfig
from drop_detector import PayloadDropDetector
from robot_interface import DexcontrolAdapter
from safety_supervisor import SafetySupervisor

# case_battery_demo 패키지(Robotiq Modbus 구현)를 import하기 위한 부모 경로.
# 경로 4의 그리퍼 감시에만 쓰이고, 없으면 그 EE만 건너뛴다.
CASE_BATTERY_DEMO_PARENT = "/home/dexmate/LGES/Dexmate/LGES"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _check_dexcontrol_version() -> None:
    """dexcontrol과 로봇 펌웨어의 마이너 버전 락스텝을 시작 시 확인한다.

    이 프로젝트는 한때 "dexcontrol 0.5.0 + 펌웨어 0.4.5" 조합에서
    E-Stop 서비스 콜이 응답 없이 영원히 멈추는 문제를 겪어 (타임아웃을
    5000초로 늘려도 동일) dexcontrol을 0.4.x로 다운그레이드했었다. 원인은
    dexcontrol과 펌웨어가 마이너 버전 단위로 락스텝되어 있다는 것이었다
    (PyPI 공식 호환성 표 기준). 이후 로봇 펌웨어가 0.5.x로 업그레이드되면서
    dexcontrol 0.5.0의 E-Stop이 정상 동작함을 확인했고, 지금은 dexcontrol도
    다시 0.5.x로 맞췄다 (config.EXPECTED_DEXCONTROL_MINOR_LINE).

    이 체크는 마이너 버전 락스텝이 조용히 다시 깨지는 걸 막기 위한 것이다 -
    버전이 안 맞으면 프로그램을 막지는 않지만(펌웨어 버전을 코드에서 직접
    조회할 방법이 없어서 완벽한 체크는 불가), 최소한 CRITICAL 로그로 강하게
    경고한다.
    """
    try:
        installed = importlib.metadata.version("dexcontrol")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("dexcontrol 패키지를 찾을 수 없습니다 - 설치 여부를 확인하세요.")
        return

    minor_line = ".".join(installed.split(".")[:2])
    if minor_line != EXPECTED_DEXCONTROL_MINOR_LINE:
        logger.critical(
            "dexcontrol 버전(%s)이 이 프로젝트가 검증된 라인(%s.x)과 다릅니다! "
            "dexcontrol과 로봇 펌웨어는 마이너 버전 단위로 락스텝되어 있어서, "
            "버전이 안 맞으면 E-Stop 같은 서비스 콜이 응답 없이 영원히 멈추는 "
            "문제가 생길 수 있습니다 (실제로 겪었던 문제). 로봇 펌웨어 버전과 "
            "다시 맞춰주세요 (config.py의 EXPECTED_DEXCONTROL_MINOR_LINE, "
            "requirements.txt의 핀도 같이 갱신).",
            installed, EXPECTED_DEXCONTROL_MINOR_LINE,
        )
    else:
        logger.info("dexcontrol 버전 확인: %s (예상 라인 %s.x와 일치)", installed, minor_line)


def _build_drop_detectors(robot, cfg: DropDetectorConfig):  # noqa: ANN001, ANN202
    """경로 4(페이로드 낙하 감지)용 EE 센서를 연결한다.

    하드웨어/의존성이 없으면 그 EE만 건너뛴다 - 낙하 감지는 부가 경로이고,
    이것 때문에 감시 시스템 전체가 안 뜨면 안 된다.

    반환: (detectors, suction_sensors) - suction_sensors는 종료 시
    소켓을 닫기 위해 돌려준다 (없으면 None).
    """
    detectors: list[PayloadDropDetector] = []
    suction_sensors = None

    # --- 흡착 컵 (좌측 팔) ---
    try:
        from end_effector_adapters import SuctionCupSensors

        suction_sensors = SuctionCupSensors(cfg)
        suction_sensors.start()
        detectors.append(PayloadDropDetector(suction_sensors, cfg))
        logger.info("낙하 감지: 흡착 컵 연결 (%s)", cfg.suction_host)
    except Exception as e:  # noqa: BLE001
        logger.warning("낙하 감지: 흡착 컵을 연결하지 못했습니다 (%s) - 이 EE는 건너뜁니다", e)

    # --- Robotiq 그리퍼 (우측 팔) ---
    # 기본 비활성이다. 같은 RS485 pass-through 채널을 데모 프로세스와
    # 공유하므로 켜면 데모의 상태 읽기를 가로챌 수 있다
    # (config.enable_gripper_drop_detection 주석 참고).
    if not cfg.enable_gripper_drop_detection:
        logger.info(
            "낙하 감지: Robotiq 그리퍼 경로는 비활성입니다 "
            "(RS485 채널 경합 - config.enable_gripper_drop_detection 참고)",
        )
        if not detectors:
            logger.warning("낙하 감지: 연결된 EE가 없어 경로 4는 비활성됩니다")
        return detectors, suction_sensors

    # robotiq.py의 Modbus 프레이밍/CRC/파싱을 재구현하지 않기 위해 데모
    # 패키지의 클래스를 그대로 쓴다. 그래서 그 패키지가 import 가능해야 한다.
    try:
        import sys

        if CASE_BATTERY_DEMO_PARENT not in sys.path:
            sys.path.insert(0, CASE_BATTERY_DEMO_PARENT)
        from case_battery_demo.robotiq import RobotiqGripper

        from end_effector_adapters import RobotiqGripperSensors

        gripper = RobotiqGripper(robot, side="right")
        if not gripper.available():
            logger.warning(
                "낙하 감지: Robotiq EE pass-through를 쓸 수 없습니다 "
                "(그리퍼가 native 인식되면 pass-through가 비활성됨) - 이 EE는 건너뜁니다",
            )
        else:
            detectors.append(PayloadDropDetector(RobotiqGripperSensors(gripper, cfg), cfg))
            logger.info("낙하 감지: Robotiq 그리퍼 연결 (우측 팔)")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "낙하 감지: Robotiq 그리퍼를 연결하지 못했습니다 (%s) - 이 EE는 건너뜁니다", e,
        )

    if not detectors:
        logger.warning("낙하 감지: 연결된 EE가 없어 경로 4는 비활성됩니다")
    return detectors, suction_sensors


def main() -> None:
    _check_dexcontrol_version()

    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"  # 인터넷 불안정 환경이므로 rtc 대신 zenoh 권장

    # estop_robot.py 예제와 달리, 감시 시스템은 프로세스 생명주기 동안 계속
    # 로봇 연결을 들고 있어야 하므로 with 블록 안에서 supervisor를 실행한다.
    with Robot(configs=configs) as robot:
        logger.info("헤드 카메라 스트림 활성화 대기 중...")
        if robot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.info("카메라 스트림 활성화 완료")
        else:
            logger.warning("카메라 스트림이 활성화되지 않았습니다 - 연결을 확인하세요")

        adapter = DexcontrolAdapter(
            robot,
            monitor_arms=("left", "right"),
            # 만약 별도로 100Hz pos/vel 스트림을 보내는 모션 루프가 있고,
            # 그 루프가 참조할 수 있는 속도 스케일 변수/콜백이 있다면 여기에
            # 연결하면 진짜 감속이 동작한다. 없으면 None으로 두면
            # slow_down()도 안전하게 estop으로 처리된다.
            external_velocity_scale_setter=None,
        )

        drop_cfg = DropDetectorConfig()
        drop_detectors, suction_sensors = _build_drop_detectors(robot, drop_cfg)

        supervisor = SafetySupervisor(
            robot=adapter,
            filter_cfg=AnomalyFilterConfig(),
            vlm_cfg=VLMConfig(),
            policy_cfg=SafetyPolicyConfig(),
            stall_cfg=StallDetectorConfig(),
            feasibility_policy_cfg=TaskFeasibilityPolicyConfig(),
            force_baseline_cfg=ForceBaselineConfig(),
            drop_detectors=drop_detectors,
            # 낙하 시 실제로 할 일(재파지 요청, 라인 알림, 대시보드 표시 등)을
            # 여기 연결한다. None이면 로그만 남는다.
            on_drop=None,
        )

        def _handle_sigint(_sig, _frame):  # noqa: ANN001
            logger.info("종료 신호 수신, 감시 루프 정리 중...")
            supervisor.stop_supervisor()
            if suction_sensors is not None:
                suction_sensors.stop()
            adapter.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_sigint)

        # 감시 루프를 켜기 전에 VLM 콜드 스타트를 미리 치른다. 실패해도
        # 기동을 막지는 않는다 - VLM이 죽어 있어도 관절 전류/힘 기반 경로는
        # 여전히 동작하므로, 감시를 아예 안 켜는 것보다 켜는 게 안전하다.
        # (실패 시 warm_up()이 CRITICAL 로그를 남긴다)
        supervisor.warm_up()

        supervisor.start()
        logger.info("안전 감시 시스템 가동 중. Ctrl+C로 종료.")
        signal.pause()


if __name__ == "__main__":
    main()
