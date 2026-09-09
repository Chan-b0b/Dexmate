"""안전 감시 메인 루프.

세 개의 독립적인 판단 경로를 병렬로 돌린다:

  경로 1) 안전 이상탐지 (위험한가?)
    카메라/힘센서 샘플
      -> LightweightAnomalyFilter (순간적 스파이크 - 충돌, 사람 근접 등)
      -> (후보일 때만) VLMAnomalyVerifier
      -> confidence에 따라 emergency_stop / slow_down

  경로 2) 작업 차단 판단 (계속할 수 있는가?)
    같은 카메라/힘센서 샘플
      -> StallDetector (움직임 없음 + 접촉이 있고 힘이 평평하게 유지됨 - 패턴만 확인)
      -> AdaptiveForceBaseline (이 힘 수준이 "이 작업 기준"으로 비정상인지 확인 -
         절대 임계값이 아니라 이번 작업에서 관측된 힘 분포 대비 상대적 편차를 본다)
      -> (비정상으로 판단될 때만) TaskFeasibilityVerifier
      -> is_blocked 여부에 따라 pause_task, 단 절대 물리적 안전 한계를 넘으면
         VLM 응답을 기다리지 않고 즉시 emergency_stop (기준선과 무관한 최후의 안전망)

  경로 3) 관절 전류 이상탐지 (팔 중간 부분 커버리지)
    관절별 모터 전류(cur)
      -> ArmCurrentAnomalyDetector (관절별 적응형 기준선, VectorAdaptiveBaseline)
      -> 어느 관절이든 기준선 대비 크게 벗어나면 emergency_stop
    손목 F/T 센서는 손목 이후의 접촉만 감지한다. 팔꿈치/팔뚝에 뭔가 부딪히면
    손목 센서는 못 보지만 그 관절의 전류에는 나타난다. 원래는 momentum
    observer(momentum_observer.py)로 정밀하게 하려 했으나, 전류->토크
    환산에 필요한 모터 토크 상수가 확인되지 않아 대신 이 경량 방식을 쓴다
    (자세한 이유는 arm_current_monitor.py 참고).

두 경로는 같은 원본 데이터(프레임, 힘/토크)를 재사용하지만 완전히 다른
질문을 던지고 다른 조치로 이어지므로 상태(쿨다운, 실패 카운터)를 분리해서
관리한다.

VLM 호출은 워커 스레드에서 비동기로 한다 (vlm_worker.AsyncVerifier):
    감시 루프가 VLM 응답(최대 request_timeout_s)을 기다리며 멈추면 그 동안
    관절 전류 이상탐지도, 힘 스파이크 감지도, 정지 미확인 상태 복구 재시도도
    같이 멈춘다. 게다가 dexcontrol E-Stop 서비스 콜의 클라이언트 타임아웃이
    50ms라, 루프 스레드가 붙잡혀 있으면 정지 호출 자체가 실패할 수 있다.
    그래서 VLM을 쓰는 두 경로는 "판정 수거(poll) -> 후보면 요청 제출(submit)"
    구조이고, 판정은 요청한 틱이 아니라 몇 틱 뒤에 반영된다. 요청이
    처리 중일 때 들어온 새 후보는 요청을 새로 만들지 않는다 (쿨다운과 동일
    취급). 자세한 포기(deadline) 정책은 vlm_worker.py docstring 참고.

카메라 프레임이 오래됐을 때(스트림 정지):
    dexcontrol의 get_obs()는 스트림이 얼어도 마지막 캐시 프레임을 계속
    돌려주므로, 확인하지 않으면 같은 프레임을 정상 프레임으로 취급하게 된다.
    그러면 광류가 0이 되어 후보가 아예 안 잡히면서 시스템은 "정상"으로
    보인다 - 조용한 실명.

    정책: 이 경우 로봇을 정지시키지 않는다. 시각 기반 신호만 무효로 하고
    (광류 트리거, stall 후보, VLM 판정) 힘/전류 경로로 계속 감시하면서
    경고를 반복해서 남긴다. 카메라가 잠깐 끊기는 것 자체는 로봇이
    위험하다는 뜻이 아니고, 그때마다 정지시키면 운영이 불가능해진다.
    다만 힘의 절대 안전 한계 검사는 후보 여부와 무관하게 항상 돌아간다.

fail-safe 원칙 (모든 경로 공통):
    - VLM이 응답하지 않거나 파싱에 실패하면 "정상/차단없음"으로 간주하지 않는다.
    - 연속 실패가 임계치를 넘으면 "판단 불가" 상태로 보고 정지/일시정지를 유지한다.
    - 정지/일시정지 이후 재개는 이 모듈이 자동으로 하지 않는다. 사람의 확인이나
      상위 운영 시스템의 명시적 호출을 통해서만 이루어져야 한다.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from config import (
    AnomalyFilterConfig,
    ForceBaselineConfig,
    SafetyPolicyConfig,
    StallDetectorConfig,
    TaskFeasibilityPolicyConfig,
    VLMConfig,
)
from anomaly_filter import LightweightAnomalyFilter
from arm_current_monitor import ArmCurrentAnomalyDetector
from force_baseline import AdaptiveForceBaseline
from robot_interface import EmergencyStopFailure, RobotInterface
from stall_detector import StallDetector
from vlm_client import (
    TaskFeasibilityVerdict,
    TaskFeasibilityVerifier,
    VLMAnomalyVerifier,
    VLMVerdict,
)
from vlm_worker import AsyncVerifier

logger = logging.getLogger(__name__)


class SafetySupervisor:
    def __init__(
        self,
        robot: RobotInterface,
        filter_cfg: AnomalyFilterConfig,
        vlm_cfg: VLMConfig,
        policy_cfg: SafetyPolicyConfig,
        stall_cfg: StallDetectorConfig | None = None,
        feasibility_policy_cfg: TaskFeasibilityPolicyConfig | None = None,
        force_baseline_cfg: ForceBaselineConfig | None = None,
    ) -> None:
        self._robot = robot
        self._filter_cfg = filter_cfg
        self._vlm_cfg = vlm_cfg
        self._policy_cfg = policy_cfg
        self._stall_cfg = stall_cfg or StallDetectorConfig()
        self._feasibility_policy_cfg = feasibility_policy_cfg or TaskFeasibilityPolicyConfig()
        self._force_baseline_cfg = force_baseline_cfg or ForceBaselineConfig()

        # 경로 1: 안전 이상탐지
        self._anomaly_filter = LightweightAnomalyFilter(filter_cfg)
        self._vlm = VLMAnomalyVerifier(vlm_cfg)
        # VLM 호출은 워커 스레드에서 한다 - 감시 루프가 응답을 기다리며 멈추면
        # 다른 감지 경로와 정지 경로까지 같이 멈춘다 (vlm_worker.py docstring).
        self._anomaly_worker: AsyncVerifier[VLMVerdict] = AsyncVerifier(
            name="anomaly",
            verify_fn=self._vlm.verify,
            failure_factory=lambda reason: VLMVerdict(False, 0.0, reason, ok=False),
            deadline_s=vlm_cfg.worker_deadline_s,
        )
        self._last_vlm_call_ts = 0.0
        self._consecutive_vlm_failures = 0

        # 경로 2: 작업 차단 판단
        self._stall_detector = StallDetector(self._stall_cfg, filter_cfg.filter_rate_hz)
        self._feasibility_vlm = TaskFeasibilityVerifier(vlm_cfg)
        self._feasibility_worker: AsyncVerifier[TaskFeasibilityVerdict] = AsyncVerifier(
            name="feasibility",
            verify_fn=self._feasibility_vlm.verify,
            failure_factory=lambda reason: TaskFeasibilityVerdict(
                False, 0.0, reason, "needs_human_intervention", ok=False,
            ),
            deadline_s=vlm_cfg.worker_deadline_s,
        )
        self._consecutive_feasibility_failures = 0
        self._force_baseline = AdaptiveForceBaseline(self._force_baseline_cfg)

        # 경로 3: 관절 전류 이상탐지 (손목 F/T 센서가 못 보는 팔 중간 부분 커버)
        self._current_detectors = {
            "left": ArmCurrentAnomalyDetector(self._force_baseline_cfg),
            "right": ArmCurrentAnomalyDetector(self._force_baseline_cfg),
        }

        # 카메라 프레임 신선도 경고 상태 (_check_frame_freshness)
        self._stale_since: float | None = None
        self._last_stale_warn_ts = 0.0
        # 절대 힘 한계 정지를 매 틱 반복하지 않기 위한 래치
        self._hard_force_tripped = False

        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 실행 제어
    # ------------------------------------------------------------------
    def warm_up(self) -> bool:
        """콜드 스타트 비용을 감시 시작 전에 미리 치른다. start() 전에 호출.

        왜 필요한가 (실측, Qwen3.5-35B-A3B / Jetson Thor):
            프로세스 첫 판정 6.3s, 새 이미지 해상도의 첫 판정 3.9s, 그 이후는
            1.5s로 안정된다. request_timeout_s(4.0s)보다 길기 때문에 워밍업
            없이 시작하면 "기동 후 첫 이상 상황"의 판정이 타임아웃으로
            날아간다 - 정작 가장 필요한 순간이다.

            워밍업 프레임은 반드시 실제 카메라에서 받아온다. 콜드 비용의 일부는
            이미지 해상도별로 따로 발생하므로(측정: 새 해상도 첫 호출 3.9s),
            합성 프레임을 다른 크기로 보내면 워밍업이 헛돌 수 있다.

        두 경로를 모두 부르는 이유는 시작 자기진단 역할도 겸하기 때문이다.
        모델 id 오타나 서버 미기동 같은 문제를 "첫 이상 상황"이 아니라
        기동 시점에 알 수 있다.

        Returns:
            두 경로 모두 판정을 정상적으로 받았으면 True.
        """
        try:
            frame = self._robot.get_camera_frame()
        except Exception:  # noqa: BLE001 - 워밍업 실패가 기동을 막아선 안 된다
            logger.exception("워밍업용 카메라 프레임을 못 받았습니다 - 워밍업 생략")
            return False

        logger.info(
            "VLM 워밍업 시작 (프레임 %dx%d, 타임아웃 %.0fs) - 콜드 스타트는 수 초 걸립니다",
            frame.shape[1], frame.shape[0], self._vlm_cfg.warmup_timeout_s,
        )

        task_context = self._robot.get_current_task_description()
        all_ok = True
        for name, verifier in (("이상", self._vlm), ("작업차단", self._feasibility_vlm)):
            started = time.monotonic()
            verdict = verifier.verify(
                frame, task_context, timeout_s=self._vlm_cfg.warmup_timeout_s,
            )
            elapsed = time.monotonic() - started
            if verdict.ok:
                logger.info("VLM 워밍업(%s) 완료: %.2fs", name, elapsed)
            else:
                all_ok = False
                logger.critical(
                    "VLM 워밍업(%s) 실패 (%.2fs) - 모델 id(%s)와 서버 상태를 확인하세요. "
                    "이대로 시작하면 첫 이상 상황에서 판정을 못 받아 fail-safe 정지가 걸립니다.",
                    name, elapsed, self._vlm_cfg.model,
                )
        return all_ok

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("SafetySupervisor 시작")

    def stop_supervisor(self) -> None:
        """감시 루프 자체를 종료 (로봇 정지와는 다름).

        VLM 워커 스레드는 daemon이라 여기서 기다리지 않는다. 남아 있는
        요청은 HTTP 타임아웃 후 스스로 끝나고, 그 결과를 받아 조치할 루프가
        이미 없으므로 그냥 버려진다.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("SafetySupervisor 종료")

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        period_s = 1.0 / self._filter_cfg.filter_rate_hz
        while self._running:
            loop_start = time.monotonic()
            try:
                self._tick()
            except EmergencyStopFailure as e:
                # 정지 호출 자체가 재시도 후에도 실패한 최악의 상황.
                # 다른 판단(카메라/힘 확인 등)을 계속하는 건 의미가 없으므로
                # 정지가 확인될 때까지 이 상태에서 벗어나지 않는다.
                self._handle_unconfirmed_stop(e)
                continue
            except Exception:  # noqa: BLE001 - 감시 루프는 절대 죽으면 안 됨
                logger.exception("감시 루프 tick 중 예외 발생")
                try:
                    self._robot.emergency_stop(reason="supervisor internal error")
                except EmergencyStopFailure as e:
                    self._handle_unconfirmed_stop(e)
                    continue

            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, period_s - elapsed)
            time.sleep(sleep_s)

    def _handle_unconfirmed_stop(self, error: EmergencyStopFailure) -> None:
        """소프트웨어로 정지를 확인할 수 없는 상태를 처리한다.

        겁먹고 다음 틱으로 넘어가는 대신, 정지가 확인되거나 감시 루프 자체가
        외부에서 멈춰질 때까지 계속 재시도한다. 이 동안 다른 감지 경로는
        멈춘다 - "정지조차 확신 못 하면서 다른 판단을 계속하는 것"은
        의미가 없고, 오히려 상황을 헷갈리게 만든다.

        이 상태에 들어갔다면 이미 소프트웨어 e-stop 경로가 신뢰할 수 없다는
        뜻이므로, 실제로는 사람이 물리적 e-stop으로 개입해야 한다. 로그를
        계속 CRITICAL로 남기는 것은 그 개입을 유도하기 위함이다.
        """
        while self._running:
            logger.critical(
                "정지 미확인 상태 - 소프트웨어로 정지를 확신할 수 없습니다. "
                "즉시 물리적 E-Stop으로 개입하세요. 원인: %s", error,
            )
            time.sleep(1.0)
            try:
                self._robot.emergency_stop(reason="정지 미확인 상태 복구 재시도")
            except EmergencyStopFailure as e:
                error = e
                continue
            logger.critical("정지 확인됨 - 감시 루프 복구. 재개 여부는 사람이 판단해야 합니다.")
            return

    def _tick(self) -> None:
        frame = self._robot.get_camera_frame()
        frame_fresh = self._check_frame_freshness(self._robot.get_camera_frame_age_s())
        force_torque = self._robot.get_force_torque()
        force_mag = float(np.linalg.norm(force_torque))
        task_context = self._robot.get_current_task_description()

        # 두 경로 모두 같은 원본 데이터를 사용한다.
        anomaly_candidate, anomaly_debug = self._anomaly_filter.check(frame, force_torque)

        if not frame_fresh:
            # 프레임이 오래됐으면 광류에서 나온 신호는 의미가 없다 (같은 프레임이
            # 반복되면 광류는 0이 된다). 힘 스파이크만으로 후보를 판단한다.
            anomaly_candidate = bool(anomaly_debug["force_spike"])

        self._run_anomaly_path(
            frame, force_torque, anomaly_candidate, anomaly_debug, frame_fresh,
        )

        # anomaly_filter가 이미 계산한 광류값을 재사용 (중복 계산 방지)
        stall_candidate, stall_debug = self._stall_detector.update(
            anomaly_debug["flow_magnitude"], force_torque,
        )

        if not frame_fresh:
            # stall 판단은 "움직임 없음"을 근거로 하는데, 오래된 프레임은
            # 실제로 멈춘 것과 구별되지 않는다. 즉 스트림이 얼면 stall 후보가
            # 계속 참이 되어버린다. 그래서 후보 자체를 무효로 본다.
            # (힘이 절대 안전 한계를 넘는지는 아래에서 후보와 무관하게 확인한다)
            stall_candidate = False

        # 기준선 학습: 이번 틱에서 "이상하다"고 플래그되지 않은 힘 샘플만
        # 정상 데이터로 취급해서 기준선에 반영한다. 이상 후보 자체를 학습에
        # 넣으면 기준선이 이상 상황을 "정상"으로 오염시켜버린다.
        if not anomaly_candidate and not stall_candidate:
            self._force_baseline.update(task_context, force_mag)

        self._run_task_feasibility_path(frame, force_mag, task_context, stall_candidate, stall_debug)

        # 경로 3: 관절 전류 이상탐지. anomaly/stall 경로와 별개로 항상 확인한다
        # (움직임/힘 패턴과 무관하게, 관절 하나가 유독 이상한 전류를 쓰는
        # 경우도 있을 수 있으므로).
        self._run_arm_current_path(task_context, anomaly_candidate, stall_candidate)

    def _check_frame_freshness(self, age_s: float | None) -> bool:
        """카메라 프레임이 신선한지 확인하고, 오래됐으면 경고를 남긴다.

        정책: 오래된 프레임을 만나도 로봇을 정지시키지 않는다. 시각 기반
        판단(광류 트리거, VLM 판정)만 끄고 힘/전류 경로로 계속 감시한다.
        카메라 스트림이 잠깐 끊기는 것 자체는 로봇이 위험하다는 뜻이 아니고,
        그때마다 정지시키면 운영이 불가능해지기 때문이다. 대신 시각 경로가
        죽은 상태로 조용히 계속 도는 것을 막기 위해 경고를 반복해서 남긴다.

        age_s가 None이면(어댑터가 신선도를 알려주지 않으면) 가드를 적용하지
        않는다 - 알 수 없다는 이유로 시각 경로를 통째로 끄면 더 위험하다.

        Returns:
            신선하면 True. False면 호출부가 시각 기반 신호를 무효로 처리한다.
        """
        cfg = self._filter_cfg
        now = time.monotonic()

        if age_s is None or age_s <= cfg.max_frame_age_s:
            if self._stale_since is not None:
                logger.warning(
                    "카메라 프레임 신선도 복구 (%.1fs 동안 오래된 프레임) - "
                    "광류/VLM 경로 재개",
                    now - self._stale_since,
                )
                self._stale_since = None
                self._last_stale_warn_ts = 0.0
            return True

        if self._stale_since is None:
            self._stale_since = now
        if now - self._last_stale_warn_ts >= cfg.stale_warn_interval_s:
            self._last_stale_warn_ts = now
            logger.warning(
                "카메라 프레임이 %.2fs 전 것입니다 (허용 %.2fs, %.1fs째 지속) - "
                "카메라 스트림을 확인하세요. 광류 트리거와 VLM 판정을 끄고 "
                "힘/전류 경로로만 감시합니다.",
                age_s, cfg.max_frame_age_s, now - self._stale_since,
            )
        return False

    def _run_arm_current_path(
        self, task_context: str, anomaly_candidate: bool, stall_candidate: bool,
    ) -> None:
        for side, detector in self._current_detectors.items():
            currents = self._robot.get_arm_joint_currents(side)
            if currents is None:
                continue  # 이 로봇 어댑터가 이 채널을 지원하지 않음

            is_abnormal, ratios, debug_info = detector.check(currents)

            if not anomaly_candidate and not stall_candidate and not is_abnormal:
                # 다른 경로에서도 이상이 없을 때만 정상 데이터로 기준선 학습
                detector.update_baseline(task_context, currents)

            if is_abnormal and ratios is not None:
                worst_idx = debug_info["worst_joint_idx"]
                logger.warning(
                    "관절 전류 이상 감지: side=%s joint_idx=%d ratio=%.1f (임계값=%.1f)",
                    side, worst_idx, debug_info["max_ratio"],
                    self._force_baseline_cfg.deviation_ratio_threshold,
                )
                self._robot.emergency_stop(
                    reason=(
                        f"{side} 팔 관절 {worst_idx}번 전류 이상 "
                        f"(기준선 대비 {debug_info['max_ratio']:.1f}배)"
                    ),
                )

    # ------------------------------------------------------------------
    # 경로 1: 안전 이상탐지 (위험한가?)
    # ------------------------------------------------------------------
    def _run_anomaly_path(
        self,
        frame: np.ndarray,
        force_torque: np.ndarray,
        is_candidate: bool,
        debug_info: dict,
        frame_fresh: bool = True,
    ) -> None:
        # 판정 수거는 후보 여부와 무관하게 매 틱 먼저 한다. 요청을 보낸 뒤
        # 1차 필터가 후보를 더 이상 안 잡아도, 요청 시점엔 이상 후보였으므로
        # 도착한 판정은 반영해야 한다.
        self._collect_anomaly_verdict()

        if not is_candidate:
            return

        logger.debug("1차 필터(이상) 후보 감지: %s", debug_info)

        now = time.monotonic()
        if (
            now - self._last_vlm_call_ts < self._policy_cfg.vlm_cooldown_s
            or self._anomaly_worker.busy()
        ):
            # 쿨다운 중이거나 이미 판정을 기다리는 중에는 VLM을 다시 호출하지
            # 않되, 이미 이상 후보가 계속 관측되고 있다는 뜻이므로 최소한
            # 감속 상태는 유지한다 (동기 호출 때와 같은 조치).
            self._robot.slow_down(scale=0.3)
            return

        if not frame_fresh:
            # 오래된 프레임을 VLM에 보내면 지나간 장면에 대한 판정을 받게
            # 된다. 힘 스파이크로 여기까지 온 상황이므로 그 신호는 유효하지만,
            # 시각 확인은 포기하고 힘/전류 경로에 맡긴다.
            return

        task_context = self._robot.get_current_task_description()
        if self._anomaly_worker.submit(frame, task_context):
            # 요청이 실제로 시작됐을 때만 쿨다운을 갱신한다. 판정은 이후
            # 틱에서 _collect_anomaly_verdict()가 수거한다.
            self._last_vlm_call_ts = now

    def _collect_anomaly_verdict(self) -> None:
        verdict = self._anomaly_worker.poll()
        if verdict is None:
            return

        if not verdict.ok:
            self._handle_vlm_failure()
            return

        self._consecutive_vlm_failures = 0
        self._apply_anomaly_policy(verdict)

    def _apply_anomaly_policy(self, verdict) -> None:  # noqa: ANN001
        cfg = self._policy_cfg
        logger.info(
            "VLM(이상) 판단: is_anomaly=%s confidence=%.2f reason=%s",
            verdict.is_anomaly, verdict.confidence, verdict.reason,
        )

        if not verdict.is_anomaly:
            return  # 정상 판단 -> 아무것도 하지 않음

        if verdict.confidence >= cfg.stop_confidence_threshold:
            self._robot.emergency_stop(reason=f"VLM 이상 감지: {verdict.reason}")
        elif verdict.confidence >= cfg.slow_confidence_threshold:
            self._robot.slow_down(scale=0.2)
        # confidence가 낮으면 로깅만 하고 별도 조치는 하지 않음 (오탐 가능성 고려)

    def _handle_vlm_failure(self) -> None:
        self._consecutive_vlm_failures += 1
        logger.warning("VLM(이상) 호출/파싱 실패 (연속 %d회)", self._consecutive_vlm_failures)
        if self._consecutive_vlm_failures >= self._policy_cfg.max_consecutive_vlm_failures:
            self._robot.emergency_stop(
                reason="VLM(이상) 연속 응답 실패 - 안전 판단 불가로 fail-safe 정지",
            )

    # ------------------------------------------------------------------
    # 경로 2: 작업 차단 판단 (계속할 수 있는가?)
    # ------------------------------------------------------------------
    def _run_task_feasibility_path(
        self,
        frame: np.ndarray,
        force_mag: float,
        task_context: str,
        is_candidate: bool,
        debug_info: dict,
    ) -> None:
        # 경로 1과 같은 이유로 판정 수거를 먼저, 후보 여부와 무관하게 한다.
        self._collect_feasibility_verdict()

        cfg = self._feasibility_policy_cfg

        # 1단계: 절대 물리적 안전 한계. **후보 여부와 무관하게 항상** 확인한다.
        # 원래는 stall 후보일 때만 봤는데, 그러면 카메라 스트림이 얼어서 stall
        # 후보가 안 잡히는 동안(광류가 0이면 움직임 없음/있음 구분이 무의미해서
        # _tick이 후보를 무효화한다) 이 최후의 안전망까지 같이 죽는다. 이 검사는
        # 힘 채널만 쓰므로 카메라 상태와 무관하게 판단할 수 있다.
        # 기준선이 어떤 이유로든 드리프트돼 있어도 이 선은 넘지 않는다.
        if force_mag >= cfg.absolute_hard_force_limit_n:
            # 15Hz로 estop 서비스를 두드리지 않도록, 한계를 넘어 있는 동안은
            # 한 번만 요청한다 (힘이 한계 아래로 내려오면 다시 무장된다).
            if not self._hard_force_tripped:
                self._hard_force_tripped = True
                self._robot.emergency_stop(
                    reason=f"힘이 절대 안전 한계 초과 ({force_mag:.1f}N)",
                )
            return
        self._hard_force_tripped = False

        if not is_candidate:
            return

        logger.debug("1차 필터(작업차단) 후보 감지: %s", debug_info)

        # 2단계: 이 작업에서 "정상적인 힘 수준"에서 얼마나 벗어났는지 확인.
        # 절대값이 아니라 이번 작업에서 관측된 분포 기준 상대적 편차를 본다.
        ratio = self._force_baseline.deviation_ratio(force_mag)
        baseline_cfg = self._force_baseline_cfg

        if ratio is not None:
            is_abnormal_for_task = ratio >= baseline_cfg.deviation_ratio_threshold
            logger.debug(
                "힘 기준선 대비 편차: force=%.1fN baseline=%.1fN ratio=%.1f (임계값=%.1f)",
                force_mag, self._force_baseline.get_baseline() or -1.0,
                ratio, baseline_cfg.deviation_ratio_threshold,
            )
        else:
            # 기준선이 아직 준비되지 않음 (작업 시작 직후) - 보수적 절대값으로 fallback
            is_abnormal_for_task = force_mag >= baseline_cfg.fallback_absolute_threshold_n
            logger.debug(
                "힘 기준선 미준비 - fallback 절대 임계값(%.1fN)으로 판단: force=%.1fN",
                baseline_cfg.fallback_absolute_threshold_n, force_mag,
            )

        if not is_abnormal_for_task:
            # 이 작업에서는 흔히 있는 힘 수준 - stall 패턴(움직임 없음)은 맞지만
            # 예를 들어 "누른 채로 유지하는" 정상 작업 단계일 수 있으므로 VLM까지
            # 갈 필요 없음.
            return

        if not self._stall_detector.cooldown_ok() or self._feasibility_worker.busy():
            return

        if self._feasibility_worker.submit(frame, task_context):
            # 요청이 실제로 시작됐을 때만 쿨다운을 갱신한다. 판정은 이후
            # 틱에서 _collect_feasibility_verdict()가 수거한다.
            self._stall_detector.mark_checked()

    def _collect_feasibility_verdict(self) -> None:
        verdict = self._feasibility_worker.poll()
        if verdict is None:
            return

        if not verdict.ok:
            self._handle_feasibility_failure()
            return

        self._consecutive_feasibility_failures = 0
        self._apply_feasibility_policy(verdict)

    def _apply_feasibility_policy(self, verdict) -> None:  # noqa: ANN001
        cfg = self._feasibility_policy_cfg
        logger.info(
            "VLM(작업차단) 판단: is_blocked=%s confidence=%.2f reason=%s action=%s",
            verdict.is_blocked, verdict.confidence, verdict.blocking_reason,
            verdict.suggested_action,
        )

        if not verdict.is_blocked:
            return

        if verdict.confidence < cfg.blocked_confidence_threshold:
            return  # 확신이 낮으면 조치하지 않음 (오탐 가능성 고려)

        self._robot.pause_task(
            reason=(
                f"작업 차단 감지 ({verdict.suggested_action}): {verdict.blocking_reason}"
            ),
        )

    def _handle_feasibility_failure(self) -> None:
        self._consecutive_feasibility_failures += 1
        logger.warning(
            "VLM(작업차단) 호출/파싱 실패 (연속 %d회)", self._consecutive_feasibility_failures,
        )
        if self._consecutive_feasibility_failures >= self._feasibility_policy_cfg.max_consecutive_vlm_failures:
            self._robot.pause_task(
                reason="VLM(작업차단) 연속 응답 실패 - 판단 불가로 fail-safe 일시정지",
            )
