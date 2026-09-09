"""URDF 기반 운동량 관측기 (momentum-based collision observer).

De Luca & Mattone (2005)의 방식을 따른다:

    p(t)  = M(q) q_dot                                (일반화 운동량)
    dp/dt = tau + C(q, q_dot)^T q_dot - g(q) + tau_ext  (운동량의 시간미분)

tau_ext(외부에서 가해진 일반화 힘/토크, 충돌이면 스파이크)를 직접 구하려면
q_dot_dot(가속도, 수치 미분 - 노이즈에 취약)이 필요하지만, 위 식을 적분
형태로 바꾸면 가속도 없이 1차 필터로 tau_ext를 추정할 수 있다:

    r(t) = K_I * [ p(t) - p(0) - integral_0^t (tau + C^T q_dot - g + r) dt' ]

r(t)는 tau_ext로 수렴하는 잔차(residual)다. K_I(게인)가 크면 반응이 빠르지만
노이즈에 민감해지고, 작으면 반응이 느려진다.

중요한 한계 (반드시 읽어야 함):
  1. tau_ext는 "충돌"만이 아니라 "의도된 접촉력"도 포함한다. 즉 이 잔차
     자체는 손목 F/T 센서가 이미 측정하는 것과 근본적으로 같은 종류의
     정보(외부 접촉력)이며, "충돌이냐 정상 작업이냐"를 이 잔차 하나로
     구분하지는 못한다. 앞서 만든 AdaptiveForceBaseline과 똑같은 방식으로
     "이 작업 기준 정상 범위에서 벗어났는가"를 따로 판단해야 한다.
     이 모듈의 실질적 가치는 "손목 센서가 못 보는 팔 중간 부분의 접촉까지
     잡아낸다"는 커버리지 확장이다.
  2. 마찰(friction)이 문제다. URDF의 <joint><dynamics friction=... damping=.../>
     태그는 대부분 정확하지 않거나 비어 있다. 마찰을 보정하지 않으면 r(t)가
     0으로 수렴하지 않고 편향(bias)이 남는다. 이 구현은 마찰을 무시하므로,
     실전에서는 "절대값이 0인가"가 아니라 "정지 상태에서의 기준 잔차 대비
     얼마나 벗어났는가"로 써야 한다 (역시 AdaptiveForceBaseline과 결합 권장).
  3. tau(관절에 실제로 가해진 토크)가 필요하다. component.py(v0.4.5) 확인
     결과, get_joint_current(joint_id=None)와 get_joint_torque(joint_id=None)
     라는 전용 public 메서드가 실제로 존재한다. 단, get_joint_torque()는
     상태 dict에 "torque" 키가 있을 때만 동작하고, 우리 로봇(Vega 팔)은
     "cur"(전류, A)만 리포트해서 get_joint_torque()를 호출하면 ValueError가
     난다 (실제 get_robot_state.py 출력으로 확인됨). 즉:
       - get_joint_pos() -> 확인됨, 그대로 사용 가능
       - get_joint_vel() -> 확인됨 (v0.4.5에서 새로 확인)
       - get_joint_current() -> 확인됨, 전류(A) 반환
       - get_joint_torque() -> 존재하지만 이 로봇에서는 ValueError
     결국 "진짜 토크(Nm)"는 여전히 얻을 수 없고, 전류->토크 환산에 필요한
     모터 토크 상수(URDF에도 없음)가 확인되지 않는 한 이 관측기를 정확한
     값으로 돌릴 수 없다는 근본적인 한계는 그대로다. 대신 전류 기반
     적응형 기준선(arm_current_monitor.py)을 실전에서는 우선 사용할 것을
     권장한다.

필요 패키지: pip install pin  (Pinocchio의 PyPI 배포판 이름은 'pin')

검증 상태: 이 구현은 De Luca & Mattone 논문의 수식을 Pinocchio API로
옮긴 것이며, 이 샌드박스에 네트워크가 막혀 있어 pinocchio 설치/실행
테스트는 하지 못했습니다. computeCoriolisMatrix, computeGeneralizedGravity,
crba 함수명은 Pinocchio 공개 문서 기준으로 작성했습니다. 실제 로봇에
배포하기 전에 간단한 시뮬레이션(예: 외력 없이 자유낙하/단순 궤적 추종)에서
잔차가 0 근처로 수렴하는지 반드시 검증하세요.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class MomentumObserverConfig:
    integration_gain: float = 20.0   # K_I. 클수록 반응 빠르지만 노이즈에 민감
    control_hz: float = 100.0        # Arm 제어 루프와 같은 주기로 돌려야 함


class JointStateProvider(abc.ABC):
    """momentum observer가 필요로 하는 관절 상태를 제공하는 인터페이스.

    dexcontrol의 실제 getter가 확인되면 이 인터페이스를 구현해서 연결한다.
    """

    @abc.abstractmethod
    def get_joint_pos(self) -> np.ndarray:
        """확인됨: dexcontrol Arm.get_joint_pos() (arm.py에서 실사용 확인)."""

    @abc.abstractmethod
    def get_joint_vel(self) -> np.ndarray:
        """확인됨 (component.py v0.4.5): RobotJointComponent.get_joint_vel()."""

    @abc.abstractmethod
    def get_commanded_torque(self) -> np.ndarray:
        """근본적 한계: get_joint_torque()는 존재하지만 이 로봇은 "torque"
        필드를 리포트하지 않아(전류만 리포트) 호출하면 ValueError가 난다.
        전류->토크 환산에 필요한 모터 토크 상수가 없어 여기서 정확한 토크를
        제공할 수 없다.
        """


class DexcontrolJointStateProvider(JointStateProvider):
    """component.py(v0.4.5)에서 확인된 전용 public 메서드를 그대로 쓰는 구현.

    이전 버전은 Arm.get_state()가 반환하는 dict의 키 이름을 추측해서 썼는데
    (get_state() 자체가 v0.4.5에서 _get_state()로 private화되면서 이 방식은
    이제 아예 동작하지 않는다), 지금은 get_joint_pos()/get_joint_vel()/
    get_joint_current() 같은 전용 메서드가 확인됐으므로 그걸 직접 쓴다 -
    키 이름을 추측하거나 후보 목록을 관리할 필요가 없어졌다.
    """

    def __init__(self, arm) -> None:  # noqa: ANN001 - dexcontrol Arm 인스턴스
        self._arm = arm

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self._arm.get_joint_pos(), dtype=np.float64)

    def get_joint_vel(self) -> np.ndarray:
        return np.asarray(self._arm.get_joint_vel(), dtype=np.float64)

    def get_commanded_torque(self) -> np.ndarray:
        # get_joint_torque()가 있지만 이 로봇에서는 "torque" 필드가 없어
        # 호출하면 ValueError가 난다 (component.py 확인). 전류->토크 환산에
        # 필요한 모터 토크 상수도 확인되지 않았다. 조용히 잘못된 값을 반환하는
        # 대신 명확하게 실패시켜서, momentum observer를 이 상태로 실전에
        # 쓰지 않도록 막는다.
        raise NotImplementedError(
            "이 로봇은 get_joint_torque()를 지원하지 않고(전류만 리포트), "
            "전류->토크 환산 상수도 확인되지 않았습니다. momentum observer 대신 "
            "arm_current_monitor.py의 전류 기반 적응형 기준선을 사용하세요.",
        )


class MomentumObserver:
    """한 팔(Arm)에 대한 운동량 기반 충돌 잔차 추정기.

    사용 예 (개념적 - 실제 로봇 연결은 JointStateProvider 구현체 필요):
        from vega_dynamics import (
            load_vega_1p_urdf_path, build_arm_dynamics_model,
            get_joint_effort_limits, LEFT_ARM_JOINT_NAMES,
        )

        urdf_path = load_vega_1p_urdf_path(variant="vega_1p_f5d6")  # 실제 손 종류에 맞게
        model, joint_ids = build_arm_dynamics_model(urdf_path, LEFT_ARM_JOINT_NAMES)
        effort_limits = get_joint_effort_limits(model, joint_ids)  # [150,150,80,80,25,25,25]
        observer = MomentumObserver(model, MomentumObserverConfig())
        while True:
            q, qd, tau = provider.get_joint_pos(), provider.get_joint_vel(), provider.get_commanded_torque()
            residual = observer.step(q, qd, tau)
            over_limit = observer.check_hard_limits(residual, effort_limits, margin_ratio=0.5)
            if over_limit.any():
                # 해당 관절 근처에 설명되지 않는 외력 - 충돌 가능성
                ...
    """

    def __init__(self, model: pin.Model, cfg: MomentumObserverConfig) -> None:
        self._model = model
        self._data = model.createData()
        self._cfg = cfg
        self._dt = 1.0 / cfg.control_hz

        self._initialized = False
        self._p0 = np.zeros(model.nv)          # 초기 운동량
        self._integral = np.zeros(model.nv)    # integral(tau + C^T qd - g + r) dt 누적값
        self._residual = np.zeros(model.nv)

    def reset(self, q: np.ndarray, qd: np.ndarray) -> None:
        """관측기 상태를 리셋한다. 작업이 바뀌거나 큰 불연속(정지 후 재개) 후 호출."""
        pin.crba(self._model, self._data, q)  # M(q) 계산 (data.M에 저장됨)
        mass_matrix = self._data.M
        self._p0 = mass_matrix @ qd
        self._integral = np.zeros(self._model.nv)
        self._residual = np.zeros(self._model.nv)
        self._initialized = True

    def step(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """한 제어 스텝만큼 관측기를 갱신하고 현재 잔차 추정치를 반환한다."""
        if not self._initialized:
            self.reset(q, qd)
            return self._residual

        pin.crba(self._model, self._data, q)
        mass_matrix = self._data.M
        p = mass_matrix @ qd

        # C(q, qd)를 명시적으로 계산해서 C^T qd를 직접 구한다. (이전 버전에서
        # rnea(q,qd,0)으로 C(q,qd)*qd 를 근사하려 했으나, 이는 C^T qd와 같지
        # 않을 수 있어 - C가 대칭이 아니면 둘이 다름 - 명시적 Coriolis 행렬
        # 계산으로 교체함. computeCoriolisMatrix는 Christoffel 기호 기반으로
        # M_dot - 2C가 skew-symmetric이 되는 표준 C를 반환하므로, 관측기
        # 공식이 요구하는 C^T qd를 정확히 얻을 수 있다.)
        coriolis_matrix = pin.computeCoriolisMatrix(self._model, self._data, q, qd)
        coriolis_transpose_term = coriolis_matrix.T @ qd
        gravity = pin.computeGeneralizedGravity(self._model, self._data, q)

        integrand = tau + coriolis_transpose_term - gravity + self._residual
        self._integral += integrand * self._dt

        self._residual = self._cfg.integration_gain * (p - self._p0 - self._integral)
        return self._residual

    @staticmethod
    def check_hard_limits(
        residual: np.ndarray, effort_limits: np.ndarray, margin_ratio: float = 0.5,
    ) -> np.ndarray:
        """잔차가 정격 토크의 margin_ratio 배를 넘는 관절을 찾는다.

        vega_dynamics.get_joint_effort_limits()로 얻은 실제 URDF 정격 토크를
        그대로 쓴다. margin_ratio=0.5라면 "정격 토크의 절반에 해당하는
        설명되지 않는 외력이 잡히면 이미 위험 신호"라는 뜻 - 100%까지
        기다리면 이미 관절이 정격 한계에 도달한 뒤라 너무 늦다.

        Returns:
            residual과 같은 길이의 bool 배열. True인 인덱스가 한계를 넘은 관절.
        """
        return np.abs(residual) > (effort_limits * margin_ratio)
