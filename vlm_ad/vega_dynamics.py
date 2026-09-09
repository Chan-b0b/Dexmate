"""dexmate_urdf 패키지에서 Vega URDF를 불러와 Pinocchio 동역학 모델을 구성.

확인된 사실 (PyPI 프로젝트 페이지 기준):
  - Dexmate가 이 URDF의 지원 플랫폼으로 Pinocchio를 공식 명시하고 있음
    ("🔄 Pinocchio - For kinematics and dynamics computation"). 즉 시각화용
    메쉬만 있는 게 아니라 실제 동역학 계산(질량/관성 파라미터)을 염두에 두고
    만들어졌다고 볼 수 있다.
  - vega_1p 폴더에는 손 종류별로 3가지 변형이 있다:
      vega_1p.urdf          # 손 없음
      vega_1p_f5d6.urdf     # F5D6 손
      vega_1p_gripper.urdf  # 그리퍼
    실제 로봇에 붙은 엔드이펙터와 일치하는 파일을 써야 한다 - 손의 무게/
    관성이 관측기 잔차에 직접 영향을 준다.

확인된 것 (사용자가 제공한 vega_1p_f5d6.urdf 원문 기준):
  - 조인트 이름: L_arm_j1~j7, R_arm_j1~j7 (7-DOF 리볼루트 + j8/ee는 고정 조인트)
  - 관절별 정격 토크(effort limit): j1/j2=150 Nm, j3/j4=80 Nm, j5/j6/j7=25 Nm
  - 관절별 속도 한계(velocity limit): j1/j2=2.4 rad/s, j3~j7=2.7 rad/s
    (dexcontrol의 arm.py에서 본 "2.8 rad/s로 clip"이라는 안전장치와 대략
    일치하는 범위 - 같은 로봇을 가리키고 있다는 정합성 확인)

확인되지 않은 것:
  - dexcontrol의 Arm이 report하는 조인트 이름(robot_info.get_component_joints())
    이 URDF의 "L_arm_j1" 같은 표기와 정확히 동일한 문자열인지는 여전히
    미확인이다. 다만 같은 회사가 같은 로봇을 위해 만든 두 패키지이므로
    일치할 가능성이 높다 - 실제 로봇에서 한 번 확인해보는 것을 권장.

중요한 단순화 (반드시 인지해야 함):
  Vega는 토르소 + 이동 베이스가 있는 휴머노이드다. 팔의 정확한 동역학
  (특히 중력 항)은 원칙적으로 로봇 전체의 관절 각도(q)에 의존한다 - 토르소가
  기울어지면 팔에 걸리는 중력 성분도 달라진다. 이 모듈은 "팔을 움직이는
  동안 토르소/베이스는 고정되어 있다"를 가정하고 팔만 떼어낸 축소 모델을
  쓴다. 이 가정은 "베이스를 세우고 팔로만 조작하는" 일반적인 픽 작업에는
  합리적이지만, 팔과 토르소/베이스가 동시에 움직이는 전신 동작 중에는
  깨진다. 그 경우 전신 모델(q 전체)을 써야 하며, 이는 이 모듈의 범위를
  넘어선다.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin


_VALID_VARIANTS = ("vega_1p", "vega_1p_f5d6", "vega_1p_gripper")

# 확인됨 (사용자가 제공한 URDF 원문 기준, robots/humanoid/vega_1p/vega_1p_f5d6.urdf).
# 7-DOF 리볼루트 조인트(j1~j7) + 손목 이후 고정 조인트(j8, ee) 구조.
# parent link가 둘 다 "arm_center"인 걸 보면 토르소의 한 지점에서 양팔이 갈라진다.
LEFT_ARM_JOINT_NAMES = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"R_arm_j{i}" for i in range(1, 8)]


def load_vega_1p_urdf_path(variant: str = "vega_1p_f5d6") -> str:
    """dexmate_urdf 패키지에서 실제 로봇 구성과 일치하는 URDF 경로를 가져온다.

    Args:
        variant: "vega_1p"(손 없음) / "vega_1p_f5d6"(F5D6 손) /
            "vega_1p_gripper"(그리퍼) 중 실제 로봇의 엔드이펙터와 일치하는 것.
    """
    try:
        from dexmate_urdf import robots
    except ImportError as e:
        raise ImportError(
            "dexmate_urdf가 설치되지 않았습니다: pip install dexmate-urdf",
        ) from e

    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant는 {_VALID_VARIANTS} 중 하나여야 합니다: {variant}")

    vega_1p = robots.humanoid.vega_1p
    return str(getattr(vega_1p, variant).urdf)


def build_arm_dynamics_model(
    urdf_path: str,
    arm_joint_names: list[str],
) -> tuple[pin.Model, list[int]]:
    """URDF를 불러와 지정한 조인트들만 남긴 축소 동역학 모델을 만든다.

    "축소"는 다른 조인트를 물리적으로 없애는 게 아니라, momentum_observer가
    다룰 q/q_dot/tau 벡터의 순서를 이 조인트 이름 리스트 순서에 맞춰 매핑해
    주는 것이다. 실제 M(q), C(q,q_dot), g(q) 계산은 여전히 전체 URDF
    구조(팔이 몸통에 어떻게 붙어 있는지 등)를 반영한다 - 다만 토르소/베이스
    관절은 URDF 기본 자세(q=0)로 고정된 값으로 취급한다 (위 모듈 docstring의
    단순화 가정).

    Args:
        urdf_path: load_vega_1p_urdf_path()로 얻은 경로.
        arm_joint_names: dexcontrol Arm이 보고하는 조인트 이름 목록
            (예: bot.left_arm의 7개 조인트 이름). 이 순서가 momentum
            observer에 넣는 q/q_dot/tau 벡터의 순서와 일치해야 한다.

    Raises:
        ValueError: arm_joint_names 중 URDF에 없는 이름이 있으면, 어떤
            이름이 문제인지와 URDF에 실제로 있는 조인트 이름 목록을 함께
            보여준다 (dexcontrol과 URDF의 이름 표기가 다를 경우 직접 매핑
            테이블을 만들 때 참고할 수 있게).
    """
    model = pin.buildModelFromUrdf(urdf_path)

    missing = [name for name in arm_joint_names if not model.existJointName(name)]
    if missing:
        available = [model.names[i] for i in range(model.njoints)]
        raise ValueError(
            f"URDF에서 찾을 수 없는 조인트: {missing}\n"
            f"URDF에 실제로 있는 조인트 이름: {available}\n"
            "dexcontrol의 조인트 이름과 URDF의 조인트 이름 표기가 다를 수 있습니다. "
            "이 경우 이름 문자열을 직접 대응시키는 매핑 딕셔너리가 필요합니다.",
        )

    joint_ids = [model.getJointId(name) for name in arm_joint_names]
    return model, joint_ids


def get_joint_effort_limits(model: pin.Model, joint_ids: list[int]) -> np.ndarray:
    """URDF의 <limit effort="..."/> 값을 조인트 순서에 맞춰 뽑아낸다.

    확인됨: 사용자가 제공한 vega_1p_f5d6.urdf 원문에 실제 값이 있다
    (L_arm_j1/j2=150 Nm, j3/j4=80 Nm, j5/j6/j7=25 Nm - 오른팔도 동일).
    이 함수는 하드코딩된 숫자를 쓰지 않고, Pinocchio가 URDF를 파싱하면서
    자동으로 채운 model.effortLimit에서 직접 읽는다 - URDF가 바뀌면
    (예: 다른 손 변형) 자동으로 반영된다.

    momentum_observer.py의 잔차(residual)를 물리적으로 말이 되는 값과
    비교하려면, "허용 토크의 몇 %가 설명되지 않는 외력으로 잡히면 위험한가"
    식으로 이 값에 비율(margin_ratio)을 곱해서 하드리밋을 정하는 것을 권장.
    (전체 정격 토크 100%가 채워질 때까지 기다리는 건 이미 너무 늦다.)

    Args:
        model: build_arm_dynamics_model()이 반환한 모델.
        joint_ids: 같은 함수가 반환한 조인트 인덱스 목록.

    Returns:
        joint_ids와 같은 순서의 effort limit 배열 (단위: Nm, revolute
        1-DOF 조인트 기준).
    """
    limits = []
    for jid in joint_ids:
        idx_v = model.joints[jid].idx_v
        limits.append(model.effortLimit[idx_v])
    return np.array(limits)
