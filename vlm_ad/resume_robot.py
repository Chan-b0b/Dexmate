"""Software E-Stop을 해제(재개)하는 전용 도구.

의도적으로 safety_supervisor.py나 main.py 안에 자동 해제 로직을 넣지 않았다.
정지는 소프트웨어가 판단해서 걸 수 있지만, 재개는 반드시 사람이 상황을
확인한 뒤 명시적으로 실행해야 한다 - 그래서 이 스크립트는:

  1. 현재 E-Stop 상태와 로봇 정보를 먼저 보여준다.
  2. 정확한 확인 문구를 타이핑해야만 해제를 진행한다 (오타 방지, 스크립트로
     자동화해서 실수로 해제하는 것을 막기 위함 - 엔터 한 번으로 넘어가지
     않도록 의도적으로 번거롭게 만들었다).
  3. 해제 호출 후 실제로 상태가 바뀌었는지 is_software_estop_enabled()로
     다시 확인한다 (서비스 콜 자체가 조용히 실패할 수 있다는 걸 이미
     겪었으므로 - misc.py 확인 결과와 동일한 이유).

사용법:
    python resume_robot.py
"""

from __future__ import annotations

import sys

from dexcontrol.robot import Robot

_CONFIRMATION_PHRASE = "RESUME ROBOT"


def main() -> None:
    print("=" * 60)
    print("Software E-Stop 해제 도구")
    print("=" * 60)

    bot = Robot()

    print("\n현재 상태:")
    bot.estop.show()

    is_enabled = bot.estop.is_software_estop_enabled()
    print(f"\nis_software_estop_enabled() = {is_enabled}")

    if not is_enabled:
        print("\n현재 E-Stop이 걸려있지 않습니다. 해제할 게 없습니다.")
        bot.shutdown()
        return

    print(
        "\n로봇이 왜 정지했는지 반드시 원인을 확인한 뒤 진행하세요 "
        "(로그, 카메라 영상, 주변 상황 등).",
    )
    print(f'계속하려면 정확히 다음 문구를 입력하세요: "{_CONFIRMATION_PHRASE}"')
    user_input = input("> ").strip()

    if user_input != _CONFIRMATION_PHRASE:
        print("확인 문구가 일치하지 않습니다. 해제를 취소합니다.")
        bot.shutdown()
        sys.exit(1)

    print("\nE-Stop 해제를 시도합니다...")
    bot.estop.deactivate()

    # 서비스 콜 자체가 조용히 실패할 수 있으므로 (misc.py 확인 결과와 동일한
    # 이유), 반드시 실제 상태로 재확인한다.
    still_enabled = bot.estop.is_software_estop_enabled()
    if still_enabled:
        print(
            "\n⚠️  경고: deactivate()를 호출했지만 상태가 여전히 True입니다. "
            "해제가 실제로 반영되지 않았을 수 있습니다. 다시 시도하거나 "
            "물리적 상태를 직접 확인하세요.",
        )
        bot.shutdown()
        sys.exit(1)

    print("\n✅ E-Stop이 해제됐습니다 (is_software_estop_enabled() = False).")
    bot.shutdown()


if __name__ == "__main__":
    main()
