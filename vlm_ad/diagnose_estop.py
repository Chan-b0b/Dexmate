"""estop_querier(서비스 콜) 문제를 진단하는 스크립트.

확인하는 것:
  1. is_software_estop_enabled()로 실제 상태 토픽을 직접 읽어서, activate()
     호출이 로그상 실패해도 실제로는 상태가 바뀌었는지 확인 (서비스 콜과
     상태 토픽은 독립된 채널임 - misc.py 확인).
  2. activate() 호출의 왕복 시간을 측정해서, 50ms 타임아웃에 얼마나
     자주/얼마나 근접하게 걸리는지 본다.
  3. 이 스크립트를 단독으로(우리 safety_supervisor 없이) 실행했을 때와,
     safety_supervisor가 같은 프로세스에서 돌고 있을 때를 비교하면
     GIL 경쟁이 원인인지 아닌지 감이 온다.

사용법:
    python diagnose_estop.py          # 단독 실행 (기준선)
    # 그리고 실제 감시 시스템이 돌아가는 상태에서 별도 프로세스로 다시 실행
    # (같은 프로세스가 아니라 별도 프로세스로 돌려도, 로봇 쪽 estop 서비스가
    #  다른 이유로 바쁜지 아닌지는 확인 가능)
"""

from __future__ import annotations

import time

from dexcontrol.robot import Robot


def measure_round_trip(bot, n_calls: int = 10) -> None:
    print(f"\n--- activate()/deactivate() 왕복 시간 측정 ({n_calls}회) ---")
    times = []
    for i in range(n_calls):
        state_before = bot.estop.is_software_estop_enabled()

        t0 = time.perf_counter()
        bot.estop.activate()
        elapsed_activate = time.perf_counter() - t0

        # 상태 토픽에 반영될 시간을 잠깐 줌 (서비스 콜 응답과 상태 퍼블리시는
        # 다른 채널이라 약간의 지연이 있을 수 있음)
        time.sleep(0.05)
        state_after_activate = bot.estop.is_software_estop_enabled()

        t0 = time.perf_counter()
        bot.estop.deactivate()
        elapsed_deactivate = time.perf_counter() - t0

        time.sleep(0.05)
        state_after_deactivate = bot.estop.is_software_estop_enabled()

        times.append((elapsed_activate, elapsed_deactivate))

        print(
            f"[{i+1}/{n_calls}] "
            f"activate={elapsed_activate*1000:.1f}ms "
            f"(상태: {state_before}->{state_after_activate}) | "
            f"deactivate={elapsed_deactivate*1000:.1f}ms "
            f"(상태: {state_after_activate}->{state_after_deactivate})",
        )

        # 상태가 호출한 대로 바뀌지 않았으면 명확히 표시
        if state_after_activate is not True:
            print("  ⚠️  activate() 호출했지만 상태가 True로 안 바뀜 - 진짜 실패 가능성")
        if state_after_deactivate is not False:
            print("  ⚠️  deactivate() 호출했지만 상태가 False로 안 바뀜 - 진짜 실패 가능성")

    activate_times = [t[0] for t in times]
    deactivate_times = [t[1] for t in times]
    print("\n--- 요약 ---")
    print(
        f"activate()   평균={sum(activate_times)/len(activate_times)*1000:.1f}ms  "
        f"최대={max(activate_times)*1000:.1f}ms",
    )
    print(
        f"deactivate() 평균={sum(deactivate_times)/len(deactivate_times)*1000:.1f}ms  "
        f"최대={max(deactivate_times)*1000:.1f}ms",
    )
    over_50ms = sum(1 for t in activate_times + deactivate_times if t > 0.05)
    print(f"50ms 타임아웃을 넘긴 호출: {over_50ms}/{len(activate_times)*2}")
    if over_50ms > 0:
        print(
            "-> 50ms를 자주 넘긴다면, 이건 실제 서비스 장애가 아니라 이 "
            "타임아웃 자체가 너무 짧다는 신호일 가능성이 높습니다 (특히 다른 "
            "무거운 Python 작업이 같은 프로세스에서 GIL을 붙잡고 있을 때).",
        )


def main() -> None:
    bot = Robot()
    print("estop.show():")
    bot.estop.show()

    measure_round_trip(bot, n_calls=10)

    # 마지막엔 반드시 정지 해제 상태로 되돌려서 로봇이 계속 estop 상태로
    # 남아있지 않게 함
    bot.estop.deactivate()
    print("\n진단 종료 - E-Stop 해제 상태로 복귀")
    bot.shutdown()


if __name__ == "__main__":
    main()
