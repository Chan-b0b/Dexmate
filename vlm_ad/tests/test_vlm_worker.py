"""vlm_worker.AsyncVerifier 검증 (VLM 서버/로봇 불필요 - 판정 함수를 가짜로 대체).

확인하는 것:
  1. 느린 판정이 진행되는 동안 poll()이 감시 루프를 막지 않는다.
  2. 처리 중 들어온 요청은 버린다 (백프레셔).
  3. deadline을 넘긴 요청은 포기하고 판단 불가로 돌려주며, 뒤늦은 응답은 폐기한다.
  4. 판정 함수가 예외로 죽어도 in-flight가 영구 고착되지 않는다.
"""

import sys
import time

from _harness import check, summary

import numpy as np
from vlm_client import VLMVerdict
from vlm_worker import AsyncVerifier

FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


def fail_verdict(reason: str) -> VLMVerdict:
    return VLMVerdict(False, 0.0, reason, ok=False)


def slow_verdict(_frame, _ctx, sleep_s: float = 2.0) -> VLMVerdict:
    time.sleep(sleep_s)
    return VLMVerdict(True, 0.9, "사람 근접", ok=True)


# --- 1) 느린 판정 중에도 poll()이 루프를 막지 않는다 -------------------------
print("[1] 2초 걸리는 판정 동안 감시 루프가 계속 도는가")
w = AsyncVerifier("slow", slow_verdict, fail_verdict, deadline_s=10.0)
check("submit 수락", w.submit(FRAME, "테스트") is True)

ticks, verdict, worst_tick = 0, None, 0.0
t0 = time.monotonic()
while time.monotonic() - t0 < 3.0:
    tick_start = time.monotonic()
    v = w.poll()                      # 감시 루프 한 틱에 해당
    worst_tick = max(worst_tick, time.monotonic() - tick_start)
    if v is not None and verdict is None:
        verdict = (v, time.monotonic() - t0)
    ticks += 1
    time.sleep(1 / 15)                # filter_rate_hz = 15Hz

check(f"3초 동안 {ticks}틱 (동기 호출이었다면 ~15틱)", ticks >= 40)
check(f"poll() 최대 소요 {worst_tick*1000:.2f}ms (블록 안 함)", worst_tick < 0.01)
check(f"판정 도착 t={verdict[1]:.2f}s, is_anomaly={verdict[0].is_anomaly}",
      verdict is not None and verdict[0].is_anomaly and 1.9 < verdict[1] < 2.5)
check("수거 후 poll()은 None", w.poll() is None)

# --- 2) 처리 중 재요청은 버린다 (백프레셔) ---------------------------------
print("[2] 처리 중 들어온 요청은 버리는가")
w2 = AsyncVerifier("busy", slow_verdict, fail_verdict, deadline_s=10.0)
first = w2.submit(FRAME, "1차")
second = w2.submit(FRAME, "2차")
check("첫 요청 수락 / 두번째 거절", first is True and second is False)
check("busy() True", w2.busy() is True)

# --- 3) deadline 초과 -> 판단 불가 + 뒤늦은 결과 폐기 -----------------------
print("[3] 워커가 안 돌아올 때 요청을 포기하고 판단 불가로 처리하는가")


def hung_verdict(_frame, _ctx) -> VLMVerdict:
    time.sleep(2.5)                                  # deadline보다 오래
    return VLMVerdict(False, 0.0, "뒤늦은 정상판정", ok=True)


w3 = AsyncVerifier("hung", hung_verdict, fail_verdict, deadline_s=1.0)
w3.submit(FRAME, "테스트")
t0 = time.monotonic()
v = None
while v is None and time.monotonic() - t0 < 2.0:
    v = w3.poll()
    time.sleep(1 / 15)
elapsed = time.monotonic() - t0
check(f"{elapsed:.2f}s 후 판단 불가 반환 (ok=False)", v is not None and v.ok is False)
check(f"deadline(1.0s) 직후에 포기: {elapsed:.2f}s", 1.0 <= elapsed < 1.3)
check("포기 후 busy() False - 새 요청 가능", w3.busy() is False)

time.sleep(2.0)  # 버려진 워커가 뒤늦게 끝나는 시점까지 대기
check("뒤늦은 결과가 판정으로 새어나오지 않음", w3.poll() is None)

# --- 4) verify_fn이 예외를 던져도 in-flight가 영구 고착되지 않는다 ----------
print("[4] 판정 함수가 예외로 죽어도 복구되는가")
print("     (아래 traceback은 의도된 것 - 워커가 예외를 판단 불가로 흡수하는지 본다)")


def boom(_frame, _ctx) -> VLMVerdict:
    raise RuntimeError("연결 끊김")


w4 = AsyncVerifier("boom", boom, fail_verdict, deadline_s=1.0)
w4.submit(FRAME, "테스트")
t0 = time.monotonic()
v = None
while v is None and time.monotonic() - t0 < 1.0:
    v = w4.poll()
    time.sleep(0.01)
check(f"예외를 판단 불가로 변환: ok={v.ok if v else None} reason={v.reason if v else None}",
      v is not None and v.ok is False and "워커 예외" in v.reason)
check("in-flight 해제됨 (고착 없음)", w4.busy() is False)

sys.exit(summary())
