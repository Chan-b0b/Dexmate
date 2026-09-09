"""테스트 공용 헬퍼.

pytest를 쓰지 않는 이유: 이 프로젝트에는 테스트 프레임워크 의존성이 없고,
테스트 성격 자체가 "실제 로봇/실제 vLLM 서버를 상대로 손으로 돌려보는
스크립트"에 가깝다. 각 파일을 그냥 `python tests/test_xxx.py`로 실행하고,
전부 돌리려면 `python tests/run_all.py`를 쓴다.

프로젝트 모듈은 flat import(`from config import ...`)를 쓰므로, 이 모듈을
import하는 것만으로 프로젝트 루트가 sys.path에 들어간다.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 실제 head_camera left_rgb에서 받은 프레임(960x600 BGR). 합성 이미지를 쓰지
# 않는 이유는 VLM 워밍업 비용이 이미지 해상도별로 따로 발생하기 때문이다
# (config.py의 warmup_timeout_s 주석 참고) - 실제 해상도로 테스트해야 한다.
FIXTURE_FRAME = Path(__file__).resolve().parent / "fixtures" / "head_camera_frame.jpg"

_failures = 0


def check(label: str, condition: bool) -> None:
    """단정문 하나를 기록한다. 실패해도 즉시 중단하지 않고 계속 진행한다."""
    global _failures
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        _failures += 1


def summary() -> int:
    """결과를 출력하고 프로세스 종료 코드를 돌려준다."""
    print()
    print("모두 통과" if _failures == 0 else f"{_failures}건 실패")
    return 1 if _failures else 0


def load_fixture_frame():
    """실제 카메라 프레임 fixture를 BGR ndarray로 읽는다."""
    import cv2

    frame = cv2.imread(str(FIXTURE_FRAME))
    if frame is None:
        raise SystemExit(f"fixture를 읽을 수 없습니다: {FIXTURE_FRAME}")
    return frame


def vlm_server_available(timeout_s: float = 3.0) -> bool:
    """vLLM 서버가 떠 있고 config의 모델 id를 서빙하는지 확인한다."""
    from config import VLMConfig

    cfg = VLMConfig()
    try:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                        timeout=timeout_s, max_retries=0)
        return any(m.id == cfg.model for m in client.models.list().data)
    except Exception:  # noqa: BLE001 - 서버가 없으면 그냥 스킵하면 된다
        return False


def skip_without_vlm_server() -> None:
    """서버가 필요한 테스트에서 호출. 서버가 없으면 SKIP으로 정상 종료한다."""
    from config import VLMConfig

    if not vlm_server_available():
        cfg = VLMConfig()
        print(f"SKIP: {cfg.base_url} 에서 모델 '{cfg.model}'을 찾을 수 없습니다.")
        print("      이 테스트는 실제 vLLM 서버가 필요합니다 (README 실행 전 준비 참고).")
        sys.exit(0)
