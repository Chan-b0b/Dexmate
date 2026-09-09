"""테스트 전체 실행.

    python tests/run_all.py            # 요약만
    python tests/run_all.py -v         # 각 테스트 출력까지

test_warmup.py만 실제 vLLM 서버가 필요하고, 없으면 SKIP된다.
어느 테스트도 실제 로봇을 필요로 하지 않는다 (전부 가짜 로봇).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TIMEOUT_S = 300


def main() -> int:
    verbose = "-v" in sys.argv
    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    results: list[tuple[str, str]] = []

    for path in test_files:
        print(f"=== {path.name} ===")
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        out = proc.stdout
        if verbose:
            print(out, end="" if out.endswith("\n") else "\n")
            if proc.stderr:
                print(proc.stderr, end="")

        if "SKIP:" in out:
            status = "SKIP"
            detail = next((l for l in out.splitlines() if l.startswith("SKIP:")), "")
            if not verbose:
                print(f"  {detail}")
        elif proc.returncode == 0:
            status = "PASS"
            n = out.count("  PASS  ")
            if not verbose:
                print(f"  {n}건 통과")
        else:
            status = "FAIL"
            if not verbose:
                for line in out.splitlines():
                    if line.startswith("  FAIL"):
                        print(line)
        results.append((path.name, status))

    print()
    print("=" * 60)
    for name, status in results:
        print(f"  {status:5s}  {name}")
    failed = [n for n, s in results if s == "FAIL"]
    print("=" * 60)
    if failed:
        print(f"실패: {', '.join(failed)}")
        return 1
    print("전체 통과" + (" (일부 SKIP)" if any(s == "SKIP" for _, s in results) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
