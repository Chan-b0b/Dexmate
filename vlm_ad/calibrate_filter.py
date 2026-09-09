"""1차 필터 임계값을 실제 카메라 프레임으로 튜닝하는 도구.

왜 필요한가:
    config.py의 flow_magnitude_threshold(8.0)와 force_delta_threshold_n(15.0)은
    예시값이다. 합성 이미지로는 검증이 불가능하다 - Farneback 광류는 실제
    장면의 구조(질감, 조명, 카메라 흔들림)에 따라 값의 범위가 완전히 달라지고,
    랜덤 노이즈 이미지에서는 의미 있는 값이 나오지 않는다.

    이 임계값이 너무 낮으면 정상 작업 중에도 후보가 계속 잡혀 VLM을 쉬지 않고
    부르게 되고(그리고 쿨다운 분기에서 slow_down -> 사실상 정지), 너무 높으면
    진짜 이상을 놓친다. 그래서 "이 로봇이 이 작업을 정상적으로 할 때 광류가
    실제로 어느 범위인지"를 먼저 재고, 그 위에 임계값을 올려야 한다.

이 도구는 센서만 읽는다. 로봇에 명령(estop/감속/모션)을 보내는 코드는
의도적으로 없다. 로봇이 평소 작업을 하는 동안 옆에서 관찰만 한다.

사용 순서:
    # 1) 정상 기준선 수집 - 반드시 로봇이 "평소 하는 작업"을 하는 동안 돌린다.
    #    로봇이 가만히 있는 상태로 수집하면 광류가 거의 0이라 임계값이
    #    비현실적으로 낮게 나오고, 실전에서 오탐이 폭주한다.
    python calibrate_filter.py record --label normal --duration 300

    # 2) 이상 상황 연출하면서 수집 (안전하게 재현 가능한 것만)
    python calibrate_filter.py record --label person_approach --duration 60

    # 3) 임계값 산출 - 정상 후보율과 이상 검출율을 같이 본다
    python calibrate_filter.py report

    # 4) 저장된 실제 프레임을 VLM에 넣어 2차 판정 오탐 확인
    python calibrate_filter.py replay --label normal

main.py가 돌고 있는 상태에서 같이 실행해도 된다 (zenoh는 다중 구독을 허용).
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import time
from pathlib import Path

import cv2
import numpy as np

from anomaly_filter import LightweightAnomalyFilter
from config import AnomalyFilterConfig, SafetyPolicyConfig, VLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate")

DEFAULT_OUT = Path("calib_data")
CSV_FIELDS = ["t", "flow", "force_delta", "motion_spike", "force_spike", "candidate", "frame"]


# ---------------------------------------------------------------------------
# 1) record - 실제 센서 스트림에서 필터 입력값을 기록
# ---------------------------------------------------------------------------
def cmd_record(args: argparse.Namespace) -> None:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    from robot_interface import DexcontrolAdapter

    cfg = AnomalyFilterConfig()
    out_dir = Path(args.out) / args.label
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"  # main.py와 동일 설정

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.error("카메라 스트림이 활성화되지 않았습니다 - 연결을 확인하세요")
            return

        # velocity/pause 훅을 주지 않는다. 이 도구는 관찰만 하므로 어떤 경로로도
        # 로봇을 멈출 수 없어야 한다.
        adapter = DexcontrolAdapter(robot, monitor_arms=("left", "right"))
        adapter.set_current_task_description(args.task)

        anomaly_filter = LightweightAnomalyFilter(cfg)
        period_s = 1.0 / cfg.filter_rate_hz
        rows: list[dict] = []
        saved_frames = 0
        last_sample_t = -1e9  # 정상 프레임 참조 저장 시각 (첫 프레임은 바로 저장)
        stop = False

        def _handle_sigint(_sig, _frame):  # noqa: ANN001
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle_sigint)

        logger.info(
            "기록 시작: label=%s duration=%.0fs rate=%.0fHz (Ctrl+C로 조기 종료)",
            args.label, args.duration, cfg.filter_rate_hz,
        )
        logger.info("작업 설명(VLM 재생 시 사용): %r", args.task)

        started = time.monotonic()
        last_log = started
        while not stop and time.monotonic() - started < args.duration:
            loop_start = time.monotonic()
            frame = adapter.get_camera_frame()
            force_torque = adapter.get_force_torque()

            candidate, dbg = anomaly_filter.check(frame, force_torque)
            elapsed = loop_start - started

            # 프레임 저장 정책: 후보로 걸린 프레임은 (왜 걸렸는지 눈으로 봐야
            # 하므로) 전부, 정상 프레임은 참조용으로 sample_interval마다.
            # 1920x1200 JPEG이 ~105KB라 전부 저장하면 15Hz에서 분당 90MB다.
            frame_name = ""
            due_for_sample = elapsed - last_sample_t >= args.sample_interval
            if (candidate or due_for_sample) and saved_frames < args.max_frames:
                tag = "cand" if candidate else "norm"
                frame_name = f"{tag}_{elapsed:07.2f}.jpg"
                cv2.imwrite(
                    str(frames_dir / frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                saved_frames += 1
                if due_for_sample:
                    last_sample_t = elapsed

            rows.append({
                "t": f"{elapsed:.3f}",
                "flow": f"{dbg['flow_magnitude']:.4f}",
                "force_delta": f"{dbg['force_delta_n']:.4f}",
                "motion_spike": int(dbg["motion_spike"]),
                "force_spike": int(dbg["force_spike"]),
                "candidate": int(candidate),
                "frame": frame_name,
            })

            if loop_start - last_log >= 1.0:
                last_log = loop_start
                recent = rows[-int(cfg.filter_rate_hz):]
                cand_n = sum(int(r["candidate"]) for r in recent)
                logger.info(
                    "%5.0fs  flow=%6.2f  force_delta=%6.2fN  최근1초 후보 %d/%d  "
                    "누적 후보 %d/%d  프레임 %d장",
                    elapsed, dbg["flow_magnitude"], dbg["force_delta_n"],
                    cand_n, len(recent),
                    sum(int(r["candidate"]) for r in rows), len(rows), saved_frames,
                )

            time.sleep(max(0.0, period_s - (time.monotonic() - loop_start)))

        adapter.shutdown()

    csv_path = out_dir / "samples.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "task.txt").write_text(args.task)
    logger.info("기록 완료: %s (%d샘플, 프레임 %d장)", csv_path, len(rows), saved_frames)
    if saved_frames >= args.max_frames:
        logger.warning("--max-frames(%d) 상한에 걸려 이후 프레임은 저장되지 않았습니다", args.max_frames)


# ---------------------------------------------------------------------------
# 2) report - 정상 후보율 vs 이상 검출율로 임계값 고르기
# ---------------------------------------------------------------------------
def _load(out_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for csv_path in sorted(out_dir.glob("*/samples.csv")):
        label = csv_path.parent.name
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        runs[label] = {
            "flow": np.array([float(r["flow"]) for r in rows]),
            "force": np.array([float(r["force_delta"]) for r in rows]),
            "duration_s": float(rows[-1]["t"]),
            "n": len(rows),
        }
    return runs


def _describe(name: str, values: np.ndarray) -> None:
    qs = [50, 90, 99, 99.9]
    pct = np.percentile(values, qs)
    print(f"    {name:12s} " + "  ".join(f"p{q}={v:7.2f}" for q, v in zip(qs, pct))
          + f"  max={values.max():7.2f}  평균={values.mean():6.2f}")


def cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    runs = _load(out_dir)
    if not runs:
        logger.error("%s 아래에 기록이 없습니다. 먼저 record를 실행하세요.", out_dir)
        return

    cfg = AnomalyFilterConfig()
    normal = runs.get(args.normal_label)

    print("=" * 78)
    print("수집된 기록")
    print("=" * 78)
    for label, run in runs.items():
        mark = " (정상 기준선)" if label == args.normal_label else ""
        print(f"  {label}{mark}: {run['n']}샘플 / {run['duration_s']:.0f}초")
        _describe("flow", run["flow"])
        _describe("force_delta", run["force"])

    if normal is None:
        logger.error(
            "정상 기준선(label=%s)이 없어 임계값을 산출할 수 없습니다.", args.normal_label,
        )
        return

    anomaly_runs = {k: v for k, v in runs.items() if k != args.normal_label}

    for metric, key, current in (
        ("광류(flow)", "flow", cfg.flow_magnitude_threshold),
        ("힘 변화율(force_delta, N)", "force", cfg.force_delta_threshold_n),
    ):
        print()
        print("=" * 78)
        print(f"{metric} 임계값 후보")
        print("=" * 78)
        header = f"  {'임계값':>8}  {'정상 후보율(/분)':>16}"
        for label in anomaly_runs:
            header += f"  {label[:14]+' 검출율':>20}"
        print(header)

        base = normal[key]
        # 후보가 될 수 있는 값의 범위를 정상 분포 위쪽에서 훑는다.
        candidates = sorted({
            round(float(v), 2) for v in np.percentile(base, [90, 95, 99, 99.5, 99.9, 100])
        } | {round(float(base.max()) * m, 2) for m in (1.1, 1.25, 1.5, 2.0)} | {current})

        recommended = None
        for thr in candidates:
            rate = float((base > thr).sum()) / max(normal["duration_s"], 1e-9) * 60.0
            line = f"  {thr:8.2f}  {rate:16.1f}"
            for run in anomaly_runs.values():
                det = float((run[key] > thr).mean()) * 100.0
                line += f"  {det:19.1f}%"
            if thr == current:
                line += "   <- 현재 설정값"
            if recommended is None and rate <= args.max_false_rate:
                recommended = thr
                line += "   <- 권장"
            print(line)

        print(f"\n  권장 임계값: {recommended}  "
              f"(정상 후보율 {args.max_false_rate}/분 이하가 되는 가장 낮은 값)")
        print(f"  현재 설정값: {current}")
        if recommended is not None and abs(recommended - current) > 1e-9:
            rate_now = float((base > current).sum()) / max(normal["duration_s"], 1e-9) * 60.0
            if rate_now > args.max_false_rate:
                print(f"  -> 현재 값은 너무 낮다: 정상 작업 중에도 분당 {rate_now:.1f}회 "
                      "후보가 잡힌다 (오탐 쪽).")
            else:
                print(f"  -> 현재 값은 권장값보다 높다 (정상 후보율 {rate_now:.1f}/분). "
                      "오탐은 적지만 그만큼 놓치는 이상도 늘어난다 - 위 검출율 열을 "
                      "보고 이 작업에서 무엇을 놓치면 안 되는지로 판단할 것.")

    # VLM 호출 부하 확인: 후보율이 그대로 VLM 호출률이 되는 게 아니라
    # 쿨다운과 판정 지연이 상한을 만든다.
    policy = SafetyPolicyConfig()
    print()
    print("=" * 78)
    print("VLM 호출 부하 상한")
    print("=" * 78)
    print(f"  vlm_cooldown_s={policy.vlm_cooldown_s}s, 실측 판정 지연 약 1.55s,")
    print("  동시 in-flight 1개 -> 후보가 계속 잡혀도 VLM 호출은")
    print(f"  최대 약 {60.0 / max(policy.vlm_cooldown_s, 1.55):.0f}회/분으로 제한된다.")
    print("  즉 임계값이 낮아 후보가 폭주하면 호출이 늘기보다, 쿨다운 분기의")
    print("  slow_down()이 계속 호출되는 게 실질적인 문제다 (속도 콜백이 없으면 정지).")


# ---------------------------------------------------------------------------
# 3) replay - 저장된 실제 프레임을 VLM에 넣어 2차 판정 확인
# ---------------------------------------------------------------------------
def cmd_replay(args: argparse.Namespace) -> None:
    from vlm_client import VLMAnomalyVerifier

    run_dir = Path(args.out) / args.label
    frames = sorted((run_dir / "frames").glob("*.jpg"))
    if args.only_candidates:
        frames = [f for f in frames if f.name.startswith("cand_")]
    if not frames:
        logger.error("%s 에 재생할 프레임이 없습니다.", run_dir / "frames")
        return
    frames = frames[: args.limit]

    task_file = run_dir / "task.txt"
    task = task_file.read_text().strip() if task_file.exists() else args.task

    vlm_cfg = VLMConfig()
    policy = SafetyPolicyConfig()
    verifier = VLMAnomalyVerifier(vlm_cfg)

    print(f"프레임 {len(frames)}장을 VLM에 재생 (model={vlm_cfg.model})")
    print(f"작업 설명: {task!r}\n")

    stops = slows = anomalies = failures = 0
    latencies = []
    for i, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        t0 = time.perf_counter()
        v = verifier.verify(frame, task)
        latencies.append(time.perf_counter() - t0)

        action = "-"
        if not v.ok:
            failures += 1
            action = "판정불가"
        elif v.is_anomaly:
            anomalies += 1
            if v.confidence >= policy.stop_confidence_threshold:
                stops += 1
                action = "정지"
            elif v.confidence >= policy.slow_confidence_threshold:
                slows += 1
                action = "감속"
            else:
                action = "로깅만"
        print(f"  [{i:3d}/{len(frames)}] {path.name:24s} {latencies[-1]:5.2f}s  "
              f"anomaly={str(v.is_anomaly):5s} conf={v.confidence:.2f}  {action:8s} {v.reason}")

    print()
    print("=" * 78)
    print(f"  총 {len(frames)}장  |  is_anomaly=True {anomalies}장  "
          f"|  정지 {stops}장  감속 {slows}장  판정불가 {failures}장")
    print(f"  지연: 최소 {min(latencies):.2f}s / 최대 {max(latencies):.2f}s / "
          f"평균 {sum(latencies)/len(latencies):.2f}s "
          f"(타임아웃 {vlm_cfg.request_timeout_s}s)")
    if args.label == "normal" and (stops or slows):
        print(f"  ⚠️  정상 기록인데 {stops+slows}장에서 조치가 발생 = 2차 판정 오탐.")
        print("      프롬프트의 오탐 억제 조건이나 confidence 임계값을 조정해야 한다.")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="데이터 저장 경로")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="실제 센서에서 필터 입력값 기록")
    p_rec.add_argument("--label", required=True, help="기록 이름 (정상 기준선은 normal)")
    p_rec.add_argument("--duration", type=float, default=300.0, help="기록 시간(초)")
    p_rec.add_argument("--task", default="평소 작업 수행 중", help="VLM 재생 시 쓸 작업 설명")
    p_rec.add_argument("--sample-interval", type=float, default=5.0,
                       help="정상 프레임 참조 저장 간격(초)")
    p_rec.add_argument("--max-frames", type=int, default=300, help="저장할 프레임 수 상한")
    p_rec.set_defaults(func=cmd_record)

    p_rep = sub.add_parser("report", help="임계값 산출")
    p_rep.add_argument("--normal-label", default="normal", help="정상 기준선 기록 이름")
    p_rep.add_argument("--max-false-rate", type=float, default=1.0,
                       help="허용할 정상 작업 중 후보율(분당 횟수)")
    p_rep.set_defaults(func=cmd_report)

    p_rep2 = sub.add_parser("replay", help="저장된 프레임을 VLM에 재생")
    p_rep2.add_argument("--label", required=True)
    p_rep2.add_argument("--limit", type=int, default=20, help="재생할 프레임 수")
    p_rep2.add_argument("--only-candidates", action="store_true",
                        help="1차 필터에 걸린 프레임만 재생")
    p_rep2.add_argument("--task", default="평소 작업 수행 중")
    p_rep2.set_defaults(func=cmd_replay)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
