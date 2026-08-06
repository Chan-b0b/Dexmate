# ik_VLM — 스크립트 IK 위의 감시·응급대처 계층

명목 거동은 **100% 기존 ik_demo 스크립트**다. 이 패키지는 그 위에서

- **열거하지 않은 이상**(open-set)을 힘 envelope 이탈로 감지하고 (`monitor.py`)
- 감지 즉시 **hold + 소폭 리프트**(Tier 0, 자동), 이후 **오퍼레이터 승인** 하에
- **재탐지 → 세계상태 분류 → 스크립트 재진입**(Tier 1, `world_state.py` + `resume_matrix.py`)
- 규칙으로 안 풀리는 상황은 **로컬 VLM이 스킬 제안**(Tier 2, `vlm_advisor.py`, 항상 승인 게이트)

를 수행한다. 원인을 몰라도 "다시 보고, 올바른 지점에서 다시 시작"이 통하는 상황은
전부 Tier 1이 흡수하고, 그 밖은 VLM/오퍼레이터로 에스컬레이션한다.

```
      [ik_demo 스크립트 (불변)]───place(tick_cb=훅)──┐ 틱 단위 abort
[SignalTap 50Hz]→[Monitor envelope]──trip──→[Tier0 hold+3cm]─승인─→
      →[lift→재탐지→WorldState]→[resume_matrix]→ retry_place/pick
                                        └─못 풀면→[VLM 스킬 제안]─승인─→실행
                                                     └─→ 오퍼레이터
```

## ik_demo에 가한 유일한 변경

`suction.py`: `place()` / `_descend_to_contact()`에 옵셔널 `tick_cb` 1개.
틱마다 `tick_cb(ee_z, f)`를 호출해 truthy면 halt 후 `reason="monitor_abort"`로
반환(파트는 계속 흡착 유지, release/오퍼레이터 게이트 없음 — 리커버리는 이 계층 소유).
`tick_cb`를 안 주면 기존 동작과 완전 동일.

## 파일

| 파일 | 역할 |
|---|---|
| `config.py` | 모든 임계값·경로·VLM 설정 |
| `signals.py` | 데몬 스레드 50Hz tared wrench 피처 탭 + phase 라벨 + jsonl 로그 |
| `monitor.py` | phase별 mean+kσ 상한 envelope, 연속 N틱 이탈 시 trip |
| `envelope_build.py` | 정상 런 로그 → `envelope.json` (오프라인) |
| `world_state.py` | seal 원샷 + BEV 재탐지 → `WorldState` |
| `resume_matrix.py` | WorldState → 재진입 결정 (retry_place/pick/vlm/operator) |
| `recovery.py` | Tier 0/1/2 실행기 (모든 후속 조치는 승인 게이트) |
| `vlm_advisor.py` | 로컬 VLM(OpenAI 호환) 스킬 제안, 실패 시 항상 operator로 강등 |
| `supervisor.py` | 오케스트레이터: `sup.place(...)` / `sup.pick(...)` / `--tap-test` |
| `replay_test.py` | 기록 데이터로 오탐률 검증 (로봇 불필요) |

## 배치 순서 (권장)

1. **부트스트랩 envelope** — 기존 정상 레코딩으로:
   ```bash
   cd /home/dexmate/LGES/Dexmate
   python -m LGES.ik_VLM.envelope_build \
       --from-recordings 'LGES/recordings/2026*/case_pick/*'
   ```
2. **오탐 검증** — 성공 에피소드에서 trip 0이 될 때까지 `--k` / `ENVELOPE_MIN_BAND` 조정:
   ```bash
   python -m LGES.ik_VLM.replay_test --recordings 'LGES/recordings/2026*/case_pick/*'
   ```
3. **탭 테스트 (로봇, 팔 무동작)** — 손으로 컵을 눌러 trip 확인:
   ```bash
   python -m LGES.ik_VLM.supervisor   # --seconds 120
   ```
4. **LOG-ONLY 런** — chassis 데모를 그대로 돌리며 신호 로그만 수집 (abort 없음,
   ik_demo 무수정 — 서브클래스 래퍼):
   ```bash
   python -m LGES.ik_VLM.run_supervised [--gripper] [--auto-move] [--dashboard]
   ```
   런당 `logs/signals_*.jsonl` 1개. 정상 런의 로그만 모아 라이브 envelope 재빌드:
   ```bash
   python -m LGES.ik_VLM.envelope_build --from-signals 'LGES/ik_VLM/logs/*.jsonl'
   python -m LGES.ik_VLM.replay_test    --signals       'LGES/ik_VLM/logs/*.jsonl'
   ```
   실패/개입이 있던 런의 로그 파일은 빼고 빌드할 것 (envelope = 정상의 정의).
5. **arm** — envelope 복구 후 실기. 의도적 교란(하강 중 케이스 밀기)으로
   trip→hold→재진입 확인.

## chassis_sequence 통합 (1줄 교체)

```python
# run() 초입에서:
from ..ik_VLM.supervisor import Supervisor
sup = Supervisor(bot, mover); sup.start()

# run_item()의
#   pres = mover.place(place_pose, expected_z=exp_z, misseat_tol_m=mtol)
# 를 다음으로 교체:
pres = sup.place(place_pose, label=label, station="target", layers=tgt_layers,
                 expected_z=exp_z, misseat_tol_m=mtol, plane_z=tgt_plane,
                 pose_key=pose_key)
# 반환은 동일한 PickResult — ZTracker 로깅 등 하류 코드 무변경.
# 트림 정책까지 재진입에 반영하려면 replan=콜러의 포즈 재계산 클로저 전달.
```

체스시스 스트래프 구간은 `sup.set_phase("transport")` / `sup.set_phase("idle")`로
라벨하면 envelope 키가 맞는다.

## 로컬 VLM 백엔드

OpenAI 호환 엔드포인트면 무엇이든. Jetson에서 Ollama 예:

```bash
ollama pull qwen2.5vl:7b     # config.py: VLM_BASE_URL/VLM_MODEL
```

엔드포인트가 없거나 죽어 있으면 advisor는 자동으로 `call_operator`로 강등된다 —
VLM은 옵션을 **추가**만 할 수 있고 오퍼레이터 폴백을 제거할 수 없다.

## 현재 한계 (v1)

- envelope은 phase별 **정상(stationary) 상한**만 본다: 시간형상(프로파일) 이탈,
  "와야 할 접촉이 안 옴"은 스크립트 자체 가드(max_descent)에 위임.
- pick 하강에는 틱 훅이 없다(스크립트의 force/seal 가드 유지) — pick 이상은
  프리미티브 경계에서 보고된다.
- misseat 리커버리 중 재하강(`_misseat_recover`)은 훅 미적용 — 해당 구간은 기존
  Phase 2 로직이 소유.
- 재코딩 부트스트랩 envelope은 10-15Hz 데이터라 `df_mag` 분산이 라이브(50Hz)와
  다르다 — LOG-ONLY 런으로 라이브 envelope을 만든 뒤 arm할 것.
- 낙하 파트 자동 재파지는 의도적으로 없음(`vlm`/오퍼레이터 경로).
