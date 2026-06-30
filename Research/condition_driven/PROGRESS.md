# Condition-Driven Transitions — 진행 요약

_최종 수정: 2026-06-25_

---

## 1. 배경

**태스크 특성:**
- case_pick — 스택에서 케이스를 흡착해 들어 올리는 작업
- 스택이 소진될수록 물체 높이가 낮아짐
- phase 구조: approach → descend → grasp(suction) → lift
- 핵심 전이는 "언제 하강을 멈추고 흡착을 켜느냐" — 이 순간이 force에 의존해야 함

실제 로봇을 돌리다 두 가지 현상을 관찰한 데서 출발:
- 흡착이 안 되었는데도 lift를 시작함
- contact이 이뤄졌음에도 계속 하강함 (push too hard)

두 경우 모두, force/suction 신호를 보면 상태가 명확함에도 모델이 이를 무시하고
시간/자세 기반으로 phase를 전환한 것으로 해석됨.
VLA가 phase를 전환할 때 진짜 물리 조건을 보는 게 아니라 학습된 상관변수(시간, 자세, 깊이)에만 의존한다는 의심으로 이어짐.

**구체적인 문제:**
- vision/pose 관점에서는 몇 mm 차이가 거의 구분이 안 됨.
- 하지만 suction/force 관점에서는 그 몇 mm가 흡착 성공/실패를 가름.
- 모델은 이미 wrench(force)를 observation으로 받고 있음에도, action 결정에 실제로 쓰지 않음.

**왜 이게 중요한가:**
- 실제 사용자는 로봇이 동작할 때 이런 물리적 단서를 자연스럽게 참고함("접촉됐으면 올려라").
  VLA도 같은 방식으로 동작해야 신뢰할 수 있음.
- 데이터를 많이 쌓기 어려운 상황에서 더욱 그러함. 다양한 조건을 커버하는 데이터를 확보하기 힘들기
  때문에, 물리 조건을 명시적으로 참조하게 만드는 것이 데이터 효율 면에서도 유리함.

**모델 I/O:**
- Input:
  - 이미지: `camera1` (RGB head), `camera2` (depth colorized)
  - state 15D (EE pos/quat, suction, seal, wrench fx~tz)
  - task instruction (tokenized)
- Output: action 7D — Δpos (3D) + Δrot rotvec (3D) + suction (1D), 50-step chunk
- 모델: SmolVLA (SmolVLM2-500M 기반), lerobot 0.5.1로 파인튜닝

---

## 2. 핵심 가설

저데이터 VLA는 phase **전이**를 진짜 **물리 조건**(접촉/실링/정렬)이 아니라
학습된 상관변수(시간/자세/깊이)에 게이팅함.

이를 case+battery 흡착 데모에서 특성화하고,
action 전이를 추론된 **접촉 조건 `ĉ`에 grounding**하는 학습형 수정을 만듦.
개선이 *조건 grounding 덕분*이지 추가 용량이나 데이터 때문이 아님을 보임.

**현재 범위:** approach→descend→grasp 전이 하나, 관측 가능한 조건 = **접촉력**.
태스크/모델 하나 — case_pick 흡착, lerobot로 파인튜닝한 SmolVLA.

---

## 3. 진단 결과

바닐라 SmolVLA는 **under-reach**: 물체 깊이와 무관하게 하강이 습관적 깊이(~0.82–0.83 m)에서 멈춤.

depletion sweep의 ungated seal%(층 0→4, 위→아래): **70 / 60 / 0 / 20 / 0** — 스택이 소진되어
물체가 깊어질수록 붕괴.

**특권 접촉-게이트**(접촉할 때까지 강제 하강, *같은 모델, 재학습 없음*)가 깊은 층을 회복:
L2 0→80, L3 20→100, L4 0→90.

→ 실패는 **조건 무시(condition-ignoring)**이지 능력/데이터/OOD 문제가 아님.

**Confound 배제:**
- 실링 센서 검증(DI0); 모든 실패에서 흡착은 *명령됨*
- 데모가 모든 깊이를 포함함(OOD 아님)

**하드 게이트의 잔여 실패:** 과압(over-press) — |F| ~16–19 N (데모 ~15.8 대비).
"너무 세게 누름"이며, 별개 모드가 아님(사용자 관찰).

---

## 4. 방법 — FiLM(ĉ) → action

`LGES/vla_training/film_contact.py`

1. **`ĉ`** = `clip((|F| − F0)/τ, 0, 1)`, F0=14 N, τ=3 N.
   학습 head 없이 직접 계산(접촉은 관측 가능). 연속값으로 graded force 정보 보존.

2. **Force-mask (보틀넥):** action 경로의 wrench 차원(state idx 9:15)을 0으로 마스킹.
   접촉이 action에 도달하는 경로가 **오직 `ĉ`뿐** — "전이가 ĉ에 의존한다"를 검증 가능하게 만듦.

3. **FiLM:** MLP 2개(cond_dim→64→hidden)가 γ,β를 생성, action-expert 특징을 변조.
   Zero-init → 시작 시 identity. 파인튜닝이 정확히 베이스 정책에서 출발해 변조를 학습.
   **주입 지점(`inject`, 2026-06-29 변경):** `suffix`(기본) = `embed_suffix`의 action-token
   임베딩(=expert 입력)을 변조 → expert 16개 층이 ĉ 조건 하에 action을 계산(강한 권한).
   `output` = `action_out_proj` 입력 변조(마지막 층 약한 tap, ablation용).

**Config (학습/평가 반드시 일치):** `FILM_COND`(채널 ⊂ {contact,seal}), `FILM_INJECT`
(suffix/output), `FILM_MASK_FORCE`(0/1), `FILM_VARIANT`(v0/v1/v2). cond/inject는 structural.

**실험 변형:**
- V0 = 바닐라(패치 없음)
- V1 = 마스킹 + **decorrelated `ĉ`**(배치 셔플) — 용량 동일, grounding만 제거
- V2 = 마스킹 + **진짜 `ĉ`** — 본 방법

### 4b. 1차 FiLM 결과 + 라이브 진단 (2026-06-29)
- **output-tap 주입은 안 먹혔음.** 학습된 γ 권한이 1.4%(contact)~17%(contact+seal)로 미미.
  action 결정은 상류(expert)에서 끝나서 마지막 층 affine으로 못 뒤집음. + 평가가 open-loop
  (n_action_steps=50 → ĉ를 3.3s마다 한 번, 그것도 hover에서) → 반응형 신호가 발화 불가.
- **ĉ 파이프라인은 정상:** 라이브 로그(run_policy `c^=[contact,seal]`)에서 힘이 14N 넘는 순간
  `c^`이 [0,0]→[1,0], seal 시 →[1,1]. obs 버그 아님.
- **contact&seal 시 lift는 베이스 정책에서 이미 됨** (우리 기여 아님).
- **타깃 확정:** descend→stop 전이만 — "contact 없으면 계속 하강, contact면 정지". descend
  국면 안에서는 ĉ가 바로 disambiguator (깊은 물체 @0.79는 c=0→더 내려가, contact는 c=1→정지).
  베이스는 이 stop을 **깊이(ee_z)**로 게이팅 → under-reach. → 주입을 expert 입력으로 옮김(위 3).
- **잔여 caveat:** 깊이 cue가 state에 남아 경쟁. 강한 권한에도 under-reach가 지속되면 다음 레버는
  깊이 cue 억제 또는 gate-oracle rollout 증류(DAgger). contact는 edge-observable이라 그 자체로
  under-reach를 예측적으로는 못 고침.

---

## 5. 실험

| # | 주장 | 검증 | 상태 |
|---|---|---|---|
| 진단 | under-reach + 조건 무시 | ungated vs gate-oracle 층별 seal% | **완료** |
| S1 | under-reach 회복, 과압 없음 | V2 ≈ 오라클, graded \|F\| | 대기 |
| S2 | grounding이 핵심 요인 | **V2 > V1 ≈ V0** | 대기 |
| S3 | action이 ĉ를 사용 (bypass 없음) | 반사실: ĉ=0/1 강제 | 대기 |
| S4 | 저데이터 시그니처 | #demos 대비 이득 | 대기 |
| S5 | 특이성 / 무해성 | 관측 불가능 조건(정렬)엔 ĉ 도움 안 됨 | 대기(삽입 태스크 필요) |

---

## 6. 도구

- `Research/condition_driven/lift_condition_probe.py` — 진단 프로브 (`--by-height`, `--timeline`)
- `Research/condition_driven/seal_monitor.py` — 실링 센서 라이브 검증
- `LGES/vla_training/film_contact.py` — FiLM 패치 (V0/V1/V2)
- `LGES/vla_training/train_film.py`, `train_film.sh` — 파인튜닝 스크립트
- `LGES/vla_training/self_test_film.py` — 유닛 테스트 (통과)
- `LGES/vla_training/run_policy.py --film` — 온로봇 평가; `--descend-until-contact` (gate oracle)

---

## 7. 다음 단계

1. **Smoke test** — `FILM_VARIANT=v2 RUN_NAME=smoke ./train_film.sh --steps=4 --save_freq=2`; finite loss + `contact_film.*` 키 저장 확인.
2. **V2 → V1 학습** (각 20k steps, `smolvla_20260624_081946`에서).
3. **평가** V0/V1/V2 + gate-oracle을 depletion sweep에서; `lift_condition_probe.py --by-height`로 비교. 목표: S1 + S2.
4. 반사실(S3), 저데이터 곡선(S4).
5. **정렬 게이팅 negative control**(S5) — 별도 전이(삽입 태스크)에서 확보.
