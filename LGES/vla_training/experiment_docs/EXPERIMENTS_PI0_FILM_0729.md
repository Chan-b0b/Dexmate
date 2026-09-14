# LGES case_pick 0729 — π0 FiLM 라운드 (토큰 레벨 주입 지점 검증)

작성: 2026-08-12. B300 서버(`/home/maverick/Dexmate/LGES/vla_training`)에서 수행.
SmolVLA의 prefix>suffix 결과를 π0에서 **토큰 단위**로 재검증한 라운드의 기록.

## 1. 질문과 배경

SmolVLA 0729 라운드의 결론: FiLM contact-conditioning은 **prefix(state 토큰) 주입 + mask_force=1**
조합에서만 접촉 정지 권한을 갖는다. 이것이 아키텍처 특수성인지 확인하기 위해:

- **π0.5 (이전 라운드)**: state가 텍스트 프롬프트로 양자화되어 **state 토큰이 없음** → suffix만
  가능했고, naive/FiLM 전 arm이 press-sim에서 전멸. "state 토큰 부재"가 원인인지 분리 불가.
- **π0 (이번 라운드)**: `state_proj`가 만드는 진짜 dense state 토큰이 있고, STATE 정규화도
  SmolVLA와 같은 MEAN_STD → 채널 캘리브레이션이 그대로 이식됨. 단, π0는 state 토큰과 action
  토큰이 **둘 다 embed_suffix 안**에 있으므로, 이 라운드가 답하는 것은 prefix-vs-suffix가 아니라
  **"어느 토큰을 조건화하느냐가 중요한가" (state vs action)** — SmolVLA 결과의 토큰 레벨 절반.

## 2. 실험 구성 (2026-08-11 학습, 08-12 프로브)

모든 런: `lerobot/pi0_base`에서 시작, bs 8, 50k steps, grad-ckpt, GPU당 ~65GB, ~1.6 step/s ≈ 9h.
FiLM 공통(학습·프로브 동일): `FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1
FILM_FZ_OFF=2.1`, F0/TAU/FZ_TAU=6/4/5 (film_contact 기본값). 데이터: `Chanho-Lee/lges_case_pick_0729`.

| run | inject | best (val loss) | 스크립트 |
|---|---|---|---|
| `pi0_naive_0729` | — | 10000 (0.12983) | `run_pi0_naive_0729.sh` |
| `pi0_film_frombase_state_0729` | **state 토큰** ('prefix' 아날로그) | 20000 (0.13136) | `run_pi0_film_frombase_0729.sh` |
| `pi0_film_frombase_action_0729` | **action 토큰** ('suffix' 아날로그) | 20000 (0.12726) | 〃 |

state/action arm은 FiLM이 붙는 토큰 하나만 다르고 나머지 전부 동일 — 단일 변수 ablation.
(onnaive arm은 의도적으로 보류. 필요 시 naive best@10k에서 warm-start.)

구현: `film_contact_pi0.py`(embed_suffix 패치; state=embs[:,:1], action=embs[:,1:]) +
`train_film_pi0.py`(런처, train_pi05의 relative_actions shim 재사용).

## 3. 헤드라인 결과 — press-sim (배포 시나리오)

press-sim = seal이 끝내 안 잡히는 misalign 상황에서, 물리적으로 자라나는 접촉력(fzdelta:
own wrench + fz에만 delta0+k·p)만 보고 하강을 멈추는가. off1=정상 타이밍, off30=2초 이른 접촉.
**모든 수치는 best ckpt** (이후 표 동일).

| arm | off1 정지 (관통) | off30 정지 (관통) |
|---|---|---|
| π0 naive | 0/6 (17.0mm) | 5/6 (8.4mm) |
| **π0 film-state** | **5/6 (7.2mm)** | 5/6 (15.3mm) |
| π0 film-action | 0/6 (22.9mm) | 0/6 (57.2mm, max 91mm) |

비교 맥락 (같은 프로브·같은 val 6 eps; naive 셀은 FiLM 캘리브레이션 무관이라 크기 비교 정당):

| | off1 | off30 |
|---|---|---|
| SmolVLA naive | 4/6 (9.8mm) | 5/6 (5.3mm) |
| SmolVLA film-prefix (recal, fromnaive) | 4/6 (8.1mm) | **6/6 (2.6mm)** |
| SmolVLA film **v1 컨트롤** (c-hat 셔플) | **0/6 (51.3mm)** | — |
| π0.5 naive | 0/6 (39.6mm) | 0/6 (66.2mm) |
| π0.5 film-suffix (fb/on) | 0/6 (79.6/89.4mm) | 0/6 (71/73mm) |

**주장할 수 있는 것**
1. **주입 토큰이 접촉 정지의 인과 요인** — π0 단일 변수 ablation에서 off1 5/6@7.2mm vs 0/6@22.9mm.
2. **아키텍처 관통 패턴** — 정지가 되는 조합은 정확히 "FiLM이 state 토큰 위"일 때뿐
   (SmolVLA prefix, π0 state). action 토큰(π0)이나 state 토큰 부재(π0.5)는 실패.
3. **컨트롤 완비** — SmolVLA v1(셔플 c) 0/6 → 신호가 원인이지 모듈 추가가 아님;
   π0 state-authority의 dRaw=0.00 → mask 병목 검증.

**같이 말해야 하는 것**
- π0 naive의 off30 5/6@8.4mm: π0 raw 경로는 이른(큰) 접촉 신호에는 반응함. FiLM-state의
  이득은 정상 타이밍 off1에 집중.
- film-state off30 15.3mm는 z~0.817 한 에피소드(28.3mm)가 견인 — naive(8.4mm)보다 깊음.
- n=6 eps, 강성 1N/mm 단순 모델, 실로봇 아님. 최종 클레임은 로봇 롤아웃 필요.

## 4. 보조 결과

### state-authority (접촉 counterfactual, 절대 dz mm/step, best)

접촉 직전 10프레임(n=60)에 first-contact 힘 패턴(wrench만, seal=0 유지)을 주입. 참고 스케일:
expert 접촉 직전 ~-1.4, 정지=0. FiLM arm은 2×2 factorial로 dTotal=dRaw+dFiLM 분해, **dRaw=0.00**.

| cell | naive | film-state | film-action |
|---|---|---|---|
| pc_fc (12N 템플릿) | -2.26 → **-0.45** | -1.62 → -1.12 | -1.74 → -1.54 |
| pc_r12 (12N 리스케일) | -2.26 → **-0.11** | -1.62 → -0.74 | -1.74 → -1.41 |
| ramp8 → ramp12 (전 하강, n=739) | -3.71 → **-0.92** | -3.02 → -3.02 | -3.44 → -3.34 |

- naive: 12N 템플릿에는 사실상 정지 수준의 단발 반응. 단 ramp8(-3.71)→ramp12(-0.92)의 점프가
  보여주듯 훈련 분포 크기(|F|≈12N)에 잠긴 템플릿 매칭 — press-sim의 비템플릿 신호(off1)에는 실패.
- film-state: 단발로는 감속(31–54%)이지만 closed-loop 누적으로 press-sim에서 정지. 8N/12N 동일
  반응(-3.02/-3.02)은 contact 채널이 F0=6N 근처에서 포화되기 때문 (크기 구분 없음, 문턱 반응).
- film-action: 절대값으로도 미미(11–19% 감속) — sim 전멸과 일관.

### eval_offline (open-loop 매핑 품질, best / last)

| arm | pos(mm) | rot(mrad) | suction |
|---|---|---|---|
| naive | 1.03 / 0.70 | 1.82 / 1.46 | 99.3 / 99.8% |
| film-state | 0.93 / 0.71 | 1.71 / 1.48 | 99.7% |
| film-action | 0.88 / 0.75 | 1.65 / 1.50 | 99.7 / 99.8% |

세 arm 동등 — FiLM이 기본 매핑을 해치지 않음 (SmolVLA·π0.5 naive와도 동일 수준).

## 5. 재현

```bash
# 학습 (GPU 2장 + 1장; 각 스크립트가 종료 시 select_best_ckpt --prune까지 수행)
./run_pi0_film_frombase_0729.sh        # state(GPU4) + action(GPU6)
GPU=5 ./run_pi0_naive_0729.sh

# 프로브 배터리 (arm당 GPU 1장 병렬 가능; probes/0729_*_pi0*.txt 생성)
GPU=4 RUNS=naive  ./probe_0729_pi0_server.sh
GPU=5 RUNS=state  ./probe_0729_pi0_server.sh
GPU=6 RUNS=action ./probe_0729_pi0_server.sh
```

프로브 라우팅: `--film-pi0` (probe_state_authority/probe_press_sim/eval_offline, 2026-08-12 추가).
`film_contact_pi0`가 `_condition_from_state`를 자기 네임스페이스로 import하므로 forced-c 훅은
`film_contact_pi0._condition_from_state`에 **별도로** 설치됨 (film_contact만 패치하면 조용한 no-op).

## 6. 주의사항 / 이 라운드에서 고친 것

1. **select_best_ckpt.py의 FiLM 로드 버그 수정 (pi0 라우팅 추가)**: 기존에는 pi0/pi05 FiLM
   ckpt의 `model.contact_film.*`이 unexpected key로 버려진 채 naive 백본으로 val loss를 쟀음.
   pi0는 `film_contact_pi0` 패치 후 로드하도록 수정. **π0.5 0729의 best 선택(예: onnaive
   best=010000)은 이 버그 영향 하에 결정된 것** — pi05 쪽은 수정하지 않고 기록만 남김.
2. **eval_offline 캘리브레이션**: eval_offline의 env 기본값은 F0/TAU/FZ_TAU=12/10/30으로 학습
   (6/4/5)과 다름. π0 배터리(`probe_0729_pi0_server.sh`)는 이를 명시적으로 export하지만,
   **π0.5 배터리의 eval 셀은 12/10/30으로 잘못 돌았음** (state/sim 셀은 정상).
3. FiLM 셀의 크기 비교는 같은 아키텍처 안에서만. 아키텍처 간에는 naive 셀 크기와 FiLM 셀의
   sign/shape만 비교할 것 (π0.5 배터리 헤더와 동일한 원칙).
4. π0/π0.5 학습·로드는 `train_pi05.py`의 relative_actions_processor shim 필요 (lerobot 0.5.1).

## 7. 다음 단계 후보

- 실로봇 롤아웃으로 press-sim 결과 확증 (특히 off1 시나리오의 naive vs film-state).
- π0 v1 컨트롤 arm (c-hat 셔플) — SmolVLA v1과 같은 인과 컨트롤을 π0에서도.
- onnaive arm (naive best@10k warm-start) — frombase와의 수렴/권한 비교.
