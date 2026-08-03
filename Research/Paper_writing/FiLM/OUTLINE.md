# ICRA 2027 Paper Plan — "Same Loss, No Cause" (working title)

작성 2026-07-30. 오케스트레이터 종합 (증거 인벤토리 → `EVIDENCE.md`).

> **⚠ 수치 상태 (사용자 지시 07-30): 실험 캠페인 진행 중 — 본 문서의 모든 결과값은 예비값
> (placeholder)이다.** 서사·구조·실험 설계가 확정 대상이고, 숫자는 sweep·V1 control·중간층
> 결과가 나오면 일괄 교체한다. 초안 작성도 수치 비의존 섹션(Intro/Related/Method/probe
> 방법론)부터 진행.

**Target venue**: ICRA 2027 — 마감 **2026-09-15** (11:59 PST, PaperPlaza), **8쪽 (참고문헌 포함)**,
**double-anonymous**. Backup: RA-L (+ICRA 2027 presentation option, 이관 마감 2026-12-31) —
분석형 논문에 유리, 페이지 유연.

---

## 1. Nugget (한 문장)

> Behavior-cloned VLA policies silently **bypass** the causal contact signal they are given;
> training loss cannot detect this, and neither architectural access nor exposure fixes it —
> only demonstrations that **decorrelate** the contact condition from its habitual correlates
> restore causal authority, offline and on the robot.

프레이밍: **진단 + 프로브 방법론 + 데이터 처방** 논문. Method(FiLM)는 성능 방법이 아니라
**계측기(instrument)** 로 포지셔닝. "force helps"류(ForceVLA/ForceFlow/PhaForce…)는 2026년
상반기에 포화 — "force가 주어져도 인과적으로 안 쓰이고 loss는 이를 못 본다"는 각도가 블루오션.

## 2. Title candidates

1. *Same Loss, No Cause: Auditing Causal Bypass of Force in Behavior-Cloned VLA Policies*
2. *Loss Is Blind: Counterfactual-Authority Probing of Contact-Gated Transitions in VLAs*
3. *Access Is Not Use: Why VLA Policies Ignore the Force Signal You Give Them*

(B3 원칙: "Same Loss, No Cause"류 캐치프레이즈는 부제/본문 1회로 제한하는 안도 검토.)

## 3. Claim structure (4단 논리)

1. **진단**: 조건 신호가 **전부** state에 concat되어 있어도 — naive도 raw wrench(6)+seal
   bit(DI0)+suction cmd를 받고, FiLM의 contact/fz 채널은 이 wrench에서 계산되는 값이라
   **conditioned 모델이 추가로 받는 정보는 0** (mask1은 오히려 정보를 제거) — vanilla VLA는
   descend→stop 전이를 depth 상관물에 게이팅
   — 습관 깊이(~0.82 m) 정지, depletion sweep seal% 70/60/0/20/0 붕괴; **같은 모델·무재학습**
   privileged contact-gate가 L2 0→80, L3 20→100, L4 0→90 회복 → 실패는 능력/데이터/OOD가
   아니라 **condition-ignoring**. **강조(사용자 지시): 이 태스크는 vision이 전이에 거의 정보가
   없는 세팅** — head cam 단일(손목캠 없음), 층간 높이 차가 원거리 head view에서 거의 구분
   불가, 반면 몇 mm 차이가 seal 성패를 가름 → 인과 신호는 force뿐인 깨끗한 진단 세팅인데도
   BC는 약한 depth/습관 단서에 바인딩(shortcut의 극단적 사례).
2. **Loss-blindness (bypass)**: FiLM 조건화를 붙여도 raw wrench가 남아 있으면(mask0) loss는
   동일(0.099 vs 0.101)한데 counterfactual authority는 0% vs 67% — **loss로는 인과 사용 여부를
   절대 판별 불가**. 판별은 counterfactual authority probe로만 가능.
3. **Shortcut economics**: 채널 분해 probe — "67–82% authority"는 접촉 транз이언트가 아니라
   사후 안정 신호(seal/가압 상태)에 바인딩; 현실적 접촉-순간 패턴은 0%. 원인 = 접촉 z가
   5.6 cm 밴드에 밀집해 depth가 정지를 완벽 예측. **exposure(oversampling os10/os3) 기각**,
   access(dfmag 채널 추가) 단독도 무효, 조합만 부분 시너지.
4. **처방 + 실기 검증 (0729 probe 반영해 수정)**: 반응형 경로는 이중으로 막혀 있음 —
   (i) 접촉-순간 반응 권한은 어떤 처방에도 ~25% 이하 plateau, (ii) 물리 산술(스택 강성
   ~5 N/mm + 청크 지연 330 ms)상 반응형 감속 게이팅만으로는 힘 한계 내 정지가 불가능.
   → 처방은 **anticipatory**여야 한다: **press-retreat 데모**(힘 목표 8–15 N 랜덤, 후퇴
   5–10 mm, force-rise→stop→+dz를 seal 이전에 시연)가 저속 접근·후퇴 행동과 사후 seal-state
   게이팅(0729 val probe std **76%**)을 가르쳐, 로봇에서 naive 0/3(전부 15–19 N overpress
   abort) → FiLM prefix **5/7** 성공(접촉력 2.5–4.8 N, seal 후 +0.23–0.29 m lift). 이때
   접촉-순간 반응 권한은 여전히 7%로 낮음 — **probe 택소노미가 "성공은 반응 게이팅이 아니라
   행동 사전(prior)+사후 게이팅에서 온다"를 정확히 설명/예측**. 온로봇 live counterfactual
   probe 부호 일치(contact +0.73, sealed +2.09, fz+6N +2.18 mm/f).

## 4. Abstract sketch (EN, ~180 words)

> Vision-language-action (VLA) policies trained by behavior cloning increasingly receive
> physical condition signals — wrist wrench forces, contact, vacuum-seal state — in their
> observation, yet whether the policy *causally uses* them is unmeasured. On a real-robot
> suction case-picking task whose critical transition — stop descending on contact — is
> observable only through these signals (single head camera; layer height differences are
> visually near-indistinguishable), we show a fine-tuned VLA ignores them: it gates the
> transition on depth correlates, under-reaching as the stack depletes and over-pressing
> otherwise. We introduce **counterfactual authority probes**: a computed condition vector
> c-hat (contact, normal force, seal) injected through zero-initialized FiLM, with the raw
> wrench masked so c-hat is the only condition pathway — the conditioned policy receives no
> information the baseline lacks — which lets us measure how much forcing c-hat changes the
> action. The probes
> expose a **bypass phenomenon**: with the wrench unmasked, conditioning trains to identical
> loss yet zero authority — training loss is blind to causal use. Decomposed probes show
> authority binds to stable post-hoc signals (seal state), not the contact transient;
> oversampling transitions fails; only demonstrations that decorrelate contact from depth
> (varied contact heights, randomized press-retreat) raise contact-moment authority — and on
> the robot convert 0/3 over-press failures into gentle (<7 N) contact-gated picks.

## 5. Section outline (8쪽 = 본문 ~7쪽 + refs ~1쪽)

### I. Introduction (~1.0쪽, Fig. 1)
- **Goal**: 접촉 전이가 force-observable한 태스크에서 BC-VLA가 force를 쓰는가.
- **Problem**: force 융합 연구는 많지만(성능 트랙 포화) *사용 여부의 측정*이 없음; loss/성공률로는
  bypass가 불가시.
- **Solution(개요)**: 진단(oracle 대비) + authority probe(계측기) + insufficiency 3종 소거
  (access/exposure/decorrelation) + 실기 검증.
- Contributions 리스트 4개 = §3의 claim 1–4.
- (A6 GPS·A7 nugget 원칙 — intro 마지막에 nugget 한 문장 명시.)

### II. Related Work (~0.75쪽)
4 클러스터: (a) force/tactile-conditioned VLA — ForceVLA, **ForceFlow(최근접, 정면 차별화)**,
PhaForce, FARM, TacForeSight, FILIC, CGP; (b) intermediate representations — RT-Affordance,
FiLM; (c) causal confusion / shortcut — de Haan'19, Geirhos'20, Shortcut-in-Generalist-Policies
(2508.06426), 개입 기반 IL(2507.22380, 2307.15980); (d) privileged distillation(oracle 관련).
차별화 문장: "이들은 force 사용을 설계/가정하고 성능으로 검증; 우리는 사용 여부 자체를 개입으로
측정한다."

### III. Task and Diagnosis (~0.75쪽, Fig. 2 or Fig.1에 통합)
- Setup: suction case-pick, descend-until-contact, 15-D state(pos+quat+suction+seal+wrench),
  SmolVLA fine-tune, 15 Hz, chunk 실행.
- **관측 비대칭 강조 (사용자 지시)**: head cam 단일(+depth 컬러화), 손목캠 없음. 층간 케이스
  높이 차이가 원거리 head view에서 거의 구분 불가한 반면, seal 성패는 몇 mm에서 갈림 →
  전이를 분별할 수 있는 관측은 사실상 force/seal뿐. (Fig.2에 layer별 head-cam 프레임을 나란히
  놓아 시각적으로 입증하는 안 — "these frames look identical; only force differs".)
- Under-reach 진단: depletion sweep 층별 seal% 붕괴 + 습관 깊이 고정.
- Gate-oracle: 동일 모델 강제 descend-until-contact → 심층 회복, 단 over-press(16–19 N)
  → graded/learned fix의 동기 + oracle upper bound.
- (주의: 이 진단은 개발 초기 셋업의 결과 — 서술 시 §10 세대 원칙 참조.)

### IV. Instrument: Computed Condition, Bottleneck, and Authority Probes (~1.25쪽, Fig. 3)
- c-hat 정의: clip((|F|−F0)/τ) 등 채널(contact, fz, seal, dfmag) — **계산되는 관측가능 조건**
  (학습 head 아님 — bypass 대상이 하나 더 늘지 않게).
- **채널 설계 의도 (저자 설계 철학)**: force 값 하나가 아니라 이 태스크의 **task-critical 조건
  집합** — 접촉 여부(contact), 수직 가압(fz), 흡착 성립(seal) — 을 조건 인터페이스로 노출.
  사람 조작자가 자연스럽게 참조하는 물리 단서("접촉했으면 멈춰라, 씰 잡혔으면 올려라")를
  정책이 같은 방식으로 참조하게 만드는 것이 목표 (PROGRESS.md 원 동기). 채널이 복수라서
  §V의 채널 분해(어떤 신호가 채택되고 어떤 신호가 bypass되는가)가 가능해짐.
- **Zero-new-information**: c-hat 전 채널이 naive도 받는 state(wrench, seal)의 결정론적
  함수 — 정보 추가가 아니라 **표현/라우팅**의 변경. mask1은 오히려 raw wrench를 제거하므로
  conditioned 모델의 정보량 ≤ naive. → "센서를 더 달아서 좋아진 것" 반론 원천 차단.
- Zero-init FiLM 주입(prefix = state-token / suffix = expert 입력) — identity에서 출발.
- **Force-mask bottleneck**: wrench 차원 마스킹 → c-hat이 유일한 force 경로 = authority 측정의
  validity 장치 (아키텍처 novelty로 팔지 않음 — C1/F1 원칙).
- **Counterfactual authority probe**: 이미지·state 고정, c-hat만 0↔1 강제, committed-descent
  구간 Δdz로 상쇄율(%) 측정. std(전채널) vs realistic(실측 접촉-순간 캘리브레이션) vs
  채널 분해(per-channel do-intervention).
- Probe 한계 명시(open-loop 단일 프레임) + §VI의 live/closed-loop 교차검증 예고.

### V. Offline Findings: Bypass, Shortcut, and What Actually Moves Authority (~1.5쪽, Fig. 4–5, Table I)
- **V-A Bypass / loss-blindness**: mask0 vs mask1 × prefix/suffix 매트릭스 — loss 동일,
  authority 0↔67%. prefix>suffix (67 vs 23%).
- **V-B 채널 분해 → depth shortcut**: all-1 82% vs 현실 접촉-순간 0%; seal-only 21–28%,
  contact-only 11–14%, dfmag/fz-drop 0%. 데이터 분석: 접촉 z 5.6 cm 밴드.
- **V-C Exposure 기각**: os10 (transition frames 10×) → 현실 패턴 여전히 0%/역부호.
- **V-D 0729 authority 매트릭스 (Table I, held-out val)**: mask0 7–8% vs mask1 54–76%
  (bypass 0729-pure 재현), prefix 76 > suffix 60, **val-best > last** (76/54, 60/42 —
  val-loss 선택이 authority도 함께 고름).
- **V-E 반응 권한의 한계**: 접촉-순간 realistic 권한은 어떤 처방으로도 낮음 (0729조차 7%;
  과거 ladder: unimodal 0% → bimodal 7–10% → 2× 24–25% plateau, os 이득 소멸·스텝 무효).
  ⚠ ladder는 과거 라운드 수치 — 0729-only 정책 하에서 본문 포함 방식(개발 히스토리 vs 제외)
  사용자 결정 필요. → 물리 산술과 함께 "반응 경로는 닫혀 있다, anticipation이 필요하다"로
  §VI 연결.

### VI. Robot Experiments (~1.0쪽, Fig. 6–7, Table II)
- **V1 control (S2)**: V2 > V1 ≈ V0 [TO RUN — GPU]. "grounding이 활성 성분" 입증.
- **0729 press-retreat 라운드**: 데이터 처방(랜덤 힘목표 압입→후퇴→hover→seal 시 lift)
  → naive 0/3 overpress vs FiLM prefix **5/7** 성공(무효 3런 정리 후; refs/main 기준 5/6,
  val-best 1런 실패 — 리비전 pin 필요), 성공런 접촉력 2.5–4.8 N (<15 N 한계 대비 여유),
  suffix 1/3 (prefix>suffix 실기 재확인). [n 확대 TO COLLECT]
- **중간층(2–4) 보간 — 논문의 클라이맥스 실험** [사용자 계획 확정, TO COLLECT]:
  학습은 층1·5(양극단)만, 평가는 층2–4 포함 전층. **depth cue는 OOD·force cue는 in-dist가
  되는 유일한 세팅** — 두 cue를 로봇에서 행동적으로 분리(dissociate)하는 실험.
  설계 체크리스트: ① 층·모델당 n≥10, ② 체크포인트 리비전 사전 pin(스냅샷 경로 고정),
  ③ log-dir 규약 `case_pick_<layer>` (`lift_condition_probe.py --by-height` 재사용),
  ④ force-limit 15·nas 5 전 조건 고정, ⑤ 지표: seal% + peak contact N + time-to-seal +
  under-reach gap + overpress율, ⑥ 층 인덱스와 함께 실제 접촉 z(m) 보고 + 학습 접촉-z
  분포(바이모달) 대비 도식. 사전 등록식 예측: naive는 층 의존 실패(습관 깊이와 표면의
  상대 위치에 따라 over-press/under-reach), FiLM은 층 불변 성공 + graded 접촉력.
- **Depletion sweep** = 위 실험의 전층 버전 (naive vs FiLM vs 여유 시 gate-oracle), seal%+CI.
- **Live authority probe (S3)**: frozen-pose 온로봇 반사실 — contact +0.73 / sealed +2.09 /
  fz+6N +2.18 mm/f (전부 올바른 부호), no_contact 0. probe↔실기 거동 상관 제시.
- (여유 시) 타 아키텍처 1종(ACT or pi0.5)에서 bypass 재현 → 일반성.

### VII. Discussion & Limitations (~0.5쪽)
- **Fidelity trap (스파이스)**: 구모델의 early-lift "결함"이 우발적 overpress 보호막이었고,
  BC 충실도가 오르자(2× 데이터) dwell 모사가 정확해져 light-touch seal에서 치명적 —
  "faithful BC ≠ competent control".
- **물리 산술**: 스택 강성 ~5 N/mm → 접촉 후 감속 게이팅만으로 15 N 이내 정지는 산술적으로
  불가 → 처방이 anticipatory(저속 접근·press-retreat 데모)여야 하는 이유 — §VI 결과의 기제 설명.
- Limitations (F1 전략 배치): 단일 태스크/로봇 — "controlled diagnostic setting"으로
  Experiments 도입부에서 선제 프레이밍; plateau ~25% — gate-oracle 증류(DAgger)를 future work로.
- (선택) Kamin blocking 유비 — 한 문단, speculation 명시.

### VIII. Conclusion (~0.25쪽)

## 6. Figure plan

| # | 내용 | 상태 |
|---|---|---|
| Fig.1 | 티저 3패널: (A) 스택 소진+습관깊이 고정 만화 + seal% 붕괴 + oracle 회복 / (B) "Same loss, opposite causality" — 동일 loss 곡선 2개 + authority 막대 0% vs 67% / (C) shortcut 원인(접촉z 밴드 산점도) + decorrelation ladder | **제작 필요** (TikZ/pgfplots + 로봇 사진) |
| Fig.2 | 태스크/셋업 + phase 다이어그램 + **layer별 head-cam 프레임 나란히 배치**("look identical; only force differs" — 관측 비대칭 입증) | 사진 필요 |
| Fig.3 | 아키텍처: c-hat 계산 → FiLM(prefix) + force-mask bottleneck 블록도 | 제작 필요 |
| Fig.4 | loss vs authority 산점도 (모든 런) — loss-blindness 정량화 | **0-compute, 기존 수치로 즉시 가능** |
| Fig.5 | 채널 분해 + decorrelation ladder 막대 (probes/*.txt 수치) | 0-compute |
| Fig.6 | 로봇 결과: naive vs FiLM force-trace 페어(14:20 vs 14:32 diagnostics) + sweep 막대 | 일부 확보, sweep TO COLLECT |
| Fig.7 | live authority probe (20260730-151402, 0729 모델) | **확보됨** |

## 7. Table plan

- **Table I**: inject × mask 매트릭스 — loss / std authority / realistic authority (0708+0727 통합)
- **Table II**: 로봇 평가 — 모델 × (성공률, overpress율, 접촉력, time-to-seal, n) [+ 층별 sweep]
- (선택) Table III: decorrelation ladder 수치 (라운드 × 데이터 특성 × authority)

## 8. Claims → Evidence 상태맵

| Claim | Evidence | 상태 |
|---|---|---|
| 진단: condition-ignoring | depletion sweep + gate-oracle (70/60/0/20/0 → 회복) | ✅ (구 셋업, ~10회/층) — 0729-only 정책에 포함 여부 확인 필요 |
| Loss-blindness bypass | 0729 held-out probe: mask0 7–8% vs mask1 54–76% | ✅ **authority 축 0729-pure 완결**; loss 축 서버 수치 대기 |
| Depth shortcut | 채널분해 + 접촉z 밴드 분석 | ✅ 강함 (0708 데이터 — 서술 위치 결정 필요) |
| Exposure 기각 | os10/os3 negative | ✅ 강함 (과거 라운드) |
| 반응 권한 plateau | 전 라운드 realistic ≤25%, 0729 7% | ✅ (offline) |
| **Grounding이 활성성분 (V1 control, S2)** | V2>V1≈V0 | ❌ **TO RUN (GPU ~0.5일, 최우선)** |
| 실기 검증: naive vs FiLM | 0729 롤아웃 (0/3 vs 5/7) | ⚠️ 소N — **sweep n≥10 TO COLLECT** |
| 층 일반화 (보간) | 중간층 2–4 테스트 (depth OOD × force in-dist 분리) | 🔜 사용자 계획 확정 (로봇) |
| On-robot 반사실 (S3) | live probe 07-30 15:14 | ✅ 확보 |
| 아키텍처 일반성 | ACT/pi0.5 bypass 재현 | ❌ 선택 (가치 높음) |
| Negative control (S5) | 삽입 태스크 | ❌ 드랍 후보 (future work) |

## 9. 마감(9/15)까지 필수 실험 — 우선순위

1. **[GPU, ~0.5일] S2 V1 decorrelated control (0729 데이터로)** — 없으면 "그냥 FiLM capacity"
   공격에 무방비. 최우선.
2. [정리, sweep 전 필수] **체크포인트 리비전 pin** — 남은 롤아웃 중 val-best(7119b99) 1런만
   실패. sweep은 스냅샷 경로 고정으로 리비전 명시 후 진행 (probe상 best가 authority 우위
   76 vs 54 — best 사용 권장하되 실기 확인).
3. **[로봇, 반나절~1일] 전층 depletion sweep n≥10/층 (중간층 2–4 보간 포함)** — 사용자 계획
   확정. naive vs FiLM(0729 prefix) (+여유 시 gate-oracle). 설계 체크리스트는 §VI 참조.
4. **[0-compute] Fig.4/5 데이터 정리** — probes/*.txt → 산점도·막대. 서버에서 0729 4모델
   train/val **loss 수치만** 회수 (probe는 회수 완료).
5. **[GPU+로봇, 여유 시] ACT or pi0.5 bypass 재현 1종** — 일반성 방어 급상승.

## 10. Writing conventions

- 영어, IEEEtran (Overleaf 권장 — 로컬 LaTeX 미설치), double-anonymous(저자·기관·HF repo명
  익명화 — `Chanho-Lee/*` 노출 금지, 익명 링크 사용).
- **데이터 정책 (사용자 결정 2026-07-30): 논문 실험은 무조건 0729 데이터로만.**
  Table I(authority 매트릭스)·로봇 결과·V1 control 모두 0729. 과거 라운드 증거(0708 채널분해·
  os10 기각·decorrelation ladder)의 본문 포함 방식은 미정(§11 Q2) — 포함 시 별도 셋업임을
  **정직하게 명시**하고 "diagnostic development phase"로 구획, 숨기지 않고 혼동도 없게.
- 수치 인용은 `EVIDENCE.md` 경유 (probe 원문 경로 포함).
- 인용 계획(~25편)은 research-analyst 보고서 §4 — ForceFlow(2605.11048)·CGP(2603.05687)·
  2507.22380 신규 추가 필수.

## 11. Open questions (사용자 확인 필요)

확정된 것 (2026-07-30): 데이터 0729-only · 중간층 2–4 보간 평가 진행 · vision-info 미미함 강조.

1. 프레이밍 승인: 진단 중심 "Same Loss, No Cause" (추천) vs method 중심?
2. **과거 라운드 증거의 사용 범위**: (a) 구 셋업 진단 sweep(70/60/0/20/0)과 0708 채널분해·
   os10 기각·ladder를 "개발 단계 진단"으로 본문 요약 포함 (추천 — 논문 논리 사슬에 필요)
   vs (b) 완전 제외하고 0729 내 실험으로 재구성 (채널분해 probe를 0729 ckpt로 재실행,
   진단은 이번 sweep의 naive 층별 실패로 대체).
3. sweep 체크포인트: val-best(probe 권한 우위 76%) vs refs/main(실기 5/6) — 첫 층에서 양쪽
   2~3런씩 확인 후 pin?
4. 원격 GPU 서버(0729 loss 수치, V1 학습) 접근 시점?
5. ICRA 직행 vs RA-L(+ICRA option)?
