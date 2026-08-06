# ICRA 2027 Paper Plan — "Same Loss, No Cause" (working title)

작성 2026-07-30. 오케스트레이터 종합 (증거 인벤토리 → `EVIDENCE.md`).

> **⚠ 증거 동결 (2026-08-06, 로봇 이동으로 환경 변경 — 사용자 확정)**: 실기 추가 실험
> 불가. 클라이맥스를 로봇 sweep(취소)에서 **기제 삼중 대조(naive/V1/v2 × 힘-스케일 스윕
> × 폐루프 sim, EVIDENCE §3.5–3.7)** 로 교체 — 완전 확보된 증거로 주장을 닫는 재구성.
> 로봇 결과는 소N 케이스 스터디 + 온로봇 반사실(고도스 crossover 포함)로 스코프.
> 수치 인용은 EVIDENCE.md 경유. 잔여 작업은 §9 (전부 책상 작업).

**Target venue**: ICRA 2027 — 마감 **2026-09-15** (11:59 PST, PaperPlaza), **8쪽 (참고문헌 포함)**,
**double-anonymous**. Backup: RA-L (+ICRA 2027 presentation option, 이관 마감 2026-12-31) —
분석형 논문에 유리, 페이지 유연.

---

## 1. Nugget (한 문장)

> Behavior-cloned VLA policies silently **bypass** the causal contact signal they are given;
> training loss cannot detect this, and neither architectural access nor exposure fixes it —
> only demonstrations that **decorrelate** the contact condition from its habitual correlates
> restore causal authority, offline and on the robot.
>
> **(08-04~06 확장, 사용자 확정)**: 그리고 아키텍처는 사용을 강제하지 못하지만 **사용의
> 형태를 결정한다** — 동일 데이터·초기화·loss에서 raw 접근은 힘이 시연을 초과하면 소멸하는
> 템플릿 반응(naive), 비접지 병목은 무반응(V1), 접지 병목만 힘에 비례해 커지는 브레이크
> (v2, 12N에서 정지+후퇴; 폐루프 sim 6/6 vs 0/6; 온로봇 crossover 재현)를 낳는다.
> = "Imitation pins down the trajectory, not the mechanism."

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
- **V-F 기제 삼중 대조 (신규 클라이맥스, EVIDENCE §3.5–3.6)**: 동일 데이터·동일 naive-init·
  동일 용량·동급 val err(0.81–0.96mm)의 세 정책을 힘-스케일 스윕(8/10/12N)과 폐루프
  press-sim(seal-never)으로 개입 —
  naive(raw): +1.41→+1.12 템플릿 감소, 4–5/6·max 14.9mm /
  **V1(비접지 병목): −0.06→−0.11 무반응, 0/6·max 431mm** /
  **v2(접지 병목): +1.20→+4.40 단조, 6/6·max 8.5mm**.
  → usage는 데이터가(press-retreat), form은 설계가(접지 병목) 결정. 용량 반론 차단(V1),
  bypass 반론 차단(mask0), "naive는 force-blind" 서사 금지(state-swap 105%, 토론 5·6).

### VI. Robot Experiments (~1.0쪽, Fig. 6–7, Table II)
- **V1 control (S2)**: V2 > V1 ≈ V0 [TO RUN — GPU]. "grounding이 활성 성분" 입증.
- **0729 press-retreat 라운드**: 데이터 처방(랜덤 힘목표 압입→후퇴→hover→seal 시 lift)
  → naive 0/3 overpress vs FiLM prefix **5/7** 성공(무효 3런 정리 후; refs/main 기준 5/6,
  val-best 1런 실패 — 리비전 pin 필요), 성공런 접촉력 2.5–4.8 N (<15 N 한계 대비 여유),
  suffix 1/3 (prefix>suffix 실기 재확인). [n 확대 TO COLLECT]
- ~~중간층(2–4) 보간 sweep n≥10~~ **[취소 — 08-06 로봇 이동, 환경 동결]**: future work로
  이동. 대신 7월 L1/L3/L5 소N 런들을 **n 명시 케이스 스터디**로 Table II에 포함 (리비전
  주석 필수 — §9). 6월 구셋업 진단 sweep(~10/층, oracle 회복)이 층-일반화 motivation을
  담당 (§III).
- **Live authority probe (S3) — 3런 시리즈 (EVIDENCE §3.7)**: ① 0729 pm1 (07-30):
  contact +0.73 / sealed +2.09 / fz+6N +2.18 mm/f (부호 전부 정상). ② fromnaive 공정-도스
  (08-06, 10포즈): film c-hat 도스-반응 단조(hover +0.29→sealed +1.41); 중간 도스에선
  naive ≥ film — 오프라인 곡선 같은 구간과 정합(토론 6 방어 재료). ③ **고도스 (08-06,
  fz+6/9/12N): crossover 온로봇 재현** — film +1.19→+4.59→**+8.56**(가속·후퇴 진입) vs
  naive +1.34→+3.18→+4.18(한계반응 붕괴). V-F의 기울기 대비가 실기 관측 위에서 성립.
- (드랍) 타 아키텍처 bypass 재현 — 시간상 future work.

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
| 진단: condition-ignoring | depletion sweep + gate-oracle (70/60/0/20/0 → 회복) | ✅ (구 셋업, ~10회/층) — "diagnostic phase"로 구획 서술 |
| Loss-blindness bypass | mask0 val 0.15075 ≈ naive 0.15087(4째 자리 동일)·권한 7% vs 76%; v2/mask0fn 0.14762/0.14750·권한 94% vs 0 | ✅ **완결 08-06 — loss·authority 양축 0729-pure** (Fig.4 즉시 가능) |
| Depth shortcut | 채널분해 + 접촉z 밴드 분석 | ✅ 강함 (0708 데이터 — 서술 위치 결정 필요) |
| Exposure 기각 | os10/os3 negative | ✅ 강함 (과거 라운드) |
| 반응 권한 plateau | 전 라운드 realistic ≤25%, 0729 7% | ✅ (offline) |
| **Grounding이 활성성분 (V1)** | 스윕 평평(−0.06→−0.11)·sim 0/6·err 동급 | ✅ **완료 08-05 (EVIDENCE §3.6)** |
| **형태 결정 (triad + 스윕)** | naive 템플릿 / V1 무반응 / v2 단조 + press-sim | ✅ **완료 08-04~05 (§3.5–3.6) — 신규 클라이맥스** |
| 실기 검증: naive vs FiLM | 0729 롤아웃 (0/3 vs 5/7) + L1/L3/L5 소N | ⚠️ 소N 케이스 스터디로 스코프 (sweep 취소 — 환경 동결) |
| On-robot 반사실 (S3) | live probe 3런 (07-30, 08-06×2) — **고도스 crossover 포함** | ✅ **완결 08-06 (§3.7)** |
| 아키텍처 일반성 | pi0.5 3.6B: naive 동일 클래스(포화 ≤37%·폐루프 0/6 max 86mm)·모방 동급 | ✅ **확보 08-06 (EVIDENCE §3.8)** — 스케일 반론 차단; film 이식은 실패(예비, 후속 3건) = Discussion 재료 |
| Negative control (S5) | 삽입 태스크 | ❌ future work |

## 9. 마감(9/15)까지 잔여 작업 — **전부 책상 작업 (실험 동결 08-06)**

1. ~~V1 control~~ **완료 (08-05)**. ~~로봇 sweep~~ **취소 (환경 동결)** — future work 문단으로.
2. **[서버 접속만] 0729 loss 수치 회수** — Fig.4(loss vs authority 산점도)의 loss 축.
   + V1 main 브랜치 = best/last 확인, V1/live probe 결과물 커밋.
3. **[로그 분석] 7월 롤아웃 리비전 pin 정리** — refs/main vs val-best 구분 주석, Table II의
   n·리비전 정직 표기.
4. **[0-compute] Fig.4/5/신규 스윕 Fig 데이터 정리** — probes/*.txt → 산점도·막대;
   신규: 힘-스케일 스윕 3곡선(naive/V1/v2, 오프라인) + 라이브 crossover 3점 오버레이.
5. **[집필] IEEEtran 스켈레톤 + Intro/Method 드래프팅 즉시 시작** — 수치 확정 상태이므로
   전 섹션 병행 가능. 실험일지 §6.10 기록(아카이브용)은 후순위.

## 10. Writing conventions

- 영어, IEEEtran (Overleaf 권장 — 로컬 LaTeX 미설치), double-anonymous(저자·기관·HF repo명
  익명화 — `Chanho-Lee/*` 노출 금지, 익명 링크 사용).
- **데이터 정책 (07-30 + 08-06 Q2 확정): 본문 증거 = 0729-only 순수.** 과거 라운드
  (0708 채널분해·os10·ladder·6월 구셋업 sweep) 수치는 본문에서 제외 — 진단·사후바인딩·
  바이모달 분석은 전부 0729 증거로 재근거(§11 참조). §III/V-B/V-C는 집필 시 이에 맞춰
  재구성 (V-B → 0729 state-swap 분해, V-C(exposure) → 본문 드랍).
- 수치 인용은 `EVIDENCE.md` 경유 (probe 원문 경로 포함).
- 인용 계획(~25편)은 research-analyst 보고서 §4 — ForceFlow(2605.11048)·CGP(2603.05687)·
  2507.22380 신규 추가 필수.

## 11. Open questions — **전부 해소 (08-06)**

확정: 데이터 0729-only(07-30) · 프레이밍 = 진단 "Same Loss, No Cause" + 형태 결정(08-04~05
토론으로 확정) · **Q2 = (b) 0729-only 순수 (08-06 사용자 확정)** — 본문은 0729 증거만:
진단은 0729 로봇 0/3 + press-sim + 포화 곡선으로, 사후-바인딩은 state-swap 분해
(st_sealed ≫ st_fc, 오프라인+라이브)로, 접촉 z 바이모달은 0729 parquet 재도출로 대체.
os10/ladder는 본문 제외(exposure 소거는 무수치 한 줄 or 삭제 — 집필 시).
sweep 관련(Q3)은 실험 동결로 무효. loss 회수(Q4) 완료. venue = ICRA 2027 직행(9/15),
RA-L 백업 유지.
