# RESULTS_LINEUP — 논문 수록 실험 결과 확정본 (08-11)

전면 재작성(prescription-led)용 결과 인벤토리. 수치 출처는 EVIDENCE.md 해당 절.
**로컬 데이터 > 구 EVIDENCE 집계** (08-11 사용자 확정, EVIDENCE §3 로봇 롤아웃 08-11 블록 참조).

## 보고 정책 (08-11 사용자 확정)
- **best-only**: 모든 표·수치는 val-loss 선택(best) 체크포인트만. last 행 전부 삭제.
  균일 규칙임을 §V 서두에 선언 ("one uniform selection rule").
- last가 하던 방어 2건은 §V remark 두 문장으로 대체: ① 어느 naive ckpt도 65% 상쇄를
  못 넘고, 늦은 ckpt는 오픈루프 반응↑·폐루프 악화 ② grounded 단조 형태는 학습 후반까지 유지.
- 로봇 9/9 배포 체크포인트 = **val-best** (사용자 확인 08-11) — 오프라인 best-only와 정합.
- intro의 "fades or saturates, never strengthening into a stop" 표현은 유지 (naive-last 인지
  하의 정직 표현).

## 모델 명명 (paper-wide)
- **naive policy** = 구 "baseline" (full wrench in state)
- **conditioned policy** = grounded pathway 정책 (computed ĉ + FiLM + wrench mask)
- **08-12: "bottleneck" 용어 전면 폐기** (사용자) — pathway/route로 통일, 형용사 쌍은
  grounded/ungrounded. 설계 고유명 없음 (조어 최소화 방침).
- ⚠ 논문에 등장하는 conditioned 모델은 **두 캘리브레이션 라운드**가 섞임 (아래 D 참조)

## A. §V Offline (통계 본진 — held-out val 6 eps)

### A1. ~~Bypass matrix (Table I)~~ — **08-12 삭제 (사용자, "압축")**
Table I과 §V-A 소절 제거. bypass는 §V ablation 내 명명 문단("Bypass: the mask, not the
module, reroutes force")으로 이동 — 메커니즘 한 문장(gradient가 raw 경로로 만족, 전용 통로
기아) + ablation의 wrench-kept 행이 1차 증거 + **원 캘리브레이션 쌍은 재현 한 절로 압축**
("7% vs 76% condition-forcing authority, near-identical loss"). §IV mask 동기는 bypass가
아니라 **중복 방지**("같은 신호 두 번 방지") 프레임 (08-12 사용자).
**⚠ 08-12 suffix 전면 제거 (사용자)**: suffix 행·"site matters" 관찰·§VII "prefix>suffix"
근거 모두 삭제. 유일한 흔적 = §IV-B 각주 반 줄 ("action-expert 주입은 예비 실험에서 낮은
authority"). π0.5 실험의 주입점 서술은 "action-expert injection"으로 표기.
(구 Table I 수치 보존: naive 0.15087/— · wrench-kept 0.15075/7%/3% · masked 0.14624/76%/7%)

### A2. Ablation 표 (Table II) — fromnaive(recal) 계열, 동일 naive-init (§3.5–3.6)
**08-12 재구성 (사용자): §V 결과 제시를 probe 순서 P1→P4로** — 표 열 순서 = P1 forcing /
P2 transplant / P3 dose / P4 press-sim; 해석 문단 재배치 (bypass→P2, form dissociation→P3,
grounding-not-capacity→P4); "Where authority binds" 소절 폐지, 내용은 P1(접촉-순간 7%)과
P2(sealed 147% vs 67%, 통계적 성격) 문단으로 흡수. label sec:offline-posthoc은 P2 문단 유지.
| 모델 (force 접근) | P1 forcing (committed Δdz) | dose 8→12N | transplant pc_fc | press-sim (seal-never) | val err |
|---|---|---|---|---|---|
| naive best@10k | — (ĉ 없음) | +1.41→+1.12 (36→28%, 감소) | +1.63 (105%) | 4–5/6, max 14.9mm | 0.84 |
| naive last@50k | — | +1.73→+2.00 (57→65%, 완만 상승) | +1.69 (108%) | 3/6, max 16.0mm | 0.71 |
| mask0-fromnaive (raw+죽은 ĉ) | 0.0mm (Tier1 FAIL) | +1.57→+1.12 (dRaw 전부) | +1.68 (dFiLM 0.04) | 4–5/6, max 11.9mm | 0.85 |
| V1 (병목, 셔플 ĉ) | −0.2mm (wrong sign) | −0.06→−0.11 (0) | −0.06 (0%) | **0/6, max 431mm** | 0.96 |
| fromnaive v2 best (접지 병목) | **+0.9mm** (12% of committed) | +1.20→**+4.40 (115%)** | +1.57 (94%, 전부 dFiLM) | **6/6, max 4.5mm** | 0.87 |
| fromnaive v2 last@20k | (미측정) | +0.97→+3.44 | +1.45 (97%) | 6/6, max 4.1mm | 0.81 |
- P1 출처: `0729_{fromnaive_best,mask0fn_main,v1_main}_std.txt` COMMITTED-desc Δdz. 논문 표는
  **mm만 표기** (% 금지 — P1 분모 −7.8mm/frame ≠ P2 분모 −1.7mm/frame, % 병렬은 오독 유발).
  probe_film_authority.py 헤더 "(counterfactual on training data)"는 stale 하드코딩 라벨 —
  실제 에피소드 = state probe와 동일한 **val ep0** (frames 0..218, first-contact ~141) 확인 08-12.
- loss: v2 0.14762 vs mask0fn **0.14750 (4째 자리 동일, bypass 재현)**; V1 0.17081(+16%, 정직 표기).
- naive 형태는 ckpt 의존 (best 감소 / last 완만 상승) → 견고 판별축 3개: 기울기 크기(~10×) /
  100% 상쇄선 통과 여부 / 폐루프 결과 (naive-last 오픈루프↑인데 폐루프 악화 3/6·16mm).
- seal-granted 대조: naive seal3에서 0.5mm 정상 정지 → naive 정지는 seal 이벤트 바인딩.
- 접촉-순간(반응형) 권한은 전 구성 7–10% — 성공 기제는 anticipatory 행동 + 사후 게이팅 (물리 산술 ~5N/mm과 정합).

### A3. π0.5 (3.6B) — 스케일/아키텍처 일반성 (텍스트만)
**⚠ 08-11 HOLD: pi0 offline 실험 재수행 중 (사용자) — 결과 나올 때까지 아래 수치로 §V-D
확정하지 말 것. 신규 결과로 대체 예정.**
- pi05_naive: transplant 65–97% (템플릿 존재), ramp ≤37% (얕은 포화), press-sim 사실상 0/6 max 86mm (SmolVLA naive보다 악화), val err 0.69–0.89.
- pi05_film (suffix, 구캘리브레이션): 전 도스 권한 0 (contact 포화에도), press-sim 0/6 max 94–102mm — V1 시그니처 재현 → "접지는 공짜가 아니다" (Discussion).

## B. §VI Robot

### B1. Closed-loop picks — **원 캘리브레이션 prefix_mask1** (07-30, 로컬 canonical)
| 모델 | 결과 | peak force (interaction window, 08-11 확정) |
|---|---|---|
| naive (L5, 3런) | 0/3 전부 외부 abort (자가 정지 0회) | 15.41 / 17.8 / 18.7 (censored 하한) |
| conditioned (L1/L3/L5 각 3런) | **9/9 자가 정지 + seal** | **0.88–14.54, median 2.50 (7/9 ≤ 3.1)**; 상위 10.16·14.54는 진짜 압축 press (fz −11) |

- **train set 높이 = L1, L5만** (08-11 사용자 확인). **L3는 학습에 없던 robustness 평가
  높이** → §VI "3/3 at an intermediate height absent from the demonstrations" 절의 근거.
  §III demo 문단의 "two different stack heights"와 정합 (val contact z 이봉: 0.762/0.817).

- **metric**: lift 전(interaction window) |F|−baseline max — meta의 전 구간 max는 carry 하중
  혼입으로 폐기 (EVIDENCE 08-11 metric 블록). 본문 표현은 "median 2.5 N, 7/9 ≤ 3 N" 권장
  (low end 0.9 N은 드리프트 ~1 N 규모라 range 강조 금지).
- 두 런은 측정 가능한 압축 없이 seal (min fz > 0) — gentleness 서술 재료.
- FiLM 활동 (성공런): |γ| +48%, |β| +79% at contact/seal vs descent.
- naive는 L5 단일 높이만 평가 (validity note).
- **NEW Fig**: per-trial force 궤적/strip (states.jsonl 15Hz, 전 런 로컬 보유). 14.54 런 숨기지 않음 — 한계 직전 자가 정지 = brake 증거.
- Table III(성공 카운트 표)는 제거, 카운트는 본문 종속절.

**pm1 자체 probe 세트 (08-11 로컬 확인 — 9/9 모델의 독립 메커니즘 증거):**
- dose ramp (`0729_state_pm1_ramp8/10/12`): committed descent 상쇄 **+3%→+7%→+13% 단조 상승**,
  전부 dFiLM (dRaw=+0.00 — mask 검증); near-contact(<2s) 구간은 +1.40→+1.61→+1.98mm
  (자기 하강 −2.14 대비 **65→93%**), pre-contact r12는 +2.03 (**95%**, ep5는 +dz 후퇴).
- transplant (`pc_fc`): +1.01mm / 자기 하강 −2.13 (47%, 전부 FiLM 경로).
- press-sim (`sim_pm1_seal0/seal3`): seal-never **3/6 정지, max 9.6mm** (높이 의존: z~0.817
  3/3, z~0.763 0/3 — fromnaive 6/6보다 약함, naive max 14.9–16mm보단 우위);
  seal3 제공 시 6/6, 3.1mm.
- live probe 07-30 (`20260730-151402`, 10포즈): contact +0.73 / sealed +2.09 / +6N +2.18 mm/f.
- ⚠ 정규화 주의: pm1 ramp의 committed-descent 프레임(dz −8.42, 2 eps)과 fromnaive
  ramp(−3.84, n=245)는 분모가 달라 **% 수치를 한 표에서 직접 비교 금지** — 형태(단조 상승)만
  공유 사실로 서술.
- ⚠ pm1은 seal-never 반사실에서 부분 정지(3/6) — 로봇 9/9는 seal 가용 조건. "pm1이
  seal-never에서도 만능"으로 쓰지 말 것.

### B2. Live counterfactual probes — **fromnaive vs naive** (08-06 ×2, 로봇 반출 직전)
- 공정 도스(≤9N, 10포즈): naive ≥ film (fc 67% vs 32%, sealed 147% vs 58%) — 오프라인 같은 도스 구간과 정합. film은 단조: hover +0.29 → preseal +0.87 → sealed +1.41 → +6N +1.70.
- 고도스(3포즈, +6/9/12N): **crossover 온로봇 재현** — film +1.19→+4.59→+8.56 (12N=자기 하강 2배=후퇴) vs naive +1.34→+3.18→+4.18 (한계반응 붕괴). 역전 6–9N. 표면 근접에서 naive +1.0~1.8 vs film +7.5.
- validity: hover 자체 재앵커, swap_drift 보정 (양모델 동일 물리 도스), film 절대치는 보수적 하한.
- (07-30 구 live probe, 원 캘리브레이션 prefix: contact +0.73 / sealed +2.09 / +6N +2.18 — 보조)

## C. Discussion 재료
- 물리 산술: 강성 ~5N/mm → 15N 이내 정지에 접촉 후 ~2mm 필요 → 반응형 게이팅 산술적 불가 → anticipatory 처방 필연.
- 운영 비대칭: F/T 드리프트 실측 0.4→1.1N/일; conditioned는 hover 재앵커 노브 (--film-auto-baseline), naive는 frozen stats.
- pi05_film 이식 실패 = "병목+데이터만으론 부족, 주입점·캘리브레이션 아키텍처별 재설계".

## D. 모델 정체성 (08-11 최종 — 단일 모델 표현)
사실관계: §VI-B1 (9/9 picks) = 원 캘리브레이션 prefix_mask1; §V-A2 (ablation) + §VI-B2 (live)
= 재캘리브레이션 fromnaive 계열. pm1도 자체 ramp/transplant/sim/live probe 보유 (B1).

**논문 표현 방침 (08-11 사용자 결정)**: 캘리브레이션 라운드 구분을 본문·캡션에서 제거하고
전부 **"the conditioned policy"**로 통칭 — 같은 설계의 인스턴스들이므로. "캘리브레이션하면
grounding이 강해진다" 진행 서사도 삭제 (그 포인트는 §IV 각주 + Discussion의 π0.5 이식
실패가 감당).

가드레일:
1. 인스턴스별 수치를 한 프로필로 합성 금지 — 각 수치는 자기 표/그림/실험 안에서만.
   (예: "76% 정책이 115% dose-response" 같은 문장 금지)
2. 인스턴스 공시는 §IV 각주 한 줄이 전담: "bypass matrix와 ablation의 conditioned 정책은
   같은 설계를 별도 학습·캘리브레이션한 인스턴스" — §IV 수정 시 기존 각주를 이 역할로 정비.
3. 라운드 간 % 직접 비교 금지 (분모 상이 — B1 주의 참조).

## E. 논문 제외 확정 (08-11)
- suffix 로봇 런 (1/3, 6.9/16.6/19.5) — 로컬 부재, 검증 불가. suffix는 offline만.
- 14:34 val-best 런 (28.1N) — 로컬 부재.
- 구 집계 "5/7, 2.5–4.8N" — 08-11 재검증으로 대체.
- 08-04/05 fromnaive 폐루프 시도들 (1/6, 0/9) — 오프셋 과보정/드리프트 이슈 세션 (EVIDENCE §3.5, §3.7 08-04 런 무효 판정). 논문은 fromnaive의 폐루프 성공을 주장하지 않음 (fromnaive는 offline+live만).
- contact-z decomp fig (분석 미수행), loss-authority scatter fig (Table I과 중복).
