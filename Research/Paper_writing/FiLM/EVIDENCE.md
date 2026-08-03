# 증거 인벤토리 (정량) — ICRA 초안용

작성 2026-07-30. 논문 수치는 이 파일 경유로만 인용 (원문 경로 병기). `vla/` = `LGES/vla_training/`.

> **⚠ 전 수치 예비값** — 실험 캠페인 진행 중(사용자 지시 07-30). 소N 로봇 수치(0729 롤아웃 등)는
> 경향 파악용이며 논문 확정 수치가 아님. sweep·V1·중간층 결과로 일괄 갱신 예정.

## 1. 진단 (구 셋업, 2026-06-25, 정량 ~10회/층)

- Vanilla under-reach: 습관 깊이 ~0.82–0.83 m 정지, depletion sweep 층0–4 seal% **70/60/0/20/0**.
- Gate-oracle(동일 모델, 무재학습, 강제 descend-until-contact): **L2 0→80, L3 20→100, L4 0→90**.
  잔여 실패는 전부 over-press(|F| 16–19 N vs 데모 15.8) — 별개 모드 아님.
- 근거: `Research/condition_driven/DESCEND_UNTIL_CONTACT_DESIGN.md` §0, `REPRODUCE.md`.
- ⚠ 이 sweep에는 alignment 실패 없음 → S5 negative control은 별도 태스크 필요.

## 2. Offline probe 결과 (vla/probes/*.txt, 39개 전수)

수치 = committed-descent에서 c-hat 강제 시 하강 명령 상쇄율. std=전채널 강제,
realistic=실측 접촉-순간 캘리브레이션 강제.

### 0708 (구 로봇, 75 eps, 30k)
| 세팅 | loss | std | realistic |
|---|---|---|---|
| naive | 0.112 | — | — |
| film suffix mask0 | 0.101 | FAIL (max Δdz 0.8mm) | — |
| film prefix mask0 | 0.099 | FAIL | — |
| film suffix mask1 | 0.116 | 23% WEAK | — |
| **film prefix mask1** | 0.101 | **67% PASS** (dz −9.1→−3.0mm) | — |
| dF prefix mask1 | 0.106 | 82% | **0% (Δ−0.06mm)** |
| dF prefix mask1 **os10** | — | 69% | **0% (역부호 −1.6mm)** |

채널 분해(decomp_*.txt): contact-only 11–14%, seal-only 21–28%, dfmag-only 0%, fz-drop-only
0%(역부호), 현실 접촉-순간 조합 0%. → all-1 수치는 사후 seal/가압 상태 바인딩.
근본 원인: 접촉 z 10–90pct **0.787–0.843 m (5.6 cm 밴드)** → depth가 정지를 완벽 예측.
abs 계열 probe는 INCONCLUSIVE(committed-descent 프레임 부재).

### 0721 (새 로봇, 58 eps, 층1·5 바이모달, 30k) — std / realistic
3ch 38%/7% · 3ch+os3 35%/8% · dF 42%/8% · **dF+os3 59% PASS / 10%**
→ 2×2: oversampling·dfmag 단독 무효, **조합만 시너지 (+17~21%p)**. 현실-접촉 최초 비영(非零).

### 0721_0727 (116 eps, 50k)
| run | loss@50k | std | realistic |
|---|---|---|---|
| naive | 0.076 | — | — |
| **prefix mask1** | 0.074 | **61% PASS** | **24% WEAK** |
| prefix mask1 os3 | 0.068 | 60% | 21% |
| suffix mask1 | 0.062 | 51% | 16% |

30k→50k 연장: loss만 개선, authority 불변(62/25→61/24). os3 이득 소멸(데이터 2배로 자연 노출
충분). **plateau ~25%**.

## 3. 0729 라운드 (최신)

### 데이터 (commit f84fd79)
- **press-retreat 데모로 전면 전환** (`collect_case_pick.py`): suction ON 하강 → 에피소드별
  랜덤 목표힘 **touch_n ∈ [8, 15] N** 압입 → 접촉면 기준 **retreat ∈ [5, 10] mm** 후퇴 →
  hover 유지, seal 시 즉시 lift. 매 take가 seal 이전 force-rise→stop→+dz 반사실을 시연.
- train 100 eps / 19,445 frames, **val 6 eps / 1,169 frames (val split 최초)**. 15 fps.
- 배포 캘리브레이션: F0=6, τ=4, fz_τ=5, **FZ_OFF=2.1**. seal 프레임 비율 17.9%.
- 학습 4종(naive / prefix_mask1 / prefix_mask0 / suffix_mask1), 50k, val-loss 기반
  best ckpt 선택(`select_best_ckpt.py`). ⚠ **loss/val 수치·0729 probe txt는 원격 GPU 서버에만
  존재** (로컬 `logs` symlink 미마운트) — 회수 필요.

### 로봇 롤아웃 2026-07-30 (전 런 --force-limit 15, --n-action-steps 5)
`vla/rollouts/smolvla_{film,naive}_0729/`
**2026-07-30 사용자 정리 후 기준** (무효 3런 삭제: 중단 1, hover-stall 1, edge 오검출 압입 1).
재계산 스크립트로 검증 완료 (peak_contact = meta.json `peak_contact_n`, baseline 차감값).

| 모델 | 성공 | overpress abort | 기타 실패 | 성공런 접촉력 |
|---|---|---|---|---|
| naive | **0/3** | 3/3 (contact 15.4–18.7 N) | 0 | — |
| film prefix_mask1 | **5/7** | 1/7 (28.1 N, val-best 리비전 런) | light-press no-seal 1 | **2.5–4.8 N** |
| film suffix_mask1 | 1/3 | 2/3 (16.6 / 19.5 N) | 0 | 6.9 N |

- 성공 6런: seal 도달 14.1–40.8 s, seal 후 lift +0.23–0.29 m, overpress 0건.
- naive 실패 모드 균일: 접촉 후 계속 하강 → force-limit abort. **논문 핵심 대비.**
- FiLM 활동 증거(성공런): |γ| abs_mean 0.057→0.085(+48%), |β| 0.041→0.073(+79%) at contact/seal.
- **리비전 구분**: prefix 7런 중 6런 = repo명 기록(refs/main 5d29f4a 추정), 1런(14:34) =
  val-best(7119b99) 스냅샷 확정 — 이 런만 overpress 실패. 리비전별로 나누면
  **refs/main 5/6, val-best 0/1**. 논문 수치 확정 전 리비전 정리 필요.

### Offline authority probe (0729 체크포인트, **held-out VAL 6 eps**, 2026-07-30 추가)
`vla/probes/0729bl_*.txt` (best/last × std/real), `0729_*.txt`(=best 중복). contact-n 6,
realistic c1=[0.6, 0.42, 0].

| 모델 (ckpt) | std 상쇄율 | realistic 접촉-순간 |
|---|---|---|
| prefix_mask1 **best** | **76% PASS** | 7% WEAK |
| prefix_mask1 last | 54% PASS | 8% |
| prefix_mask0 best / last | **7% / 8% WEAK** | 3% / 4% |
| suffix_mask1 best / last | 60% PASS / 42% | 10% / 1% |

핵심 판독:
1. **Bypass가 0729 데이터만으로 완결 재현** — mask0 7~8% vs mask1 54~76% (동일 데이터·동일
   아키텍처·held-out 측정). Table I을 0729-pure로 구성 가능 (loss 수치만 서버 회수 필요).
2. **prefix > suffix 재확인** (best 76 vs 60).
3. **val-best가 last보다 authority 높음** (76 vs 54 / 60 vs 42) — val-loss 선택이 authority도
   함께 고름. 체크포인트 선택 문단의 보너스 발견.
4. **⚠ 서사 수정 필요**: realistic 접촉-순간 권한은 3~10%로 0727(24%)보다 오히려 낮은데 로봇
   성공률은 급등 — 폐루프 성공의 주 기제는 반응형 접촉 게이팅이 아니라 **press-retreat 데모가
   가르친 anticipatory 저속 접근·후퇴 행동 + 사후 seal/가압 게이팅(std 76%)**. 물리 산술
   (강성 ~5 N/mm → 반응형 감속 게이팅 불가)과 정합 — probe가 "왜 반응 경로가 아니라 데이터
   처방이어야 하는지"를 예측한 구조.

### Live authority probe (온로봇 반사실, S3)
`vla/live_film_probes/smolvla_film_0729_prefix_mask1/20260730-151402_*`
- 10개 frozen pose, c-hat 강제 시 mean Δdz: **contact +0.73 / sealed +2.09 / fz+6N +2.18 mm/f**
  (전부 상승=올바른 부호), no_contact 0. Δsuction: contact +0.126 / sealed +0.217.
- 구세대 live probe 세트: `live_film_probes/2026072{7,9}-*` (clearance −4~+5 cm, fz-delta ±3/±6).

### states.jsonl 스키마 (분석 재현용)
프레임당: `i, t, ee.pos[3], ee.quat_wxyz[4], wrench.{fx..tz}, suction_cmd, vacuum_sealed,
chunk_boundary, action_pred[7], action_cmd[7], film.{cond_names, c_hat[3], gamma/beta 통계}`.
raw |F|에 ~5.2 N 장착 오프셋 포함 — meta.json의 `baseline_force_n` / `peak_contact_n` 사용.
r02/r03 = `--loop` 세션 내 반복 번호.

## 4. 0727 실패 분석 (Discussion 재료 — fidelity trap)

2026-07-29 로봇 평가, `smolvla_film_0721_0727_prefix_mask1` 8런 중 7 실패 분해:
- fz_off 오설정 3런 (부호 오타 등 — 배포 노브 문제, 논문에선 제외 or 각주).
- **light-touch seal → dwell-abort 2런**: |F| 4.8–6에서 seal — 학습 데이터에 없는
  (seal=1, contact=0) 조합 → 모델이 데모의 19프레임 dwell을 충실 재현하는 동안 진공
  벨로우즈가 힘 21–26 N로 램프 → force-limit abort. **구모델(30k/1×)의 early-lift '결함'이
  이 구간을 건너뛰게 해주던 우발적 보호막** — BC 충실도 향상이 실패를 도입.
- 하강 overpress 2런: 접촉 권한 ~25% plateau + 청크 지연(330 ms) + 하강속도 −1.4 mm/f
  (구모델의 ~3배) → 6프레임에 30 N.
- **물리 산술**: 스택 강성 ~5 N/mm (접촉→30 N에 5.1 mm) → 15 N 이내 정지엔 접촉 후 ~2 mm 내
  완전정지 필요 = 감속 게이팅 산술적 불가 → anticipatory 접근 또는 후퇴만 가능.
  (0729 press-retreat 데모가 정확히 이 처방.)
- nas2(청크 2) 실험: contact=1·fz=−4·seal=1 포화에도 감속만(−1.3→−0.7 mm/f) — plateau 폐루프 확정.

## 5. Figure 자산

| 자산 | 경로 | 용도 |
|---|---|---|
| live probe 0729 (오늘) | `vla/live_film_probes/smolvla_film_0729_prefix_mask1/20260730-151402_*.png` | Fig.7 |
| rollout diagnostics (z + |F| 시계열) | `vla/rollouts/smolvla_film_0729/<run>/film_diagnostics.png` (12/13런) | Fig.6 (성공 14:32 vs overpress 14:34 페어) |
| state sensitivity | `Research/contact_aware_vla/results/state_sensitivity.png` | 보조 |
| seal→lift 인과 | `Research/contact_aware_vla/results/lift_causality.json` (case seal-gating 0.65) | 진단 보조 수치 |
| chunk staleness | `Research/reactive_chunking/p0_results/*.png` (반응지연 mean 1.09 s) | 청크 지연 논거 |
| probe 원문 | `vla/probes/*.txt` | Table I, Fig.4–5 |

## 6. 미확보 (회수/수집 필요)

**⚠ 데이터 정책 (2026-07-30 사용자 결정): 논문 실험은 무조건 0729 데이터로만.**
→ Table I(bypass 매트릭스)은 0729 4모델(naive/prefix_mask1/**prefix_mask0**/suffix_mask1)로
재구성 — mask0가 이미 학습돼 있어 loss-blindness 주장 성립 가능. 채널분해·realistic probe도
0729 체크포인트에서 재실행 필요.

1. 원격 GPU 서버: 0729 4모델 train/val **loss 수치** (probe는 2026-07-30 회수 완료 — §3 참조),
   prefix_mask0 체크포인트 로컬 확보. (+ 필요 시 0729 체크포인트 채널분해 decomp probe.)
2. 0729 롤아웃 리비전 확정 (val-best vs refs/main) — 남은 runs: refs/main 추정 6, val-best 1(실패).
3. 로봇: depletion sweep n≥10/층 (naive/FiLM/oracle), 중간층 2–4 보간.
4. V1 decorrelated control 학습+probe (S2).
5. 층 분포 메타데이터 (lerobot 변환에서 layer tag 미보존 — 0729 sweep 수집 시 log-dir 규약
   `case_pick_<layer>` 준수).
