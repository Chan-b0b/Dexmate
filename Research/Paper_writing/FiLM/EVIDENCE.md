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

#### ⚠ 08-11 재검증 — 위 표를 대체 (사용자 확정: 로컬 데이터 > 위 집계)
로컬 `vla/rollouts/` meta.json 전수 재검(08-11). 사용자 확인: "전부 실제 성공런이었다."

| 모델 | 성공 | peak (interaction window) |
|---|---|---|
| naive (L5만, 3런) | **0/3** 전부 force-limit abort | 15.41 / 17.8 / 18.7 |
| film prefix_mask1 (L1/L3/L5 각 3런) | **9/9** 전부 자가 정지 후 seal | **0.88–14.54, median 2.50** (7/9 ≤ 3.12) |

**⚠ 08-11 metric 확정 (사용자)**: 공식 peak = **interaction window** (하강~press~retreat,
lift 시작 전 = EE z가 최저점+20mm를 마지막으로 벗어나기 전) 내 |F|−baseline 최대.
- 이유: meta `peak_contact_n`(전 구간 max)은 gentle 런들에서 **seal 후 carry 하중**(fz>0
  인장)을 잡음 — L1 run1은 press 1.5N인데 meta 4.83N(z 0.830, 표면+7cm). abort가 감시하는
  양과 동일 층위라 15N 선과 비교 일관, naive는 lift 없어 불변.
- 층별 interaction peak: L1 = 2.13/10.16/3.12* → 창 재계산 [2.13, 10.16, 3.12],
  L3 = [1.36, 2.50, 2.85], L5 = [2.50, 0.88, 14.54]. (*스크립트 `scripts/make_force_traces.py`
  및 재계산 로그 08-11)
- 두 hard 런(10.16/14.54)은 min fz −11.3/−10.7 = 진짜 압축 press. 두 런(L1 run1, L5 r04)은
  min fz > 0 = **측정 가능한 압축 없이 seal**.
- 14.54 런의 |F| 피크 프레임은 fz +3.9·횡력 지배 (표면 높이, pre-seal) — "누른 힘" 성분
  주의. 압축만으로는 max −fz ≈ 10.7.

- 사용자 확인(08-11): 로컬 9런 전부 **val-best 체크포인트** 배포 — 논문 best-only 보고
  정책과 정합.
- 구 집계(5/7, 성공런 2.5–4.8 N)와 14:34 val-best 런(28.1 N)은 로컬에 부재 → **논문 미사용**.
- suffix 롤아웃도 로컬 부재 (1/3, 6.9/16.6/19.5는 검증 불가) → 로봇 suffix 수치 논문 제외 권장.
- naive는 L5 단일 높이에서만 평가 — §VI validity note 필요.
- 논문 서사 확정: "naive는 매 시도 외부 abort로만 종료(censored ≥15.4 N) vs conditioned는
  매 pick 자가 정지(median 4.3 N)" — 성공 카운트 표 대신 per-trial force figure로 제시.

### Offline authority probe (0729 체크포인트, **held-out VAL 6 eps**, 2026-07-30 추가)
`vla/probes/0729bl_*.txt` (best/last × std/real), `0729_*.txt`(=best 중복). contact-n 6,
realistic c1=[0.6, 0.42, 0].

| 모델 (ckpt) | std 상쇄율 | realistic 접촉-순간 |
|---|---|---|
| prefix_mask1 **best** | **76% PASS** | 7% WEAK |
| prefix_mask1 last | 54% PASS | 8% |
| prefix_mask0 best / last | **7% / 8% WEAK** | 3% / 4% |
| suffix_mask1 best / last | 60% PASS / 42% | 10% / 1% |

### train/val loss (`vla/0729_training_results.csv`, 서버 회수 08-06) — Table I loss 축 완결

| run | train loss | **val loss** | best step | val probe std 권한 |
|---|---|---|---|---|
| naive | 0.0520 | 0.15087 | 10k | — |
| prefix_mask0 | 0.0540 | **0.15075** | 15k | **7–8%** |
| prefix_mask1 | 0.0590 | **0.14624** | 5k | **54–76%** |
| suffix_mask1 | 0.0540 | 0.15924 | 5k | 42–60% |

판독: ① **mask0 val loss = naive와 소수 4째 자리 동일** (0.15075 vs 0.15087) — 권한 0의
bypass 모델이 loss로는 완전 무구분. ② mask1은 오히려 **최저 val loss**인데 권한 76% —
loss 3% 차이가 권한 10× 차이를 신호하지 못함. ③ **loss 순위와 권한 순위가 역상관**:
suffix_mask1이 val loss 최악(0.1592)인데 권한 60%, mask0은 loss 양호한데 권한 7%.
→ Fig.4 산점도(loss vs authority) 무상관의 정량 근거.

핵심 판독:
1. **Bypass가 0729 데이터만으로 완결 재현** — mask0 7~8% vs mask1 54~76% (동일 데이터·동일
   아키텍처·held-out 측정). Table I은 0729-pure로 **완성** (loss 축 회수 완료, 위 표).
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

### 3.5 0729 recal/fromnaive 라운드 (서버 2026-07-31~08-03, 회수 08-04 — 커밋 5f2ea6d/e58fa27)

재캘리브레이션(실측 0729 분포): contact=(|F|−5.5)/1, **fmag 채널 신설** (|F|−5.5)/1,
fz=(fz−3.0)/0.7 (`vla/run_case_pick_0729_recal.sh` 주석 = 단일 출처). 신규 학습 2종:
`…prefix_mask1_recal`(base init, 50k) / `…prefix_mask1_recal_fromnaive`(**naive best@10k
warm-start**, 20k). 배포 env: `FILM_COND=contact,fmag,fz,seal F0=5.5 TAU=1 FMAG_OFF=5.5
FMAG_TAU=1 FZ_OFF=3.0 FZ_TAU=0.7`. ⚠ HF 미업로드(스크립트 준비), 실험일지 §6.10 미기록.

**state-swap probe** (실측 first-contact state를 pre-contact 하강 프레임에 주입 — §3의
합성 c-hat "realistic"과 **측정법 다름, 혼용 금지**; `vla/probes/0729_state_*_pc_fc.txt`,
val 60프레임): naive 상쇄 **105%** (전부 raw 경로) / pm1r best 62%·last 90% /
fromnaive best **94%**·last **97%** (전부 dFiLM, dRaw=+0.00).
→ **naive도 힘 반응을 학습함 — "force-blind" 서사 사용 금지** (DISCUSSION_LOG 08-04).

**힘-스케일 스윕 — "형태 결정 장치" 주장의 1차 증거** (`--swap fcscale`, fc wrench를
8/10/12N로 스케일, 전 하강 프레임 n=245; `vla/probes/0729_state_{naive,fromnaive_best}_ramp*.txt`):

| 주입 \|F\| | naive 상쇄 (of −3.94mm/f) | fromnaive 상쇄 (of −3.84, 전부 FiLM) |
|---|---|---|
| 8N | +1.41mm (36%) | +1.20mm (31%) |
| 10N | +1.24mm (31%) | — |
| 12N | +1.12mm (28%) | **+4.40mm (115% = 완전정지+후퇴)** |

pre-contact만(n=60, `*_pc_r12.txt`): naive +1.35 vs fromnaive **+5.42mm**.
**응답-대-힘 기울기 부호가 정반대**: naive=템플릿 바인딩(힘↑반응↓), FiLM=단조 외삽
(힘↑브레이크↑; 12N이면 fmag c-hat=6.5, 학습범위 ~1.2의 5배 밖에서도 방향 보존 — 채널
단조성은 학습이 아니라 설계). 인과 고정: fromnaive=동일 초기화·데이터·병목만 추가.

**press-sim 폐루프** (`vla/probe_press_sim.py`; seal_depth=never = 오정렬/씰실패 시나리오,
force-model fzdelta; `vla/probes/0729_sim_*.txt`):
naive off1/off30 = 4/6 정지·mean 9.8mm(**max 14.9**) / 5/6·5.3mm(max 9.4);
fromnaive last = **6/6·7.0mm(max 8.5) / 6/6·1.7mm(max 4.1)**.
seal 3mm 제공 시(seal3) naive 0.5mm 정상, 훈련 템플릿 그대로인 pattern 모델에서도 naive
1.0mm 정상 → **naive 정지 = seal 이벤트/템플릿 의존, FiLM = 힘 자체 바인딩** (fzdelta+
seal-never에서만 갈라짐 — 조건 명시해 주장할 것).

**offline eval** (`vla/eval_offline.py`, val 액션 오차/step; `vla/probes/0729_eval_*.txt`):
naive 0.84 / pm1r 0.94(best)·0.81(last) / fromnaive 0.87(best)·**0.81mm(last)** —
병목의 정확도 비용 0. fromnaive는 **last(20k)가 전 오프라인 지표 우세** (swap 97%·
sim 1.7mm·err 0.81) — val-best 정책과 상충, 로봇에서 A/B 필요.

**운영 증거 (배포 노브)**: F/T baseline 드리프트 실측 — 7/30 하루 내 +0.4→+1.1N,
08-04 +1.03N (rollout meta `baseline_force_n`, `vla/measure_force_baseline.py`).
FiLM은 오프셋 재앵커로 런마다 보정(`run_policy.py --film-auto-baseline` 구현, 학습
ep-start 앵커 |F| 4.59N/fz 1.96N); naive는 frozen stats — 보정 노브 없음. sim의
pattern-정상/fzdelta-붕괴 대비가 템플릿 취약성의 간접 증거.
⚠ 08-04 live probe에서 오프셋의 **측정-포즈 민감성** 실증: 수동 측정(hover 5.63N)
기반 env 파일이 실제 probe 포즈(4.9N) 대비 ~0.7N 과보정 → c-hat 오앵커. 학습 데이터
확인 결과 ep-start(view park)와 하강 직전 정적 hover의 힘 통계는 0.1N 내 일치
(4.59/1.96 vs 4.62/2.04) → 자동 재앵커는 두 포즈 모두 유효; live probe에도 하강 전
hover 자체 재앵커 추가(08-05, `--baseline-hover`).

### 3.6 V1 decorrelated control — "형태 결정" 주장 완결 (2026-08-05 로컬 probe)

`Chanho-Lee/smolvla_film_0729_prefix_mask1_recal_fromnaive_v1` (naive-init 20k, recal
세팅 동일, **학습 시 c-hat 배치 셔플** = grounding만 제거; main = **val-best@5,000**,
val loss 0.1715 — HF 커밋 메시지로 확인 08-06). 로컬(Jetson, torch 2.11) 파이프라인 검증:
fromnaive pc_fc 재실행 = +1.57mm(94%) — 서버 수치 정확 재현.
체크포인트 정리(HF main, 전부 val-best): fromnaive=**@2,500**, V1=@5,000,
mask0_fromnaive=val-best — 5–15k 과적합 발견(§6.9)과 일관되게 전부 이른 지점.

**동일 데이터·동일 naive-init·동일 용량의 사중 대조** (`vla/probes/0729_*{v1,mask0fn}*`,
mask0fn·fn_last는 08-06 로컬 추가 — 전 행 HF main=val-best, fn last만 20k):

| 모델 (force 접근) | 8N→12N 응답 | pc_fc | press-sim fzdelta (seal-never) | val err |
|---|---|---|---|---|
| naive best@10k (raw 6-d) | +1.41→+1.12 (감소) | +1.63 (105%) | 4–5/6, max 14.9mm | 0.84mm |
| naive last@50k | +1.73→+2.00 (완만 상승, 57→65%) | +1.69 (108%) | **3/6, max 16.0mm** | 0.71mm |
| **mask0-fromnaive (raw+무력 c-hat)** | +1.57→+1.12 (**dRaw가 전부**, 템플릿) | +1.68 (dRaw +1.67 / dFiLM +0.04) | 4–5/6, max 11.9mm | 0.85mm |
| **V1 (병목, 비접지 c-hat)** | **−0.06→−0.11 (0)** | −0.06 (0%) | **0/6, mean 51–167mm, max 431mm** | 0.96mm |
| fromnaive v2 best (병목, 접지) | +1.20→+4.40 (단조) | +1.57 (94%, 전부 dFiLM) | 6/6, max 4.5mm | 0.87mm |
| fromnaive v2 last(20k) | +0.97→+3.44 (단조) | +1.45 (97%) | 6/6, max 4.1mm | 0.81mm |

판독:
1. **예측 적중 — 스윕 완전 평평 (전 힘 레벨 0, dRaw=0·dFiLM≈0)**: fromnaive의 단조
   브레이크는 용량·재학습·병목의 존재가 아니라 **c-hat의 grounding**에서 온다.
   용량 반론 사망 → §3.5 "형태 결정 장치" 주장 완결.
2. **비접지 병목 = 최악** (0/6, 431mm): mask1이 raw 경로를 지웠는데 c-hat이 무의미
   → 진짜 force-blind. 활성 성분은 병목 자체가 아니라 **접지된 병목**. std probe도
   NO/WRONG-SIGN (Δ−0.2mm).
3. **동일 모방 전제 성립**: val err 0.81–0.96mm — 전 행 동급. 다섯 정책은
   매니폴드 위에서 구분 불가, 개입에서만 갈라짐 (토론 5의 실측 완성).
4. ⚠ ramp n(739 vs 245) — descent 프레임 선정이 정책 예측 의존. 상쇄율 결론 무관.
5. **mask0-fromnaive (08-06): bypass 4번째 재현 + 분해 완결** — c-hat std probe
   Tier 1 FAIL(Δ−0.04mm), state-swap 분해 dRaw ≈ 전부·dFiLM ≈ 0, 그리고 **도스-반응
   곡선이 naive와 사실상 일치** (+1.57→+1.12 vs +1.41→+1.12; 12N 동값) — FiLM 모듈을
   "붙이기만" 하면(마스크 없이) 아무것도 변하지 않음을 정량으로. naive-init·recal·
   fmag 채널 포함 조건에서도 우회 재현 = 가장 강한 버전.
   **loss 완결 (08-06 회수, `vla/0729_training_results.csv`)**: v2 val 0.14762 vs
   mask0fn 0.14750 — **소수 4째 자리 동일**, 권한 94% vs 0. 0708의 "0.101 vs 0.099"가
   fromnaive 세팅에서 재재현. V1 val 0.17081(+16%) — 셔플 노이즈로 약간 높음(정직
   표기), 단 loss 차(+16%)와 권한 차(94%→0, 범주적)는 규모가 다름.
6. **형태의 체크포인트 안정성**: fromnaive last(20k)도 단조 유지 (+0.97→+3.44) —
   best@2.5k와 같은 형태, 크기만 소폭 감소. 단조 브레이크는 특정 체크포인트의
   우연이 아니라 학습 전 구간에서 유지되는 성질.
7. **⚠ naive의 형태는 체크포인트 의존 (08-06, 서술 주의)**: best@10k는 감소(36→28%),
   last@50k는 완만 상승(57→65%) — "naive=항상 감소 템플릿"으로 일반화 금지. 견고한
   판별축은 ① **기울기 크기** (v2 Δ+2.4~2.5mm vs naive Δ−0.3~+0.3mm, ~10×),
   ② **100% 상쇄선 통과 여부** (v2만 12N에서 하강 완전 상쇄+후퇴; naive는 어느 ckpt도
   65% 이하), ③ **폐루프 결과** (v2 6/6·max 4mm vs naive-last 3/6·max 16.0mm —
   오픈루프 반응이 커져도 폐루프는 오히려 악화 = "반응 크기 ≠ 안전"의 실측).

### 3.7 Live probe 시리즈 — 온로봇 반사실, fromnaive vs naive (08-04 / 08-06 ×2)

`vla/live_film_probes/smolvla_film_0729_prefix_mask1_recal_fromnaive/run{1..4}_*/`
(런별 서브폴더 + README.md로 재구성 08-06 — 유효성·모델·핵심 수치 명기; json +
`_vs_naive.png`). 동일 동결 관측에 두 모델 예측 — naive는 raw-state swap,
film은 c-hat 강제 + 같은 swap. 로봇 이동 직전 마지막 실기 데이터 (08-06).

- **08-04 런**: env 파일 오프셋이 probe 포즈 대비 ~0.7N 과보정 → film 쪽 무효,
  naive 쪽만 유효. 교훈 2개 = 오프셋 포즈 민감성 + 절대값 swap×재앵커 비대칭
  (§3.5 운영 증거 참조).
- **08-06 #1 (공정 도스, 10포즈, hover 자체 재앵커)**: 중간 도스(≤9N)에서 naive ≥ film
  — fc 67% vs 32%, sealed **147%** vs 58% (자기 하강 대비). **오프라인 곡선의 같은
  도스 구간(8N: naive +1.41 > film +1.20)과 정합** — 모순 아님 (토론 6).
  film c-hat 도스-반응 단조: hover +0.29 → preseal +0.87 → sealed +1.41 → fz+6N +1.70.
- **08-06 #2 (고도스 3포즈, fz +6/9/12N)**: **crossover 온로봇 재현** —
  film +1.19 → **+4.59** → **+8.56**mm (가속, 12N = 자기 하강 −4.1의 2배 = 후퇴),
  naive +1.34 → +3.18 → +4.18 (한계반응 붕괴 +1.84→+1.00). 역전 지점 6–9N,
  12N에서 film 2×. 표면 근접 포즈(1–3cm)에선 naive 12N +1.0~1.8 vs film +7.5.
- 설계 노트: ① 절대값 swap(fc/sealed)은 측정 드리프트만큼 평행이동해야 두 모델이
  같은 물리 반사실을 받음 (08-06 수정, `swap_drift` JSON 기록 — 미보정이면 film이
  드리프트만큼 저도스). ② suction=0 오프-매니폴드 등 라이브 전이 손실로 film
  절대치는 과소평가 가능(각주 후보). ③ 소N(3포즈) — 고도스는 표적 보충 측정으로 서술.

### 3.8 pi0.5 라운드 — 아키텍처 일반성 (서버 probe 08-06, `vla/probes/0729_*pi05*`)

**pi05_naive_0729** (lerobot/pi05_base 3.6B fine-tune, bs8 50k; best@10k/last@50k):

| ckpt | pc_fc | ramp8→12 (of own descent) | press-sim fzdelta seal-never | val err |
|---|---|---|---|---|
| best@10k | +1.46 (65%) | +0.69→+0.97 (17→25%) | **0/6, mean 39.6–66.2, max 85.6mm** | 0.89mm |
| last@50k | +1.86 (97%) | +0.79→+1.02 (29→37%) | 1/6·0/6, max 72.6mm | 0.69mm |

판독 — **SmolVLA naive와 동일 클래스, 스케일 무관**: ① 온매니폴드 템플릿 반응 존재
(65–97%), ② 얕은 포화 곡선 — 어떤 지점에서도 100% 상쇄선 근처도 못 감(≤37%),
③ 폐루프 힘-램프 붕괴 — SmolVLA naive(3–5/6, max 16mm)보다 오히려 심함(사실상 0/6,
max 86mm), ④ 모방 정확도 동급(0.69–0.89mm). → bypass/shortcut/형태 문제는
0.45B→3.6B 스케일업과 아키텍처 교체(SmolVLA→pi0.5)에 불변 — **"더 큰 모델이 해결"
반론 차단**. §8 아키텍처 일반성 행 복원.

**pi05_film — 접지 이식 실패 확정 (08-06 서버, film_contact_pi05 라우팅 probe)**:
`pi05_film_frombase_0729` / `pi05_film_onnaive_0729` (suffix 전용, 구캘리브레이션
cond=contact,fz,seal mask1 FZ_OFF=2.1):

| | pc_fc | ramp8→12 | press-sim (seal-never) | val err |
|---|---|---|---|---|
| filmfb best | +0.05 | +0.04→+0.06 | **0/6, mean 71–80, max 94.6mm** | 0.89mm |
| filmon best | −0.02 | −0.02→−0.03 | **0/6, mean 73–89, max 102.3mm** | 0.91mm |
| (last, 부분) | +0.01~+0.10 | 동일 0 | — | — |

1. **전 도스 완전 무권한 — 풀도스 포함**: ramp12의 swap c-hat = **[contact 1.0(포화),
   fz 0.86, seal 0]**인데도 dFiLM ≈ 0 → "약함"이 아니라 **권한 0** (풀도스 강제 후속
   불필요해짐). 배선 검증: FiLM 키 전부 로드 + c-hat 실이동 확인.
2. **V1 시그니처가 pi0.5에서 재현**: 비접지 병목 = raw 접근보다 나쁨 — press-sim
   침투가 pi05 naive(off1 mean 39.6mm)의 2배(filmon 89.4mm). mask1이 raw 경로를
   지웠는데 c-hat이 죽어 있으니 진짜 force-blind.
3. **모방 동급** (0.89–0.91mm) — 매니폴드 위 무구분, 개입에서만 드러남 (일관).
4. 해석: suffix 주입 + 구캘리브레이션이 pi0.5 flow-expert에서 접지 실패 →
   **"병목+반사실 데이터만으론 부족; 주입점·캘리브레이션은 아키텍처별 재설계 필요"**
   (SmolVLA의 suffix<prefix·재캘리브레이션 필수와 일관) — Discussion "접지는 공짜가
   아니다" 문단 확정 재료. 잔여: last ramp12 2건(형식적, 진행 중).

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

1. ~~0729 4모델 train/val loss~~ **회수 완료 (08-06, `vla/0729_training_results.csv`** —
   fromnaive 계열 3종 포함 7모델; §3 loss 표·§3.6 판독 5 반영**)**.
2. 0729 롤아웃 리비전 확정 (val-best vs refs/main) — 남은 runs: refs/main 추정 6, val-best 1(실패).
3. 로봇: depletion sweep n≥10/층 (naive/FiLM/oracle), 중간층 2–4 보간.
4. ~~V1 decorrelated control~~ **완료 (08-05, §3.6)** — 예측 적중(스윕 평평·0/6 sim).
   잔여: V1 main 브랜치가 best/last 어느 쪽인지 서버 확인 + probe txt 커밋.
5. 층 분포 메타데이터 (lerobot 변환에서 layer tag 미보존 — 0729 sweep 수집 시 log-dir 규약
   `case_pick_<layer>` 준수).
