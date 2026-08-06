# 토론 로그 — ICRA FiLM 논문 (다른 기기에서 이어서 작업용)

세션 기록. 최신이 아래. `OUTLINE.md`(논문 골격)·`EVIDENCE.md`(정량 증거맵)와 함께 읽을 것.

---

## 2026-07-30 ~ 08-03 — 초기 기획 + 프레이밍 토론

### 확정된 결정
1. **벤류**: ICRA 2027 (마감 2026-09-15, 8쪽 참고문헌 포함, double-anonymous). RA-L(+ICRA
   option, 12/31) 백업 — 미결.
2. **데이터 정책**: 논문 실험은 **무조건 0729 데이터로만**. Table I(bypass 매트릭스)은 0729
   4모델(naive/prefix_mask1/prefix_mask0/suffix_mask1)로 구성 가능 확인.
3. **중간층 평가 진행**: 학습 층1·5, 평가 층2–4 포함 — depth cue OOD × force cue in-dist
   분리 실험. 논문의 클라이맥스. (07-30 저녁 L1/L3/L5 sweep 시작됨 — rollouts 커밋 참조.)
4. **vision-info 미미함 강조**: head cam 단일(손목캠 없음), 층간 높이차 시각적으로 거의 구분
   불가, 몇 mm가 seal 성패 결정 → 관측 비대칭이 "깨끗한 진단 세팅"의 근거.
5. **수치는 전부 예비값** — 실험 캠페인 진행 중. 서사·구조·실험설계만 확정, 숫자는 추후 일괄
   교체. 드래프팅은 수치 비의존 섹션(Intro/Related/Method)부터.
6. 프레이밍: 진단 중심 **"Same Loss, No Cause"** 방향으로 진행 중 (사용자 명시적 최종 승인은
   아직 — 토론으로 다듬는 중).

### 토론 1 — "Contact VLA도 bypass한다고 할 수 있나?"
결론: **3층위로 구분해서만 주장.**
- (a) 주장 가능: "그들의 증거(success rate, 재학습 ablation)로는 인과적 사용이 입증되지 않는다"
  — 재학습 ablation은 다른 함수 비교라 결정-순간 사용을 못 봄.
- (b) 주장 가능: **존재 증명** — 우리 mask0 FiLM(=미니 force-fusion)이 동일 loss·권한 0으로
  실제 bypass. "전용 경로가 있어도 데이터가 상관되면 면역이 아니다."
- (c) 주장 불가: "ForceVLA는 bypass한다" — 대신 **반증가능 예측**으로: "상관 데이터로 학습된
  force-fusion은 접촉-순간 권한 ~0일 것" (여유 시 pi0.5/ACT로 자체 부분 검증).
- 정직한 한정: 그들 태스크(삽입 jamming 등)는 vision이 원리적으로 못 보는 force 정보가 있어
  shortcut 자체가 없을 수 있음. 우리 주장 = "shortcut이 존재할 때 그걸 감지할 수단이 없다".

### 토론 2 — 차별성 4축
1. **질문**: "force 넣으면 좋아지나"(성능) vs "결정 순간에 인과적으로 쓰나"(감사/audit).
2. **계측 해상도**: 우리 all-1 probe(67–82%)조차 속았고 채널분해가 사후 seal 바인딩임을 폭로
   — "force가 중요하다"와 "임계 전이는 force 무시"가 공존 ("right signal, wrong moment").
   그들의 ablation은 all-1 probe보다도 거친 도구.
3. **발견**: loss-blindness 실측 + access/exposure/decorrelation 소거 + 사후 바인딩 구조.
4. **처방 방향**: 아키텍처 추가(그들) vs 데이터 상관구조가 활성 성분 + 물리가 반응형 배제 →
   anticipatory 데이터 설계(우리).
- 한 줄: "그들은 force를 넣는 법을 제안하고, 우리는 넣은 force가 쓰이는지 재는 법과 안 쓰이는
  이유, 쓰이게 만드는 조건을 제시한다."

### 토론 3 — 왜 "loss로 못 본다"를 강조하나
- ① 실무자의 유일한 계기판 고장 (배포 안전성: loss+몇 번 성공으로 검증된 모델이 스택 높이
  변화에 과압) ② **버그가 살아남는 기제 그 자체** (likelihood가 shortcut에 만족 → 인과 특징으로
  갈 gradient 압력 0 → access/exposure 처방이 실패한 이유) ③ 계측기 기여의 존재 이유.
- 강조 방식 수정: "loss 하나"가 아니라 **"표준 검증 스택 전체의 단계적 실명"** (loss → success
  → coarse ablation → all-1 probe까지 순차적으로 눈멀고, moment-resolved 개입에서야 보임).
- "당연한 얘기 아니냐" 방어: 원리는 causal confusion 문헌에 있으나, 소수점 셋째 자리까지 같은
  loss에 권한 0 vs 67%라는 실측 사례 + 가시성의 정량적 경계선은 새로움.
- **검증 대기 예측**: 0729 press-retreat 데이터에선 loss-blindness가 부분적으로 깨질 것 —
  후퇴 시점이 랜덤 힘목표(8–15N) 조건이라 depth만 보는 naive는 전이 구간 val loss가 higher여야
  함. 서버 loss 수치 회수 시 확인. 맞으면 "decorrelation이 표준 지표의 판별력 자체를 복원"으로
  주장 완결.

### 토론 4 — "중요한 정보(seal·contact 포함)를 보게 만드는 게 핵심" (사용자 설계 철학)
- 동의 + 배치: **Goal = 설계 철학** (사람이 참조하는 물리 조건 — "접촉했으면 멈춰, 씰 잡혔으면
  올려" — 을 정책도 참조해야; PROGRESS.md 원 동기), **Problem = 넣어도 안 보고 잴 수단 없음**,
  **Solution = 계측기 + 채택 성립 조건**. 헤드라인으로 걸면 method-paper 함정(PhaForce도 같은
  주장)이므로 Goal 자리에.
- 정제된 명제: **채택은 신호의 '중요도'가 아니라 '통계적 성질'이 결정** — seal(안정·지속)은
  공짜로 배워짐(naive조차 lift를 seal에 게이팅 65%; FiLM 단일채널 최강 21–28%, live +2.09mm/f),
  contact(1–2프레임 транз이언트+depth 중복)는 반사실 없이는 절대 채택 안 됨. 복수 채널 설계
  덕에 채널분해 발견이 가능했음.
- **Zero-new-information (state 구성 확인됨)**: state(15) = pos3+quat4+suction1+seal1(DI0)+
  wrench6. c-hat 전 채널이 naive도 받는 신호의 결정론적 함수 → conditioned 모델의 추가 정보
  0, mask1은 오히려 정보 제거. "센서 추가" 반론 원천 차단, 센싱 논문이 아니라 **학습 역학
  논문**. → OUTLINE claim 1·abstract·§IV에 반영됨.

### 진행 상황 / 다음 할 일
- [x] OUTLINE.md / EVIDENCE.md 작성 (수치는 예비값 표기)
- [x] 0729 held-out val probe 회수·정리 (mask0 7–8% vs mask1 54–76%; val-best>last)
- [x] 0729 로봇 롤아웃 정리 후 재계산 (naive 0/3 vs prefix 5/7) + git 추적 시작
- [ ] **V1 decorrelated control (0729 데이터, GPU)** — 최우선
- [ ] 층 sweep 진행 중 (L1/L3/L5 시작, 07-30 저녁) → n≥10/층, 체크포인트 리비전 pin
- [ ] 서버에서 0729 train/val loss 회수 (+ 전이구간 loss 예측 검증)
- [ ] 과거 라운드(0708/0721/0727) 증거의 본문 사용 범위 결정 (OUTLINE §11 Q2)
- [ ] IEEEtran 스켈레톤 + Intro/Method 드래프팅 (프레이밍 최종 승인 후)

### 참고 경로
- 실험 일지: `LGES/vla_training/experiment_docs/EXPERIMENTS_CASE_PICK_0708.md` (§6.8까지)
- probe 원문: `LGES/vla_training/probes/` (0729* = 최신)
- 롤아웃: `LGES/vla_training/rollouts/` (git 추적됨, 커밋 58db51d부터)
- 관련연구 지도: `Research/condition_driven/related_work/RELATED_WORK.md` + 신규(ForceFlow
  2605.11048 최근접, CGP 2603.05687, 2507.22380, Shortcut-in-Generalist 2508.06426)

---

## 2026-08-04 — 0729 recal/fromnaive 라운드 분석 + "형태 결정 장치" 주장 확정

서버 커밋 5f2ea6d(experiments)/e58fa27(post_train) 회수 후 분석 세션. 수치는
EVIDENCE.md §3.5에 정리 (원문 경로 병기).

### 확정된 결정 (사용자 승인)
1. **핵심 주장 문장 확정**: **"FiLM+mask1은 정책이 힘을 쓰게 만드는 장치가 아니라,
   힘이 사용된다면 어떤 형태로 사용되는지를 결정하는 장치다."**
   - **usage(쓸지/얼마나)는 데이터가 결정** — 우리 기록이 근거: 0708 현실 접촉 권한 0%,
     §6.6 oversampling 강제 실패, 0729 press-retreat+바이모달 높이에서야 권한 발생.
     "힘을 위주로 보라"는 설계는 어디에도 없음 (보조 loss 없음) — 이 인정이 주장의
     정직성을 지탱.
   - **form(어떤 형태로)은 설계가 결정** — 단조 캘리브레이션 스칼라 채널 + mask1 병목의
     귀납편향. 1차 증거 = 힘-스케일 스윕의 **기울기 부호**: 주입 |F| 8→12N에서 naive
     상쇄 +1.41→+1.12mm(단조감소, 템플릿 바인딩) vs fromnaive +1.20→+4.40mm(단조증가,
     12N=완전정지+후퇴; fmag c-hat 6.5로 학습범위 5배 밖 외삽).
   - 인과 고정: **fromnaive = 통제실험** (동일 초기화·동일 데이터·병목만 추가 → 형태
     반전) + mask0 붕괴 (병목만 제거 → 7–9%, 3개 데이터셋 재현).
2. **live probe는 dz 비교만** (film vs naive 비교 플롯 dz 전용 — 구현 반영됨).
3. 사용 금지 서사 추가: ~~"naive는 force-blind"~~ — state-swap에서 naive 105% 반응
   (raw 경로). 이점 축은 '반응 여부'가 아니라 '반응의 형태 + 운영성(보정·감사·개입)'.

### 기존 서사와의 접속
- **토론 4 정제명제의 확장**: "채택은 신호의 통계적 성질이 결정"(usage) 위에
  "형태는 병목 설계가 결정"(form)을 얹음 — 두 축을 분리하면 §3 판독 4의 서사 수정
  (anticipatory 주기제)과 충돌 없음: press-retreat 데모가 usage를 만들고, 병목이 그
  행동(후퇴)의 **힘-단조 외삽**을 만든다. naive는 같은 데이터로 같은 행동을 배웠지만
  (swap 105%) 형태가 템플릿이라 외삽이 죽는다 — 동일-데이터 내 대비로 깔끔.
- fzdelta/seal-never press-sim(naive 4–5/6·max 14.9mm vs fromnaive 6/6·max 8.5mm,
  seal 제공 시 naive 정상)은 "naive 정지=seal 이벤트 의존, FiLM=힘 바인딩"의 폐루프
  버전 — over-press 실패 모드(§1 진단·0727 분석)와 직결.
- **운영성 주장 신규 재료**: F/T 드리프트 실측(하루 내 +0.4→+1.1N) + run별 재앵커
  구현(`--film-auto-baseline`). naive는 frozen stats라 보정 노브 자체가 없음 —
  Discussion의 배포 문단 재료.

### 토론 5 — "기존 VLA는 모방일 뿐, 인과관계에 의해 행동하지 않는다"와의 결합 (사용자 승인)
- **원문 그대로는 사용 금지** — 두 가지 이유:
  ① 우리 naive가 반례 (state-swap 105% = 개입에 대한 행동 변화 = 인과적 의존의 표준
  정의). naive의 문제는 의존의 부재가 아니라 (a) 결정 순간 바인딩이 물리 원인이 아닌
  상관 대리물(depth·seal)이고 (b) 의존의 형태가 템플릿이라 분포 밖에서 소멸하는 것.
  ② "모방→인과 없음"은 존재론적 주장 — causal confusion 문헌에 "알려진 얘기"로 치이거나
  "vision도 인과 입력" 반론으로 소모전.
- **정제 명제 (확정)**: "모방 목적함수는 데이터 매니폴드 위의 행동만 고정하고, 그 행동이
  '무엇 때문에' 나오는지(기제)는 미결정으로 남긴다. 어떤 신호에 바인딩될지는 목적함수가
  아니라 데이터의 상관구조가 결정하고, 표준 검증(loss·성공률)은 그 차이를 볼 수 없다."
  = **"모방은 궤적을 고정할 뿐, 기제를 고정하지 않는다."**
- 3단 논증 매핑:
  1) **동일 모방, 상이한 기제** — naive vs fromnaive: 액션 오차 0.81/0.84mm·loss 동급으로
     매니폴드 위 구분 불가, 개입(힘 스윕)하면 기제 정반대(템플릿 vs 단조). *모방이 기제를
     미결정으로 남김*의 실측.
  2) **기제는 상관구조의 사고(事故)** — mask0/0708: 같은 아키텍처·같은 loss, 상관 데이터면
     권한 0. "Same Loss, No Cause"가 이 명제의 정량 버전.
  3) **기여 접속** — 기제를 강제할 수는 없으나(usage=데이터) 형태를 결정·측정 가능하게
     만들 수 있다(form=설계, 결정 1).
- 스코프 한정: "기존 VLA 전부"가 아니라 "**상관된 시연으로 BC 학습된 정책, 상관 shortcut이
  존재할 때**" (토론 1의 정직한 한정과 일관).
- 배치: 사용자 원문 직관 = Intro 첫 문단 동기 → 즉시 정제 명제로 좁힘. 영어 한 줄 후보:
  *"Imitation pins down the trajectory but not the mechanism: equally faithful clones can
  act from opposite causes, and the loss cannot tell them apart."*

### 토론 6 — "중간 도스에선 naive가 더 세게 반응하는데, naive가 더 좋은 것 아닌가?" (예상 리뷰 질문, 08-06)
- **사실 인정**: 프레임 단위·중간 도스(≤9N)에서는 naive ≥ film — 오프라인 8N(+1.41 vs
  +1.20)과 08-06 공정-도스 라이브(fc 67% vs 32%, sealed 147% vs 58%) 모두. 템플릿 매칭은
  템플릿 지점에서 원래 강하다.
- **반박의 축 = 도스-반응 곡선의 방향**: naive는 템플릿에서 정점 후 힘↑=반응↓(12N +1.12),
  film은 단조 증가(12N +4.40 = 정지+후퇴). 실제 과압은 힘이 템플릿을 **지나쳐** 상승하는
  동역학이라, naive의 브레이크는 필요해질수록 약해짐 → 폐루프 발산 (sim 최대 14.9mm 관통,
  실기 0/3 overpress) vs film 수렴 (6/6, max 8.5mm, 실기 5/7).
- **이중 결함**: naive 최강 반응은 sealed 템플릿(+3.32)인데 over-press = seal이 안 오는
  상황 — 가장 센 브레이크가 부재 신호에 바인딩.
- 비유(본문 후보): naive = "정확히 그 소리가 나면 멈춤", film = "소리가 클수록 세게 멈춤".
- 08-06 라이브 공정-도스 결과는 오프라인 곡선의 중간 구간과 정합 (모순 아님) — 고도스
  (≥12N) 대비는 통제된 오프라인 계기로 측정했다고 서술. 라이브 film 수치는 전이 손실
  (suction 비트 오프-매니폴드 등)로 과소평가 가능성도 각주 후보.

### 남은 반론과 실험
- **"병목이 아니라 재학습/용량 효과" 반론 미차단** → V1 컨트롤이 정확히 이 실험.
  **fromnaive-V1**(naive init + 셔플 c-hat, 20k, recal 세팅 동일)로 돌리는 것이 최강
  — 예측: 힘-스케일 스윕이 평평(또는 naive형 감소). 스윕 재현되면 주장 완결.
- live probe(naive 비교판)의 `st_fz±N` 스윕 = 기울기 부호의 온로봇 재현 시험.

### 다음 할 일 (07-30 리스트에 추가/갱신)
- [x] **V1 (fromnaive-V1) 학습+probe — 완료 (08-05, EVIDENCE §3.6)**: 스윕 완전 평평
  (8N −0.06 / 12N −0.11mm), sim 0/6·max 431mm, val err 0.96mm(동급). **예측 적중 —
  용량 반론 사망, "형태 결정 장치" 주장(결정 1) 증거 사슬 완결.** 비접지 병목이
  naive보다도 나쁨(0/6 vs 4–5/6) → 활성 성분 = "접지된" 병목.
- [ ] fromnaive/v1 HF 업로드 완료; **pm1r 업로드 잔여** → 로봇 평가
  (`robot_eval_0729_recal.sh`, auto-baseline 포함)
- [ ] live probe 재실행 1회 — 08-04 런은 naive 쪽 결과만 유효 (env 파일 오프셋이
  probe 포즈 대비 ~0.7N 과보정 → film 쪽 c-hat 오앵커; 측정-포즈 민감성의 실증
  사례로는 사용 가능). 08-05 수정으로 probe가 하강 전 hover에서 자체 재앵커
  (`--baseline-hover`) — GPU 여유 시 재실행.
- [ ] 실험일지 §6.10 (recal/fromnaive/V1 라운드) 기록 — 로봇 결과와 함께
- [ ] EVIDENCE §3.5의 "state-swap 94–97%"와 §3 "realistic 3–10%" **측정법 구분 용어**
  확정 (실측-state 주입 vs 합성 c-hat 패턴 — 본문 혼용 금지)
- [x] V1 main = **val-best@5,000** (HF 커밋 메시지 확인 08-06; fromnaive=@2,500,
  mask0fn=val-best — 전부 val-best로 일관). 잔여: V1/live/mask0fn probe txt 커밋.
  서버 업로드 노트의 "DEPLOY: FILM_F0 = field hover baseline + 1.5" = 우리
  auto-baseline 재앵커와 독립 수렴 (운영 처방 문단 보강 재료).
- [x] **mask0-fromnaive probe (08-06) — 사중 대조 완성** (EVIDENCE §3.6 표 확장):
  c-hat 권한 0(Δ−0.04mm) + 분해 dRaw=전부/dFiLM=0 + 도스-곡선이 naive와 일치
  (+1.57→+1.12) → bypass 4번째 재현, "모듈만 붙이면 무변화" 정량화.
- [x] fromnaive **last(20k) ramp**: +0.97→+3.44 단조 유지 → 형태의 ckpt 안정성.
- [ ] pi05_naive_0729 오프라인 probe (업로드 완료 대기; 러너·policy-agnostic 수정
  준비됨) → 아키텍처 일반성 §8 행 복원 후보.
