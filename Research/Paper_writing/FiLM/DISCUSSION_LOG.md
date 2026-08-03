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
