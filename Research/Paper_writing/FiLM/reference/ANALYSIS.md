# Reference 논문 평가 방식 분석 (Intro 주장 검증)

작성: 2026-08-11. force-aware 논문 17편 전문(pdftotext 추출본)을 4개 에이전트로 병렬 분석한 결과.

## 검증 대상 주장 (1_intro.tex)

> "All of this work shares a validation recipe---add the signal, retrain, and validate by task
> success (occasionally alongside rollout force statistics)---and none of it measures whether
> the trained policy *causally uses* the signal when it matters: whether, holding everything
> else fixed, changing the force input changes the action."

**결론: 인과 검증 부재(핵심 절)는 17편 전부에서 성립.** 어떤 논문도 학습된 정책의
force/tactile/torque **입력**을 inference-time에 개입(zeroing/perturbation/counterfactual
forcing)하면서 나머지 입력을 고정하고 action 변화를 측정하지 않음. 모든 ablation은
재학습 변형 비교. "report task success"가 너무 좁았던 부분은 08-11 수정으로
`(occasionally alongside rollout force statistics~\cite{phaforce,forceflow})` 삽입해 방어 완료.

**§II 반영 완료 (08-11)**: 아래 지뢰 4건 전부 2_related.tex에 선제 반영 — 클러스터 1에
TA-VLA 열거 추가 + hedge 미러링 + near-miss 구분 문장(TA-VLA/Tactile-VLA/ForceFlow),
클러스터 3에 FuSe·FM-VLA(인지했으나 SR로만 검증), 클러스터 4에 ForceSight(출력 goal 개입)
구분. "We contribute the missing measurement"를 "inference-time intervention on the trained
policy's force input, with the action as the readout"으로 구체화. 신규 bib: tavla, tactilevla,
fuse, fmvla, forcesight (저자 arXiv API 검증).

## 요약 테이블

| 논문 | venue | SR 외 보고 메트릭 | ablation 방식 | force 입력 인과 개입 | 판정 |
|---|---|---|---|---|---|
| ForceVLA (인용) | NeurIPS 2025 | peel length, MoE router 시각화(상관) | 재학습 | 없음 | 주장 지지 |
| PhaForce (인용) | arXiv 2026 | 평균 contact normal force, over/under-pressure 비율, wiping score | 재학습 | 없음 | SR-only엔 반례, 인과 절은 지지 |
| ForceFlow (인용) | arXiv 2026 | Force Fidelity MAE, rollout 힘 곡선, 힘 예측 head 검증 | 재학습 | 없음 (**단 Appendix E: vision 입력 마스킹 개입 있음**) | SR-only엔 반례, 인과 절은 지지 |
| FILIC (인용) | arXiv 2025 | SR only (force estimator 정확도는 센서 검증) | 재학습 (명시적 controlled ablation) | 없음 | 주장 지지 (가장 깨끗한 사례) |
| TA-VLA | CoRL 2025 | torque 예측 정확도(auxiliary), HSIC(상관) | 재학습 | 없음 (**토큰 노이즈 개입은 baseline π0의 encoder/state 토큰 대상**) | 주장 지지 |
| Tactile-VLA | arXiv 2025 | 지시어별 출력 힘(N): "softly" 0.51N vs "hard" 2.57N | 재학습 | 없음 (**language 입력 dose-response는 있음**) | SR-only엔 반례, 인과 절은 지지 |
| TaF-VLA | arXiv 2026 | SR only | 재학습 | 없음 | 주장 지지 |
| FM-VLA | arXiv 2026 | inference latency | 재학습 | 없음 | 주장 지지 |
| FoAR | RA-L 2025 | task score, ASR | 재학습 | 없음 | 주장 지지 |
| ForceMimic | ICRA 2025 | rollout 힘 곡선 (평균 9N vs 예측) | 재학습 | 없음 | 주장 지지 |
| Adaptive Compliance Policy | ICRA 2025 | 예측 stiffness 시각화(출력 관찰) | 재학습 | 없음 | 주장 지지 |
| Reactive Diffusion Policy | RSS 2025 | 물리 교란 회복 score, 촉각 반응 보정 시각화 | 재학습 + 데이터 변형 | 없음 (교란은 물리적 → 모든 입력 동시 변화) | SR-only엔 반례, 인과 절은 지지 |
| ForceSight | ICRA 2024 | fingertip 거리, force RMSE | 혼합 | **있음 — 단 force goal(출력)에 대한 개입** (90%→45%, 동일 초기조건) | 스코프 밖 (force가 정책 입력이 아님) |
| ForceVLA2 | ICRA 2026* | SR only (힘 관련은 정성적 데모) | 재학습 | 없음 | 주장 지지 |
| FAVLA | ICRA 2026* | 평균 peak contact force (7.7N vs π0 12.0N) | 재학습 (inference 개입은 실행 주파수 대상) | 없음 | SR-only엔 약한 반례, 인과 절은 지지 |
| FD-VLA | ICRA 2026* | SR only (inference에 force 입력 자체가 없음 — distilled token) | 재학습 | 없음 | 주장 지지 |
| FuSe | ICRA 2025 | SR only (touch/audio) | 재학습 + auxiliary loss ablation | 없음 (modality-모호 task 설계로 간접 확인) | 주장 지지 |

## 리뷰어 방어용 지뢰 목록

1. **ForceFlow Appendix E** — 학습된 정책의 **vision** 입력을 inference-time에 완전 마스킹하는
   진짜 개입 실험 존재 (readout은 SR: Plug/Insert 0%, Stamp 80–90%). force 입력은 안 건드림.
   → intro 주장은 "changing the **force** input"으로 스코프돼 있어 안전. "no inference-time
   intervention at all" 같은 넓은 표현은 금지.
2. **Tactile-VLA Table 2** — 동일 장면에서 부사만 바꾸는 **language** 입력 held-fixed
   dose-response (출력 힘 단조 변화, novel 부사 "harder"에 2.94N). tactile **입력** 개입은 없음.
   → language→force-output 인과는 보였지만 force-input→action 인과는 미검증.
3. **TA-VLA Table 2/4** — inference-time 토큰 노이즈 주입 실험 있으나 대상은 **baseline π0의
   encoder/state 토큰**(adapter 배치 위치 선정용). 학습된 torque-aware 모델의 torque 채널은
   안 건드림. HSIC(Fig 3)는 표현 간 통계적 의존성 = 상관, 개입 아님.
4. **ForceSight** — 동일 초기조건 matched comparison으로 force goal 무시 시 90%→45% 입증.
   단 force는 네트워크의 **출력**(예측 goal)이고 F/T 센서는 hand-coded 저수준 컨트롤러로만
   들어감 → "force를 backbone에 융합" 스코프 밖. 주장 문장이 force-as-input 정책으로
   한정돼 있는 한 안전.

## 논거 강화용 재료 (분야가 문제를 알면서도 측정하지 않음)

- **FM-VLA §3.2.1**: wrench history *길이*가 에피소드 진행도를 누출해 모델이 shortcut한다고
  스스로 진단 → 랜덤 노이즈 prefix augmentation으로 패치 → 검증은 다시 SR로만.
  ("even authors aware of shortcut learning validate the fix only by end-task success")
- **FuSe §III**: "the finetuned model empirically tends to predominantly rely on the
  pre-training modalities, ignoring the new sensors" — 문제를 명시하고도 대응은 auxiliary
  loss, 검증은 SR. modality-모호 task 설계(예: "round object that feels squishy")는 환경/프롬프트
  변화이지 held-fixed 입력 개입이 아님.
- **ForceVLA2 §5.2**: "simple concatenation can act as nuisance input" — 진단 근거가 SR 비교뿐.
- **ForceVLA §5.5**: "simply adding force input does not ensure closed-loop adaptation" —
  역시 재학습 변형의 SR 비교에서 추론.

## 방법론 메모

- 원문: `arxiv.org` PDF → pdftotext 추출본 (`scratchpad/texts/`, 세션 종료 시 소멸 — 필요하면
  이 폴더의 PDF에서 재추출).
- 구분 기준: **재학습 ablation**(w/o force 변형을 따로 학습)과 **rollout 힘 통계**(관찰)는
  인과 검증이 아님. **inference-time 입력 개입**(학습된 모델 고정, force 입력만 변경)만
  인과 검증으로 인정. 이 기준은 intro의 "holding everything else fixed" 문구와 일치.
