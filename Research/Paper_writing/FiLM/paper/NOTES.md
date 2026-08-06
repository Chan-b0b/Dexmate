# 초안 상태 노트 (v0 스켈레톤+전섹션 드래프트, 2026-08-06)

빌드: 로컬 LaTeX 없음 → **Overleaf에 `paper/` 통째 업로드** (main.tex 루트). `\todo{}`(빨강)·
`\decide{}`(파랑) 매크로가 본문에 렌더링됨 — 제출 전 전부 해소 후 매크로 제거.

## 섹션 상태
| 파일 | 상태 | 비고 |
|---|---|---|
| 0_abstract | v0 완성 | ~210 words — 상한 확인 필요 (ICRA PaperPlaza 제한) |
| 1_intro | v0 완성 | nugget 마지막 문단, contributions 4개 = claim 1–4 |
| 2_related | v0 완성 | CGP(2603.05687) 인용 미배치 — cluster (a)에 끼울지 결정 |
| 3_task | v0 완성 | action space 7-D 표기 TODO verify |
| 4_method | v0 완성 | **프로브 용어 확정판**: P1 condition forcing / P2 state transplant / P3 dose-response sweep / P4 press simulation / P5 live counterfactual |
| 5_offline | v0 완성 | **§V Experiments 우산으로 통합 (08-06 사용자 지시)** — A~D 오프라인. Table I(bypass), Table II(quintuple, table*) 포함. contact-z 바이모달 분석 = TODO(파케이 재도출 대기) |
| 6_robot | v0 완성 | §V의 E~F 서브섹션 (Robot validation). Table III(robot), 리비전 각주 TODO, 토론 6 방어 문단 포함 |
| 7_discussion | v0 완성 | fidelity trap(0727 개발단계 관찰) 포함 — Q2 제외목록에 없음, 사용자 확인 대기. exposure 무수치 한 줄 = \decide 마커 |
| 8_conclusion | v0 완성 | |
| refs.bib | 시드 | 2025–26 arXiv 엔트리 저자/제목 TODO-verify 다수 → bibliography-auditor 패스 필요 |

## 08-06 사용자 결정 (질문 4건 회신 반영)
- **타이틀: "loss" 단어 금지 + loss 축 디강조 희망** → 워킹 타이틀 변경:
  **"Access Is Not Use: Auditing Causal Bypass of Force in Behavior-Cloned
  Vision-Language-Action Policies"**. §V-A 헤딩도 "standard validation cannot see
  causal use"로 변경. **Abstract의 "training loss is blind to causal use"도 삭제 (08-06
  추가 지시)** → "a difference invisible to standard validation"으로 대체. Intro의 해당
  문장은 유지 중 — 더 빼길 원하면 polish 패스에서. loss-free 대안 후보: "Imitation Pins
  the Trajectory, Not the Mechanism" / "Same Demonstrations, Different Mechanism".
- **Fig.1 티저: 텍스트 마무리 후 논의** (보류).
- **fidelity trap(0727) 문단: 삭제** (0729-only 순수성 서사 수준까지 확장).
- **exposure 무수치 한 줄: 포함 확정** (Limitations, \decide 마커 해소).

## 집필 중 내린 결정 (사용자 확인 대상)
1. ~~워킹 타이틀 = 후보 1~~ → 위 08-06 결정으로 대체.
2. **프로브 용어**: state-swap → "state transplant (P2)", 합성 c-hat → "condition forcing (P1)",
   fcscale → "dose-response sweep (P3)". EVIDENCE의 혼용 금지 요건을 P번호로 해결.
3. **quintuple 명명**: baseline(best/last) / "FiLM, wrench kept"(mask0fn) / "FiLM, mask,
   shuffled ĉ"(V1) / "FiLM, mask, grounded"(v2 best/last). V1/v2/mask0fn 등 내부 코드명 본문 미사용.
4. **pi0.5 배치**: naive 일반성 = §V-D (오프라인 소견), film 이식 실패 = §VII "Grounding is
   not free" (Discussion).
5. **캘리브레이션 2세대 정직 각주** (§IV eq.1 각주): Table I 모델 = 구캘리브레이션, quintuple =
   재캘리브레이션+fmag — pi05 이식 실패와 연결해 "캘리브레이션 품질 자체가 소견" 프레임.
6. **§III에 데모 설계(press-retreat)를 선치** — naive도 같은 데이터로 학습됐음을 명시해
   로봇 대비가 form 축을 분리함을 구조적으로 보장.
7. **exposure 소거**: 본문 주장에서 완전 제거, Discussion limitations에 무수치 한 줄 후보로만
   (\decide 마커) — 포함/삭제 사용자 결정.
8. **익명화**: 로봇 브랜드·회사·HF repo 전부 미기재. "dual-arm mobile manipulator",
   "suction case picking" 일반 서술.
9. **§V = Experiments 통합 (08-06 사용자 지시)**: 구 "V Offline Findings"+"VI Robot
   Experiments" → 단일 "V. Experiments" (A bypass / B binding / C quintuple / D scale /
   E robot picks / F live counterfactuals). 최종 섹션 수 7 (I~VII).

## 수치 출처 맵 (EVIDENCE.md 기준)
- Table I: §3 loss 표 + val probe (76/54, 7/8, 60/42; realistic 7/8, 3/4, 10/1)
- Table II (quintuple): §3.6 표 그대로 (6행)
- Table III (robot): §3 롤아웃 표 (0/3, 5/7, 1/3)
- §V-D pi05: §3.8 naive 블록 / §VII 이식 실패: §3.8 film 블록
- §VI live: §3.7 run3(공정)·run4(고도스)
- §VII 물리 산술·fidelity trap: §4

## Figure 계획 (전부 미제작 — 0-compute 데이터는 probes/*.txt에서)
- Fig.1 (fig:teaser — 아직 본문 \ref 없음): **재설계 필요** — 구 설계 A패널(depletion sweep)이
  Q2로 제외됨. 제안: (A) 태스크+overpress vs gentle-pick force trace 페어 (B) same-loss 막대
  (0.15075 vs 0.15087) + authority 막대 (7% vs 76%) (C) dose-response 3곡선 + live crossover
  오버레이. ← 사용자 논의 대상
- Fig.2 (fig:setup): 셋업 사진 + layer별 head-cam 프레임 ("look identical") — **로봇 해체 전
  사진 필요 (시간 민감)**
- Fig.3 (fig:arch): c-hat → FiLM(prefix/suffix) + mask 블록도 (TikZ)
- Fig.4 (fig:loss-authority): loss vs authority 산점도 (7모델)
- Fig.5 (fig:decomp): transplant 분해 + contact-z 분석 (파케이 재도출 후)
- Fig.6 (fig:doseresponse): 스윕 곡선 naive/ungrounded/grounded (+pi05 점선)
- Fig.7 (fig:live): live probe dz 비교 (run4 고도스)
- ⚠ 본문 \ref{fig:...} 라벨들에 대응하는 figure 환경 아직 미삽입 — Overleaf 첫 컴파일 시
  undefined reference 경고 정상.

## 페이지 예산 우려
현재 프로즈 분량 추정 본문 ~6.5–7쪽 + 표 3개(1개 table*) + 그림 7개 → **8쪽 초과 확실**.
1차 컴파일 후 감축 후보: Related 압축(0.5쪽), §VII pi05 문단 압축(\decide), Fig 통합
(4+5, 6+7), 토론 6 방어 문단 축약.
