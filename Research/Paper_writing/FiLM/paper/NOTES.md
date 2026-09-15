# 초안 상태 노트

## 2026-09-15 — 읽기 교정 패스 (사용자 지적 → 합의 수정) + 빌드 TeX Live 전환
- 빌드: 로컬 TeX Live 2023 설치됨 (사용자). latexmk 없음 → `pdflatex; bibtex main; pdflatex; pdflatex`.
  tectonic/XeTeX 가드는 남겨두었지만 이제 pdfLaTeX 경로가 기준. **8쪽, 경고 0.**
- 수정 6건: ① §IV-B "(initialized to identity)" 괄호 삭제, 항등 초기화는 retrofit 문장에 한 번(붙인 순간
  = 학습된 정책) ② SmolVLA 이름은 §V 서두 설정 문장 한 곳만 (작은 공개 모델 선택 이유 + π0 재확인), §III 실패
  문장에서 제거 ③ fig:setup 캡션 "is carried by" → "must be carried by" ④ §IV-B "preliminary experiments"
  각주 삭제 → §V-C π0 state/action 비교로 대체 ⑤ §IV-C 프로브 P1–P5 문단 전부 질문 문장으로 시작
  ⑥ 쪽수 복구: table* 2개를 §V 첫머리로 이동([!t]), §II loss 4째자리 절 삭제, §V-A 안내문·P1/P2 꼬리·§VII
  되풀이 문장 2개 삭제.

- 추가 합의 (09-15 오후): ⑦ P5 첫 문장 풀어쓰기 ⑧ §V 서두 두 대조군을 조작 그대로 서술(no mask / shuffled ĉ)
  ⑨ 2,500 스텝은 §V에서만 20k 맥락과 함께(마지막 ckpt에도 단조 형태 유지 +0.97→+3.44), 초록·기여는 "a few
  thousand steps" ⑩ 제안 모델 표·범례 이름 = **"FiLM (ours)"** (산문은 "the conditioned policy" 유지, §I 정의에
  괄호로 연결) — Table I 행, dose/live/traces 범례 전부 교체 ⑪ **그래프 전부 TeX**: 힘 궤적 그림도
  `scripts/make_force_traces_pgf.py` → `figs/robot_force_traces.tex` (pgfplots). 남은 이미지 = 사진 2장 +
  architecture.pdf(블록도). 구 matplotlib PDF/PNG는 figs/에 남아 있으나 미참조.
- 사용자 의문 "no mask가 더 못한다는 게 말이 되나" → 데이터는 "naive와 같다(새 정보를 안 씀)"이며 "더 못하다"가
  아님을 확인, 삭제 안 함. §V-B에 "정보가 더 많은데도 naive보다 나아지지 않음" 한 문장 추가 제안(미결).

## 2026-09-14 (저녁) — 구조 위주 재프레이밍 (사용자 결정, ICRA 9/15 마감 유지)
사용자 방향: **FiLM 경로(구조) 자체의 장점 위주** — ① drop-in(백본 무변경·정보 추가 0·모방오차 비용 0,
π0에도 붙음) ② retrofittable(학습된 naive에 붙여 2,500 스텝) ③ 못 본 값에 옳은 방향(12N 정지+후퇴, L3).
"같은 시연, 다른 기제"는 §V-A P3 해석 문장으로 내려감. 결과는 **오프라인 위주**로 서술하되 로봇 결과는
**수치·분량 전부 유지**(§V-D 폐루프 / §V-E 라이브) — 뉘앙스만 "오프라인에 더해 우리 실로봇에서도 해봤다"
(사용자: 실험을 덜 한 것처럼 보이면 안 됨). 0816 미수록, π0.5 전면 미언급, fromnaive 로봇 재실험 안 함.
- 워킹 타이틀: "Grounded Force Conditioning for Behavior-Cloned VLA Policies: A Drop-In, Retrofittable FiLM
  Pathway That Brakes Beyond the Demonstrations" (구 "Access Is Not Use…"는 main.tex 주석에 보존).
- 바뀐 곳: abstract 전면 / §I 기여 3개 = 세 성질 / §IV 제목 "The Conditioning Pathway" + retrofit 문장 +
  프로브 소절 "Evaluation:" / §V 서두 retrofit(2,500 스텝) / **§V-C = π0 복원** (naive 0/6 17.0mm vs
  film-state 5/6 7.2mm, action-token 0/6 22.9mm, err 0.93 vs 1.03, dRaw 0; 각주: fmag 채널 없음·오프라인만)
  / §V-D·E 로봇 서두 "Beyond the offline battery, we also ran…" + 인스턴스 공시 본문화, "Isn't the naive
  better" 논증은 §V-A P3 끝으로 이동(로봇 절엔 포인터 1줄) / §VII 운영 노브 문장·한계 갱신 / §VIII 재작성.
- 08-13 "action-injection 각주-only" 결정은 π0 복원으로 해제 — §V-C에 본문 1회.
- fig:doseresponse에서 π0/π0.5 점선 제거 (SmolVLA 4정책만).
- 빌드: 8쪽, 경고 0. 참고문헌이 8쪽 하단 끝까지 채움 (여유 없음 — 추가 시 감축 필요).
- **fig:setup 제작 (09-14 밤, 사용자 사진 제공 `paper/images/`)**: (a) `figs/setup_robot.jpg` (Robot_env.jpeg 크롭)
  (b) `figs/setup_pictogram.tex` TikZ 픽토그램 — 두 높이 스택 + 고정 헤드캠 + 접촉면 cm 차 + F<15N
  (Battery_stack.jpeg는 사용자 의견대로 사진 대신 그림; 원본은 images/에 보존) (c) `figs/setup_headcam.jpg`
  (Battery_box.jpeg 크롭, "층이 바뀌어도 시각 정보는 거의 안 변함"). §III 참조 2건 복원. 여전히 8쪽, 경고 0.
- **남은 구멍 (정직 공시로 처리)**: retrofit 인스턴스·π0 인스턴스 로봇 폐루프 없음; press-retreat 데모
  대조 없음(“쓸지는 데이터, 형태는 구조”); 과제·플랫폼 단일; Fig.1 티저 미제작(본문 ref 없음).

## 2026-09-14 (오후) — 마무리 패스 (0909/0816 라운드 미수록, 0729 증거로 확정)
사용자 결정: 0909 실험 결과는 논문에 쓰지 않음 → 기존 0729 증거만으로 마무리.
- **로컬 빌드 가능**: tectonic(XeTeX) 0.17 aarch64 바이너리로 컴파일 → **8쪽, undefined ref/overfull 0**.
  main.tex에 `\ifXeTeX` 가드로 TeX Gyre Termes(Times 메트릭 호환) 지정 — pdfLaTeX/Overleaf 경로 무영향.
  빌드: `tectonic main.tex` (paper/ 에서). Overleaf도 그대로 동작.
- **\todo 3건 해소**: ① Abstract 스케일 문장 = §V-C 기존 π0/π0.5 수치로 작성 ② §IV P2 전이 차원 확정
  (probe_state_authority.py: `--swap firstcontact/fcscale`는 wrench 6차원(state 9:15)만 교체, seal 비트
  불변) ③ fig:doseresponse 제작 — `scripts/make_dose_response_pgf.py` → `figs/dose_response.tex`
  (probes/0729_state_*_ramp*.txt ALL-frames 행, 표와 동일 수치; π0/π0.5 naive 회색 점선 포함).
- **\memo 3건 제거** (intro 문구 강등 / §IV mask-정보량 문장 부활 / loss 축 디강조): 본문은 현행 유지.
  loss 축 표현(abstract "fourth decimal", §I "unchanged validation loss", §V "fourth decimal", 마무리 문장
  "indistinguishable in loss")은 그대로 — 바꾸려면 §V-B 문단의 imitation error 0.85 vs 0.87 mm/step로 대체.
- **fig:setup 참조 2건 삭제 (§III)**: 셋업 사진·헤드캠 프레임이 로컬에 없음 (datasets_local은 meta만,
  rollouts는 states.jsonl만). 서버(B300) 0729 데이터셋 영상에서 L1/L5 프레임을 뽑으면 복원 가능 — 단
  현재 8쪽 꽉 참이라 넣으려면 감축 필요.
- **draft 매크로(\todo/\decide/\memo) 정의 제거** — 본문에 남은 사용 0.
- **refs.bib**: TODO-verify 13건 전부 arXiv abs 페이지 메타태그(ForceVLA/CGP는 Semantic Scholar)로 채움
  (제목·저자, 2026-09-14 기준). IEEEtranBSTCTL(저자 6명 초과 → 1st et al.) 추가해 참고문헌이 8쪽 안에 끝남.
  ForceVLA는 S2 기준 NeurIPS 2025 — camera-ready 때 booktitle 결정.
- **main.pdf 갱신**: 09-14 tectonic 빌드 결과 (8쪽). main.aux/.bbl 등 구 중간파일은 08-27 것 그대로.
- 미제작 유지: Fig.1 티저(본문 \ref 없음), fig:setup.

## 이전 노트 (2026-08-06 v0 스켈레톤+전섹션 드래프트)

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
