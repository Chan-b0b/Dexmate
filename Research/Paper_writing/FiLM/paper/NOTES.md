# 초안 상태 노트

## 2026-09-15 (저녁, 이 컴퓨터) — "few thousand steps" 삭제 + 초록·서론 구조 프레임 정렬
- 빌드 환경: 이 머신은 sudo 없음 → 사용자 공간 TeX Live 2026 (`~/texlive/2026/bin/x86_64-linux`, scheme-medium
  + ieeetran/pgfplots/makecell/multirow). 빌드 = `pdflatex; bibtex main; pdflatex; pdflatex`. **8쪽, 경고 0.**
- **"a few thousand steps of continued training suffice" 주장 삭제** (사용자): 근거가 단일 실행 val-best 위치
  (2,500/20k) 하나라 얇고, "초기 ckpt 고른 것 아니냐" 방어(20k 마지막 ckpt +0.97→+3.44)까지 끌고 와야 해서
  사슬 전체 제거. 초록·§I 기여②·§VIII 문구 제거, §V 서두는 "val-best checkpoint of a 20k-step run" 한 문장으로.
  retrofittable 성질은 구조(항등 초기화·naive ckpt에서 이어 학습)로만 주장.
- **초록 ②** 내용 보강: "붙일 수 있다"가 ①과 겹쳐서 → "붙여서 같은 데모로 더 학습하면 모방오차 변화 없이 힘 반응
  방식이 바뀐다"로.
- **§I 부분 재정렬** (사용자 "다 진행"): 1단락 감사 논지("none measures causal use", "pins the trajectory not the
  mechanism") 삭제 → "how force is delivered" 구조 질문으로. 기여 ①에 mask 명시 + matched ablation 이동, ② =
  retrofit만, ③ 로봇 수치(15N/2.5N) 제거(5단락·초록에만). 2단락 대시 중첩 문장 풀어쓰기. 마지막 한 문장 요약 =
  "같은 데모·백본·신호에서 어느 정책이 나오는지는 경로가 결정". 4·5단락은 유지 (사용자 검토 대기).
- **제목 확정 (09-15 저녁)**: "How Force Enters Matters: FiLM Conditioning for Behavior-Cloned VLA Policies".
  구 워킹 타이틀 "Grounded Force Conditioning … Brakes Beyond the Demonstrations"(09-14)는 성질 셋을 부제에
  다 넣어 길어서 폐기. 초록 논지 "how force is delivered, not what"을 제목으로.
- **초록 추가 다듬기 (09-15 저녁, 사용자 제안 → 합의)**: 태스크 일반화("suction picking task with a
  force-sensitive payload", 배터리·case 삭제) / 15N·2.5N 숫자 삭제 / naive 실패 톤 완화("shows the familiar
  failure: the wrench in its state does not come to govern its behavior") / "state token·raw wrench masked"
  삭제(mask는 트릭이지 구조적 발견 아님 — 사용자) / matched-ablation 문장 삭제(초록에 mask·bypass 미등장) /
  drop-in = "applies to any VLA that consumes a proprioceptive state"(π0 언급 삭제) / retrofittable에서
  "at no change in imitation error" 삭제(①과 중복) / 성질 ③ = **"generalizes beyond the demonstrations"**
  (힘=외삽, 층 높이=내삽이라 extrapolate 부적절; "naive's response fades" → "imitation alone leaves the
  response unspecified") — §I 기여 ③ 표제도 동일 변경. §I 1단락 wrench 첫 등장에 "(the 6-D wrench)" 정의.
- **용어 정리 (09-15 밤, 사용자)**: ① "grounded/grounding/ungrounded" 논문 전체 제거 — 논지는 "같은 힘 정보를
  FiLM 구조로 넣었다"이고 shuffled 대조군은 "ĉ 값의 역할을 보는 대조"로만 서술 (표 1 access 열 "ĉ only" /
  "shuffled ĉ only", §V-C 소절 제목 "the mask and the ĉ values", "Grounding, not capacity" → "The values, not the
  capacity"). ② 산문 "conditioned policy" → **"FiLM policy"** 전면 치환(~35곳, 파생 "conditioned pick" →
  "FiLM-policy pick"); 표·범례 "FiLM (ours)" 유지. §I 정의 문장 = "We call the policy trained behind it the
  FiLM policy (``FiLM (ours)'' in tables and figures)". §II의 "force-conditioned" 일반 용법은 유지.
  ③ §I 2단락: 배터리 셀·15N 허용치를 태스크 소개로 앞당기고 abort 문장의 15.4–18.7N 삭제(§III·§V-D에 있음).
  ④ §I "computed" 이탤릭 제거. ⑤ 초록 "externally aborted" → "drives contact force past the limit".
- **전체 정합 패스 (09-16 새벽, 14건, 제안→사용자 컨펌 방식)**: 구 감사 프레임·오늘 초록에서 뺀 표현의 잔재를
  §I–§VIII에서 제거. ① §I 기여② "at unchanged imitation error" 삭제 ② 기여③ "naive's template saturates" →
  "imitation alone leaves the response unspecified" ③ 요약문 "indistinguishable in loss" → "in imitation error"
  ④ **§II 첫 단락 후반 재작성**: "we contribute the missing measurement" 삭제, 선행 연구를 "힘을 어떻게 넣었나"로
  정리, 우리는 가장 가벼운 선택(백본 무변경·사후 부착), 프로브는 "평가도 다르게 한다" 수준 ⑤ §II ĉ 단락
  "sole, measurable route"·mask 삭제 ⑥ §II "usage/form dissociation"(데이터가 사용 여부 결정) 문장 통째 삭제 —
  미검증 ⑦ §II 마지막 단락 "prescription"·"whether it uses the signal at all" 제거 ⑧ §III 마지막 단락의 진단
  가설 3개·"the instrument we build next" 삭제 → §IV 포인터 1문장 ⑨ §IV 서두 "measurable"을 세 번째 성질이
  아닌 추가 성질로("; it is also measurable") ⑩ "recipe/ingredient" 7곳 → "ablation controls / element of the
  pathway / carry the effect / injection site matters" (ablation 용어는 유지, 사용자) ⑪ **§V-B bypass 단락**:
  "bypass of Sec. I" 깨진 참조 해소(여기서 정의), loss 4째자리 수치 삭제, 이유 명시 — 시연 위에서 두 경로가
  구분되지 않아 옵티마이저에 압력이 없고 retrofit에선 raw 경로가 이미 학습됨; base 학습 쌍(7% vs 76%)도 동일
  ⑫ §VII "the data decides whether, the pathway decides the form" 삭제, "bypass audit" → 평이한 표현
  ⑬ "prescription" 2곳 → demonstrations/design ⑭ **§VIII 재작성**: 초록 톤과 일치, mask·audit·active
  ingredient·"Imitation pins down the trajectory" 제거, 마지막 문장 = 제목 회수.
- **성질③ 귀인 변경 (사용자 지적)**: "because ĉ is a calibrated monotone scalar" → "because force magnitude
  reaches the action through FiLM's affine modulation" (초록·§I·§VIII; §V-A P3는 "unsaturated force scalar
  applied through affine modulation"). 이유: 단조 외삽의 원인은 표현(스칼라)보다 γ·h+β 구조이고, 그게 논지
  "how force enters matters"와 맞음. 스칼라 비포화는 필요조건(π0 각주). "ĉ를 concat으로 넣은" 대조군은 없음.
- 빌드 09-16 새벽: **8쪽, 경고 0**. 잔여 검색(grounded/ingredient/recipe/prescription/audit/fourth decimal/
  conditioned policy) 본문 0건.
- **09-16 오전 추가 (사용자 컨펌)**: "second backbone" → "another backbone" 3곳(§I 기여①·§V-C 제목·§VIII) /
  §I 4단락(시연 설계 한 문장) 삭제 — mask·bypass 문장 제거 후 고아가 됨, §III에 상세 있음 / §I 5단락에
  "crossover" 정의 추가("the two responses trading places as force rises past the demonstrated range") /
  §I 2단락 "force is the only signal" → "the task hinges on force" / 기여① "never stops" → "does not stop in the
  press simulation". **Fig. 2 (architecture)**: ⊗ 노드의 × 를 mathtext 대신 선분 2개로 그려 중심 정렬, 하단
  이탤릭 라벨 "the only pathway from force to action" 삭제; `scripts/make_architecture.py` OUT 경로를 스크립트
  상대경로로 수정(구 절대경로는 다른 컴퓨터용). 이 머신엔 matplotlib 없음 → `uv run --with matplotlib`.
- **PaperPlaza 제출 점검 (09-16)**: 1차 업로드 거부 사유 = "Type 3 fonts on page 3" (Fig. 2 architecture.pdf,
  matplotlib 기본 pdf.fonttype=3). `make_architecture.py`에 `pdf.fonttype=42` 추가 후 재생성 → PDF 폰트 Type0/Type1만.
  용지 Letter·여백·Overfull 0 확인. 제목 줄바꿈 "…FiLM Conditioning\\for Behavior-Cloned VLA Policies"(두 줄),
  Index Terms "imitation learning, vision-language-action models, force and tactile sensing, contact-rich
  manipulation". ICRA 2027 double-anonymous 확인(저자 블록 Anonymous 유지, Dexmate Vega-1p 모델명은 사용자 결정으로 유지).
  PaperPlaza 키워드 권장: Imitation Learning / Force and Tactile Sensing / Deep Learning in Grasping and Manipulation.
- 미결 유지: §V-B "정보가 더 많은데도 naive보다 나아지지 않음" 문장 추가 여부.

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
