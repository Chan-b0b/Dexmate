# LGES case_pick 0708 — SmolVLA / FiLM 실험 기록

작성: 2026-07-11. H200 서버(`/home/maverick/Dexmate/LGES/vla_training`)에서 수행.
다른 PC에서 이어서 진행할 수 있도록 환경·결과·재현 커맨드·주의사항을 모두 기록한다.

## 1. 목표

case_pick 데모(suction, descend-until-contact)에 대해:
1. Naive SmolVLA 파인튜닝 베이스라인
2. FiLM contact-conditioning(`film_contact.py`, V2)의 **주입 지점(inject)** 과
   **force 마스킹(mask_force)** 이 c-hat의 실제 행동 권한에 미치는 영향 규명

## 2. 환경 (다른 PC에서 재현)

```bash
uv venv ~/vla_venv --python 3.12          # lerobot 0.5.1은 py>=3.12 요구
uv pip install --python ~/vla_venv/bin/python "lerobot[smolvla]==0.5.1" tensorboard
# 확인된 조합: lerobot 0.5.1 / transformers 5.3.0 / torch 2.10.0+cu128 / torchcodec 0.10
```

모든 실행 전 공통 env:

```bash
export VENV=~/vla_venv                     # train_*.sh가 이 venv를 사용
export CUDA_VISIBLE_DEVICES=4              # ← 2026-07-11부터 GPU 4 사용 (6은 혼잡)
export HF_HOME=$HOME/.cache/huggingface    # 공유 /data/cache/hf는 token 퍼미션 문제 있음
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE
```

- HF 인증: `huggingface-cli login` (Chanho-Lee 계정; 업로드에 필요)
- `--num_workers`: 기본 32로 상향(train_*.sh 반영됨). 12→24에서 0.55→약 4 step/s (~8배).
  30k 스텝 ≈ 2–2.5시간 (H200 1장, bs 32 기준).

## 3. 데이터셋

| repo | 액션 | 비고 |
|---|---|---|
| `Chanho-Lee/lges_case_pick_0708` | delta 7d (Δpos3+rotvec3+suction) | 75 eps / 17,783 frames / 15fps |
| `Chanho-Lee/lges_case_pick_0708_abs` | absolute 8d (xyz+quat wxyz+suction) | 동일 에피소드 |

공통 스키마: `observation.state` **(15)** = pos3+quat4+suction+**seal(idx 8)**+**wrench(idx 9:15)**,
카메라 `head`(RGB)+`head_depth`(turbo 컬러화) → rename_map으로 base의 camera1/2에 매핑, camera3 마스크.
`film_contact.py`의 인덱스(SEAL_IDX=8, WRENCH_LO:HI=9:15)와 일치해야 함.

로컬 다운로드 위치: `datasets/lges_case_pick_0708{,_abs}` (FiLM stats용 `meta/stats.json` 필요).

## 4. 완료된 실험 매트릭스 (모두 HF `Chanho-Lee/<run_name>` 업로드됨)

모든 런: `lerobot/smolvla_base`에서 시작, bs 32, cosine(peak 1e-4→2.5e-6, 30k 스케줄).
FiLM 런 공통: `FILM_VARIANT=v2`, `FILM_COND=contact,fz,seal`, F0=12, tau=10, fz_tau=30.

| run (delta / abs) | 세팅 | 최종 loss (delta/abs) | counterfactual probe (delta) |
|---|---|---|---|
| `smolvla_naive_0708{,_abs}` | naive, 30k | 0.112 / 0.014 | — |
| `film_on_naive_0708{,_abs}` | naive-30k ckpt에서 FiLM, suffix, mask0, 10k | 0.102 / 0.013 | FAIL (무권한) |
| `smolvla_film_0708{,_abs}` | from base, suffix, mask0, 30k | 0.101 / 0.013 | FAIL (max Δdz 0.8mm) |
| `smolvla_film_0708_prefix{,_abs→_abs_prefix}` | from base, prefix, mask0, 30k | 0.099 / 0.013 | FAIL (하강 게이팅 못 함) |
| `smolvla_film_0708_mask1{,_abs_mask1}` | from base, suffix, **mask1**, 30k | 0.116 / 0.014 | **WEAK: 하강 23% 상쇄** |
| `smolvla_film_0708_prefix_mask1{,_abs_prefix_mask1}` | from base, **prefix, mask1**, 30k | 0.101 / 0.013 | **PASS: 하강 67% 상쇄** ★ |

probe = `probe_film_authority.py` (case_pick 에피소드, 이미지·state 고정, c-hat만 0↔1 강제,
committed-descent 구간에서 dz 변화 측정). 원문: `probes/*.txt`.
abs 계열은 probe의 phase 판정이 delta 전제라 공식 INCONCLUSIVE지만 평균 Δ는 같은 순위
(prefix_mask1 +46mm ≫ suffix_mask1 +10mm ≫ mask0 ~0).

### 핵심 결론

1. **mask_force=1(force 마스킹)이 필수.** 원시 wrench가 state에 남으면(mask0) inject 지점과
   무관하게 학습이 c-hat 경로를 우회한다 — loss는 동일하게 낮아지지만 counterfactual 권한이 0.
   loss로는 이 차이가 절대 안 보임. 판단은 반드시 probe로.
2. **inject=prefix > suffix** (mask1 하에서 67% vs 23%). wrench를 마스킹한 채 c-hat을 state
   토큰에 주입하면 "증류된 접촉 센서"를 관측에 되돌려 넣는 셈이라 정책이 진짜 입력으로 사용.
3. **평가/배포 시 학습과 동일한 `FILM_COND` + `FILM_INJECT` + `FILM_MASK_FORCE` 필수.**
   추천 체크포인트: `Chanho-Lee/smolvla_film_0708_prefix_mask1` (delta) /
   `_abs_prefix_mask1` (abs) — `FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1`.

## 5. 재현 커맨드

```bash
# naive (train_smolvla.sh는 HF_DATASET_REPO/HF_CACHE_DIR 사용)
HF_DATASET_REPO=Chanho-Lee/lges_case_pick_0708 HF_CACHE_DIR=$PWD/datasets/lges_case_pick_0708 \
  RUN_NAME=<run> ./train_smolvla.sh --steps=30000

# FiLM from base (이긴 세팅). INIT_CKPT는 smolvla_base 스냅샷 경로:
BASE=$($VENV/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")
IF='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
FILM_VARIANT=v2 FILM_INJECT=prefix FILM_MASK_FORCE=1 RUN_NAME=<run> INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708 DATASET_ROOT=$PWD/datasets/lges_case_pick_0708 \
  FILM_DATASET_ROOT=$PWD/datasets/lges_case_pick_0708 \
  ./train_film.sh --policy.input_features="$IF" --steps=30000

# resume(스텝 연장 포함): 저장 config 사용. FiLM 런은 env를 그대로 다시 지정할 것.
FILM_VARIANT=v2 FILM_INJECT=prefix FILM_MASK_FORCE=1 FILM_DATASET_ROOT=... \
  $VENV/bin/python train_film.py --config_path=outputs/<run>/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=<확장 스텝>

# 권한 probe
FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_INJECT=prefix $VENV/bin/python probe_film_authority.py \
  --checkpoint outputs/<run>/checkpoints/last \
  --dataset-root datasets/lges_case_pick_0708 --repo-id Chanho-Lee/lges_case_pick_0708

# HF 업로드 (META 딕셔너리에 run 추가 후)
$VENV/bin/python upload_weights.py <run> [...]
```

오케스트레이션 예시는 `run_case_pick_0708.sh`, `run_mask1_sequence.sh` 참고
(병렬 launch + wait + 체인, save_freq=4000 + optimizer state 정리 패턴).

## 6. 함정 / 트러블슈팅 (전부 실제로 겪음)

- **lr 스케줄**: `steps < scheduler.num_decay_steps(30000)`이면 lerobot이 스케줄을 압축함.
  10k로 돌리면 10k에 lr 바닥. `--resume --steps=30000`으로 연장하면 30k 스케줄로 재구성되어
  lr이 ~7.5e-5로 warm-restart됨 (실측: loss 추가 하락, 유효했음).
- **resume은 체크포인트 내 `train_config.json`의 `output_dir`에 저장**한다. 런 디렉토리를
  리네임했으면 그 안의 모든 train_config.json의 `output_dir`/`job_name`을 패치할 것.
  (mask1 resume이 mask0 디렉토리를 덮어쓴 사고 있었음 — HF 업로드본으로 복구.)
- **tb_log.py hang(수정됨)**: done 마커가 `with_suffix`로 계산돼 `train.log.done`을 못 봤음.
  이 hang이 train_smolvla.sh의 EXIT trap을 영원히 막아 후속 체인이 안 돌았다.
- **디스크**: 체크포인트 하나 ≈ 1.3G(모델 866M + optimizer 395M). 공유 디스크가 가득 차면
  tee가 ENOSPC로 죽고 트레이너가 **에러 로그 없이** SIGPIPE로 급사한다(진행바 중간에서 끊김).
  대응: `--save_freq=4000` + 완료 후 중간 체크포인트의 `training_state/` 삭제('last' 것만 유지).
- **GPU**: 공유 서버. 시작 전 `nvidia-smi`로 확인. 학습은 GPU-bound가 아니라 dataloader-bound —
  같은 GPU에 여러 런 병렬 가능(런당 ~10-13GB).

## 6.5 dfmag 라운드 (2026-07-16~17 추가)

fz의 baseline이 payload에 따라 변하는 문제(빈 툴 ~13N vs loaded ~16N+) 때문에
**힘 변화율 d|F|/dt (dfmag)** 채널을 추가했다. 접촉은 1–2프레임의 급격한 힘 하락
транз이언트(|F| ~10→0.6N → dfmag ≈ −5 N/frame)라 baseline 이동에 강건하다.

- **파생 데이터셋**: `derive_df_dataset.py`가 기존 데이터셋에서 dfmag(에피소드 내 diff,
  첫 프레임 0)를 state 16번째 차원으로 추가 → HF `Chanho-Lee/lges_case_pick_0708_dF{,_abs_dF}`.
  meta/stats.json + per-episode stats 자동 갱신.
- **코드**: `film_contact.py`에 dfmag 채널(idx 15, `FILM_DFMAG_TAU` 기본 5, mask_force에 포함),
  `train_film.py`/`probe_film_authority.py`는 *_dF 데이터셋이면 stats 자동 로드,
  `run_policy.py` ObsBuilder는 16-dim 체크포인트 자동 감지 후 이전 프레임 버퍼로 dfmag를
  라이브 계산(>0.5s 공백이면 새 롤아웃으로 보고 0 — 학습의 에피소드 첫 프레임 규약과 동일).
- **학습**: `run_dfmag.sh` — prefix+mask1+`FILM_COND=contact,fz,seal,dfmag`, 30k →
  `smolvla_film_0708_dF_prefix_mask1` (rel, loss 0.106) / `_abs_dF_prefix_mask1` (abs, 0.014).
- **probe (전건 강제 all-0↔all-1)**: rel 기준 82% 상쇄 — dfmag 없는 prefix+mask1(67%)보다 높음.
  probe에 `--c0/--c1` per-channel 패턴 옵션 추가.
- **⚠ 채널 분해 probe가 위 수치의 해석을 뒤집음 (probes/decomp_*.txt)**. committed-descent
  상쇄율을 채널별로 분리하면 (c0 = 하강 상태):
  | 강제 패턴 | dF ckpt | 3ch prefix_mask1 |
  |---|---|---|
  | all-1 (기존 수치) | 82% | 67% |
  | **현실적 접촉 순간** (contact=1, fz=0, seal=0, dfmag=−1) | **0%** | **0%** |
  | contact만 1 | 11% | — |
  | dfmag만 −1 | 0% | — |
  | fz 하락만 (0.47→0) | 0% (부호 반대) | — |
  | seal만 1 | 21% | 28% |

  즉 67%/82%는 접촉-순간 시그니처가 아니라 **all-1 조합(≈ sealed+가압+힘상승 상태)**이 만든
  권한이고, 실제 descend→stop 전이 시점의 신호 조합으로는 **어느 체크포인트도 게이팅하지
  못한다**. 단일 채널로는 seal이 가장 강하지만(21–28%) 약하고, 접촉 транз이언트(contact,
  dfmag)는 거의 무권한. 원인 추정: BC 데이터에서 접촉 딥은 에피소드당 1–2프레임뿐이라
  gradient 노출이 극소 + z-바닥과 접촉이 항상 co-occur라 반사실이 없음 → 모델은 더 안정적인
  사후 신호(seal/가압)에 정지를 바인딩. PROGRESS.md의 "깊이 cue 억제 또는 gate-oracle
  rollout 증류" 레버가 여전히 필요하다는 결론.
- **배포 관점**: dF ckpt는 all-1류 상태(씰링 후)에는 강하게 반응하므로 "seal 후 정지/리프트"는
  기대 가능하나, "접촉 즉시 정지"는 probe상 근거 없음. 로봇 평가로 확인 필요.

## 6.6 접촉 전이 oversampling 라운드 (2026-07-20)

분해 probe의 "gradient 노출 부족" 가설을 검증: `train_film.py`에 `FILM_OVERSAMPLE_BOOST`
(WeightedRandomSampler 주입, ≥2N/frame 힘 하락 ±5프레임 = 1,099개 프레임을 10배 가중,
샘플의 ~40%)를 추가하고 dfmag 세팅으로 재학습 →
`smolvla_film_0708_dF_prefix_mask1_os10` (rel, loss 0.076) / `_abs_…_os10` (abs, 0.010).

**결과: 실패 — 노출 가설 기각.** (probes/os10_*.txt)
| 강제 패턴 | os10 | os 없음(6.5) |
|---|---|---|
| all-1 | 69% | 82% |
| **현실적 접촉 순간** | **0% (부호 반대, −1.6mm)** | 0% |
| contact만 | 14% | 11% |
| dfmag만 | 0% | 0% |

**근본 원인 (데이터 분석)**: 75 에피소드의 접촉 z가 10–90퍼센타일 기준 **0.787–0.843m,
~5.6cm 밴드**에 몰려 있음. 깊이(ee_z/이미지) 단서가 정지 시점을 거의 완벽히 예측하므로,
BC는 1–2프레임짜리 노이즈 낀 힘 транз이언트를 쓸 이유가 없다 — 샘플링을 아무리 몰아줘도
전이 프레임 안에서조차 깊이가 co-occur하는 교란은 그대로다. **반사실이 있는 데이터**
(접촉 높이가 실제로 다양한 depletion 상태들에서의 시연/롤아웃)나 **깊이 cue 억제**만이
이 shortcut을 끊을 수 있다.

**남은 레버**: ① depletion 상태(케이스 채움 높이)를 바꿔가며 추가 시연 수집 → 재학습
② gate-oracle rollout 증류(DAgger) ③ 학습 시 ee_z 마스킹/노이즈 (공격적, 부작용 위험).
로봇 평가는 probe(단일 프레임 open-loop)와 다른 폐루프 거동을 보여줄 수 있으므로
dF_prefix_mask1 / os10 체크포인트로 S1 겸 실기 확인 권장.

## 6.7 0721 라운드 — 새 로봇 (2026-07-22~24)

로봇 교체로 힘 분포가 완전히 바뀜: **접촉 = |F| 상승** (`film_contact.py` 부호 반전됨),
hover |F| p50=4.8N / sealed p50=8.4N (구 로봇 ~14N 스케일의 절반 이하), unsealed p90=8.9N으로
**분포가 크게 겹침** → 고정 임계 contact는 약하고 접촉 점프(+2.6N/frame)가 주 신호.
데이터: `Chanho-Lee/lges_case_pick_0721` (58 eps, rel 7d, **layer 1·5 양극단만** — 접촉 z
바이모달 ~0.77/0.82). 캘리브레이션: **FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5** (체크포인트 버퍼에
저장됨). 참고: F0=12 기본값이면 contact 채널이 죽음 — 검증 스크립트가 잡아냄
(run_case_pick_0721.sh의 힘 프로파일 검증 단계).

학습 4종 (GPU 7, 30k, save_freq 4000) + probe (pre-contact 임계 --contact-n 6, 현실 패턴
c0=[0,-3.5,0(,0)] c1=[0.6,-3.1,0(,+0.5)]):

| run | loss | std all-1 | 현실적 접촉 순간 |
|---|---|---|---|
| `smolvla_naive_0721` | 0.057 | — | — |
| `…film_0721_prefix_mask1` | 0.059 | 38% WEAK | 7% WEAK |
| `…film_0721_prefix_mask1_os3` | 0.050 | 35% WEAK | 8% WEAK |
| `…film_0721_dF_prefix_mask1` | 0.055 | 42% WEAK | 8% WEAK |
| `…film_0721_dF_prefix_mask1_os3` | 0.051 | **59% PASS** | **10% WEAK** |

**2×2 분해 (std 상쇄율)**: 3ch 38% / 3ch+os 35% / dF 42% / **dF+os 59%** —
oversampling 단독(3ch+os)은 효과 없음; dfmag 단독도 +4%p뿐. **둘의 조합만 시너지**
(+17~21%p). 해석: 전이 프레임을 아무리 노출해도 транз이언트를 표현할 채널(dfmag)이
없으면 배울 수 없고, dfmag가 있어도 노출이 부족하면 안 쓴다.

**해석**: 0708 대비 두 가지 진전 — ① 현실적 접촉-순간 패턴이 처음으로 **올바른 부호의
0이 아닌** 상쇄(7–10%)를 보임 (0708은 전부 0%/역부호) → 바이모달 접촉 높이가 depth
shortcut을 일부 깨는 데 실제로 기여. ② oversampling(os3)이 처음으로 일관된 우위
(std 59% PASS, 현실 10%). 순서: os3 > dF > 3ch. 다만 여전히 depth 지배가 크므로
**결정적 판정은 로봇에서 학습에 없는 중간 layer(2–4) 보간 테스트**.
전이 oversampling은 새 로봇에서 |Δ|F||≥2 프레임이 16%나 돼(노이즈) boost 3(≈36% 샘플)으로
조정함 — boost 10이면 65%로 과함.

로봇 평가 커맨드 (권장 1순위 os3). **⚠ 0721 세대는 _contact_F0/_contact_tau/_fz_tau가
`persistent=False`라 체크포인트에 저장되지 않는다 — 아래 FILM_F0/FILM_TAU/FILM_FZ_TAU를
배포 시 반드시 명시할 것 (빼먹으면 기본 12/10/30이 적용돼 contact 채널이 죽는다).**
dfmag_tau(5)와 wrench/seal/dfmag 통계는 persistent 버퍼라 자동 로드된다. fz의 −20 오프셋은
코드 리터럴이므로 0721 체크포인트는 현재 코드(−20 포함)와만 호환 (0708 FiLM ckpt는
−20 없던 코드로 학습됨 — 현 코드로 배포 금지):
```bash

python run_policy.py --checkpoint Chanho-Lee/smolvla_naive_0729 \
--go --force-limit 15 --n-action-steps 5 --log-dir rollouts/smolvla_naive_0729

FILM_COND=contact,fz,seal FILM_INJECT=suffix FILM_MASK_FORCE=1 \
FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5 FILM_FZ_OFF=2.1 \
FILM_DATASET=/home/dexmate/LGES/Dexmate/LGES/vla_training/local_film_stats/lges_case_pick_0729 \
python run_policy.py --film --checkpoint Chanho-Lee/smolvla_film_0729_suffix_mask1 \
--go --force-limit 15 --n-action-steps 5 --log-dir rollouts/smolvla_film_0729

FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 FILM_FZ_OFF=1.8 FILM_DATASET=/home/dexmate/LGES/Dexmate/LGES/vla_training/local_film_stats/lges_case_pick_0721_0727 python run_policy.py --film --checkpoint Chanho-Lee/smolvla_film_0721_0727_prefix_mask1 --go --force-limit 15 --n-action-steps 2 --log-dir rollouts/film_0721_0727_nas2

FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5 FILM_FZ_OFF=2.6 \
FILM_DATASET=/home/dexmate/LGES/Dexmate/LGES/vla_training/local_film_stats/lges_case_pick_0721_0727 \
python probe_film_authority_live.py --go \
  --clearances 0.05 0.04 0.03 0.02 0.01 0.00 -0.01 -0.02 -0.03 -0.04\
  --checkpoint Chanho-Lee/smolvla_film_0721_0727_prefix_mask1 \
  --fz-deltas-n -6 -3 3 6

# 베이스라인: smolvla_naive_0721 (--film 없이)
```

## 6.8 0721_0727 라운드 — 데이터 2배 + 멀티모델 비교 (2026-07-28~29)

데이터: `Chanho-Lee/lges_case_pick_0721_0727` (116 eps / 24,648 frames, rel 7d, layer 1·5).
fz 오프셋을 데이터 중앙값(2.6)으로 변경하고 **현세대 캘리브레이션을 코드 기본값으로 고정**
(F0=6 tau=4 fz_tau=5 fz_off=2.6 — env 없이 배포 가능; 구세대는 env 필수, §6.7 참조).
스토리지는 /data 볼륨으로 이사 (`/data/home/maverick_data`, 심링크 경유 — /home 포화 재발 방지).

전 런 50k (30k에서 warm-restart 연장, save_freq=10000, 종료 후 last만 유지):

| run | loss@50k | probe std | probe 현실 접촉 |
|---|---|---|---|
| `smolvla_naive_0721_0727` | 0.076 | — | — |
| `…prefix_mask1` | 0.074 | **61% PASS** | **24% WEAK** |
| `…prefix_mask1_os3` | 0.068 | 60% PASS | 21% WEAK |
| `…suffix_mask1` | 0.062 | 51% PASS | 16% WEAK |
| `xvla_0721_0727` | 0.001* | (probe는 SmolVLA 전용) | |
| `pi05_naive_0721_0727` / `act_0721_0727` | 완료 후 기입 | | |

(*loss 정의 상이 — 모델 간 loss 직접 비교 불가; 판단은 로봇 평가로)

**관찰**: ① 30k→50k 연장은 loss만 소폭 개선, c-hat 권한은 불변 (62/25→61/24) — 권한은
데이터가 결정하고 스텝으로는 안 오름. ② prefix > suffix 재확인 (61/24 vs 51/16).
③ os3 이득 소멸 (데이터 2배로 자연 전이 노출 충분). ④ 현실-접촉 권한은 ~25%에서 plateau —
남은 레버는 접촉 높이 다양화(중간 layer 데이터) 또는 gate-oracle 증류.
⑤ 새 베이스라인 3종 추가: pi0.5(train_pi05.py shim), X-VLA(train_xvla.py shims 4종),
ACT. FiLM-pi0.5 포팅(film_contact_pi05.py, suffix 전용, 스모크 통과) 준비됨.

**로봇 평가 1순위**: `smolvla_film_0721_0727_prefix_mask1` (env 불필요, 코드 기본값 일치)
+ 베이스라인 `smolvla_naive_0721_0727`, 여유 되면 pi05/ACT/XVLA 순.
**핵심 테스트: 학습에 없는 중간 layer(2–4) 보간.**

## 7. 앞으로 진행할 실험 (PROGRESS.md S1–S5 매핑)

| 우선순위 | 실험 | 내용 | 필요 자원 |
|---|---|---|---|
| 1 | **S2: V1 decorrelated control** | 이긴 세팅(prefix+mask1)으로 `FILM_VARIANT=v1`(배치 셔플 c-hat) 30k 학습 → probe에서 V2 ≫ V1이면 grounding 입증 | GPU만, ~2.5h |
| 2 | **S1: 로봇 depletion sweep** | prefix_mask1 vs naive vs gate-oracle, `lift_condition_probe.py --by-height`로 under-reach 회복/과압 확인 | 로봇 |
| 3 | **S3: 온로봇 반사실** | 실기에서 c-hat 강제 0/1 (오프라인 probe는 이미 PASS) | 로봇 |
| 4 | **S4: 저데이터 곡선** | 15/30/50/75 eps 서브셋 재학습, FiLM 이득 vs 데이터양 | GPU, 데이터 서브셋 생성 |
| 5 | **S5: negative control** | 관측 불가능 조건(정렬/삽입)에서 c-hat 무익 확인 | 삽입 태스크 데이터 수집 |
| 옵션 | film_on_naive를 prefix+mask1로 | naive 30k ckpt에 사후 FiLM 부착이 되는지 (배포 워크플로 가치) | GPU, ~1h |
| 옵션 | abs용 probe 보강 | phase 판정을 절대 액션에 맞게 수정 → abs 체크포인트 공식 판정 | 코드만 |

주의: V1 학습·평가 역시 cond/inject/mask_force 구조는 V2와 동일하게 맞출 것 (`FILM_VARIANT`만 v1).
