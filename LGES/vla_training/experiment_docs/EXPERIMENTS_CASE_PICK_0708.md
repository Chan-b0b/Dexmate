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
