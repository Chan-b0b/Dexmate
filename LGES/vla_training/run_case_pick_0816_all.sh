#!/usr/bin/env bash
# 0816 round (2026-08-16): naive / film-state / film-layers × SmolVLA / ACT / π0 / GR00T
# = 12 arms on the NEW Chanho-Lee/lges_case_pick_0816 dataset (head/head_depth cameras,
# state 15, action 7; FZ_MEDIAN=1.4 from logs/validate_0816.out; hover 5.6N < F0=6 <
# press 7.3N so the 0729 F0/TAU stay valid).
# Recipes match the 0729 rounds exactly except dataset + FILM_FZ_OFF; smolvla film-state
# = inject 'prefix' (its state-token point), pi0/groot/act film-state = inject 'state'.
# Per arm: train -> select_best (val-best, --prune) -> full probe battery
# (run_battery_0729cal.sh with VAL_ROOT/FZ_OFF/PFX=0816 overrides).
#
# GPU map (all 8 allowed this round; GPU4 shared with the still-running 0729 pi0-layers):
#   0 pi0_naive | 1 pi0_film_state | 2 pi0_film_layers
#   3 groot_naive + groot_film_state | 5 groot_film_layers + act_naive
#   6 act_film_state + act_film_layers | 7 smolvla x3
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0816
REPO_VAL_ID=Chanho-Lee/lges_case_pick_0816_val
RT="$DIR/datasets/lges_case_pick_0816"
RV="$DIR/datasets/lges_case_pick_0816_val"
FZ=1.4
RENAME_PI0='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
FB=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FZ"
    FILM_DATASET_ROOT="$RT")

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[0816] datasets missing under $DIR/datasets" >&2; exit 1; }
mkdir -p "$DIR/logs" "$DIR/probes"
BASE_SMOLVLA="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"

# chain: train -> select_best -> battery, all pinned to one GPU. Usage:
#   chain <run_name> <gpu> <battery_kind> <inject|-> -- <train cmd...>
chain() {
  local name=$1 gpu=$2 kind=$3 inject=$4; shift 5   # 5th arg is the '--' separator
  "$@" > "$DIR/logs/orch_${name}.out" 2>&1
  local rc=$?
  echo "[0816] $name train rc=$rc"
  [[ $rc == 0 ]] || return $rc
  env CUDA_VISIBLE_DEVICES="$gpu" "${FB[@]}" FILM_INJECT="${inject/-/state}" \
    "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$name" \
    --val-root "$RV" --repo-id "$REPO_VAL_ID" --prune >> "$DIR/logs/best_0816.out" 2>&1
  echo "[0816] best($name): $(readlink "$DIR/outputs/$name/checkpoints/best" 2>/dev/null)"
  VAL_ROOT="$RV" REPO_VAL="$REPO_VAL_ID" FZ_OFF="$FZ" PFX=0816 \
    "$DIR/run_battery_0729cal.sh" "$name" "$kind" "$inject" \
    "$DIR/outputs/$name/checkpoints/best" "$gpu" >> "$DIR/logs/battery_0816.out" 2>&1
  echo "[0816] battery($name) rc=$?"
}

# ---- π0 (GPUs 0/1/2, bs8 50k save10k, grad-ckpt, pi0 rename map) ----------------
pi0_train() { # gpu extra_env... -- script args...
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" "$@" \
    --policy.path=lerobot/pi0_base --policy.device=cuda --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME_PI0" \
    --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16
}
chain pi0_naive_0816       0 naive     -      -- pi0_train 0 "$PY" -m lerobot.scripts.lerobot_train \
  --output_dir="$DIR/outputs/pi0_naive_0816" --job_name=pi0_naive_0816 & P01=$!
chain pi0_film_state_0816  1 film-pi0  state  -- pi0_train 1 FILM_INJECT=state "$PY" "$DIR/train_film_pi0.py" \
  --output_dir="$DIR/outputs/pi0_film_state_0816" --job_name=pi0_film_state_0816 & P02=$!
chain pi0_film_layers_0816 2 film-pi0  layers -- pi0_train 2 FILM_INJECT=layers "$PY" "$DIR/train_film_pi0.py" \
  --output_dir="$DIR/outputs/pi0_film_layers_0816" --job_name=pi0_film_layers_0816 & P03=$!

# ---- GR00T (GPUs 3/3/5, bs8 50k save10k) ----------------------------------------
groot_train() { # gpu extra_env... -- script args...
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" "$@" \
    --policy.type=groot --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id="$REPO" --dataset.root="$RT" \
    --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16
}
chain groot_naive_0816       3 naive      -      -- groot_train 3 "$PY" "$DIR/train_groot.py" \
  --output_dir="$DIR/outputs/groot_naive_0816" --job_name=groot_naive_0816 & P04=$!
chain groot_film_state_0816  3 film-groot state  -- groot_train 3 FILM_INJECT=state "$PY" "$DIR/train_film_groot.py" \
  --output_dir="$DIR/outputs/groot_film_state_0816" --job_name=groot_film_state_0816 & P05=$!
chain groot_film_layers_0816 5 film-groot layers -- groot_train 5 FILM_INJECT=layers "$PY" "$DIR/train_film_groot.py" \
  --output_dir="$DIR/outputs/groot_film_layers_0816" --job_name=groot_film_layers_0816 & P06=$!

# ---- ACT (GPUs 5/6/6, bs32 50k save5k, from scratch) -----------------------------
act_train() { # gpu extra_env... -- script args...
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" "$@" \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id="$REPO" --dataset.root="$RT" \
    --batch_size=32 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=16
}
chain act_naive_0816       5 naive-act -      -- act_train 5 "$PY" -m lerobot.scripts.lerobot_train \
  --output_dir="$DIR/outputs/act_naive_0816" --job_name=act_naive_0816 & P07=$!
chain act_film_state_0816  6 film-act  state  -- act_train 6 FILM_INJECT=state "$PY" "$DIR/train_film_act.py" \
  --output_dir="$DIR/outputs/act_film_state_0816" --job_name=act_film_state_0816 & P08=$!
chain act_film_layers_0816 6 film-act  layers -- act_train 6 FILM_INJECT=layers "$PY" "$DIR/train_film_act.py" \
  --output_dir="$DIR/outputs/act_film_layers_0816" --job_name=act_film_layers_0816 & P09=$!

# ---- SmolVLA (GPU 7 x3, bs32 50k save5k, from smolvla_base) ----------------------
# train_smolvla.sh / train_film.sh defaults already rename head->camera1,
# head_depth->camera2 — exactly this dataset's schema.
chain smolvla_naive_0816 7 naive - -- env CUDA_VISIBLE_DEVICES=7 \
  HF_DATASET_REPO="$REPO" HF_CACHE_DIR="$RT" RUN_NAME=smolvla_naive_0816 \
  "$DIR/train_smolvla.sh" --steps=50000 --save_freq=5000 --num_workers=16 \
  --policy.scheduler_decay_steps=50000 & P10=$!
# same input_features as train_smolvla.sh's default (camera3 declared-but-absent is the
# established lges_suction-schema convention) so all three smolvla arms match exactly.
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
smolvla_film() { # inject run_name
  env CUDA_VISIBLE_DEVICES=7 "${FB[@]}" FILM_INJECT="$1" RUN_NAME="$2" \
    INIT_CKPT="$BASE_SMOLVLA" DATASET_REPO="$REPO" DATASET_ROOT="$RT" NUM_WORKERS=16 \
    "$DIR/train_film.sh" --policy.input_features="$IF15" \
    --steps=50000 --save_freq=5000 --policy.scheduler_decay_steps=50000
}
chain smolvla_film_state_0816  7 film prefix -- smolvla_film prefix smolvla_film_state_0816 & P11=$!
chain smolvla_film_layers_0816 7 film layers -- smolvla_film layers smolvla_film_layers_0816 & P12=$!

echo "[0816] launched 12 arms: pi0=$P01/$P02/$P03 groot=$P04/$P05/$P06 act=$P07/$P08/$P09 smolvla=$P10/$P11/$P12"
FAIL=0
for p in $P01 $P02 $P03 $P04 $P05 $P06 $P07 $P08 $P09 $P10 $P11 $P12; do
  wait "$p" || FAIL=$((FAIL + 1))
done
echo "[0816] DONE (failures: $FAIL)"
[[ $FAIL == 0 ]]
