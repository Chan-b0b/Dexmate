#!/usr/bin/env bash
# 0909 round (2026-09-09): naive / film-state / film-layers x pi0 / GR00T / SmolVLA
# = 9 arms on Chanho-Lee/lges_case_pick_0909 (head/head_depth, state 15, action 7).
#
# Narrowed from 0816's 12+3 arms by that round's press-sim result: ACT (5/1/0 of 20 at
# off1, 0/20 at off30) and pi0.5 (0/20 everywhere — no state token to condition) are
# dropped. Kept: GR00T (20/20 both offsets; FiLM buys depth, 5.2 -> 1.1mm at off30),
# pi0 (FiLM-state creates the authority naive lacks, 2/20 -> 10/20 at off1) and SmolVLA
# (0/20 -> 9-10/20), i.e. every arm that showed signal.
#
# pi0 runs 20k, not 50k: the r2 val curves rise monotonically past 10k
# (0.122@10k -> 0.253@50k) and best landed on 010000 in all three pi0 arms.
# GR00T stays 50k (val flat 0.036-0.044 across the whole run, best at 45k).
#
# Recipes are otherwise identical to run_case_pick_0816_all.sh; only the dataset, the
# FiLM calibration (F0/TAU/FZ_OFF/FZ_TAU/C1, all read from the validate gate rather than
# hardcoded) and num_workers differ.
# Workers are budgeted against nproc=256 across 9 concurrent arms: 24+24+32 per family.
#
# GPU map (0-4, per this round's allowance; 5/6/7 hold another user's jobs):
#   0 pi0_naive | 1 pi0_film_state | 2 pi0_film_layers
#   3 groot_naive + groot_film_state
#   4 groot_film_layers + smolvla x3
#
# Prereq: ./validate_0909.sh must have passed (it writes logs/validate_0909.out).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0909
REPO_VAL_ID=Chanho-Lee/lges_case_pick_0909_val
RT="$DIR/datasets/lges_case_pick_0909"
RV="$DIR/datasets/lges_case_pick_0909_val"
RENAME_PI0='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'

# ---- 1. gate: the validate step must have passed; take the calibration from it -----
# The whole calibration is DERIVED by validate_0909.sh, never written here — this
# collection's sensor remount moved the pre-touch |F| baseline +5.4N (5.12 -> 10.51N),
# so 0816's F0=6 would sit BELOW hover: the contact channel would read ~0.8 while
# hovering and saturate at contact, discriminating nothing.
V="$DIR/logs/validate_0909.out"
grep -q '\[0909\] validation OK' "$V" 2>/dev/null \
  || { echo "[0909] run ./validate_0909.sh first (no 'validation OK' in $V)" >&2; exit 1; }
cal() { grep -o "^CAL_$1=.*" "$V" | tail -1 | cut -d= -f2-; }
F0="$(cal F0)"; TAU="$(cal TAU)"; FZ="$(cal FZ_OFF)"; FZT="$(cal FZ_TAU)"; C1="$(cal C1)"
FZD="$(cal FZ_DELTA0)"; FBASE="$(cal F_BASE)"; FCAP="$(cal F_CAP)"; PRECON="$(cal PRE_CONTACT)"; FCFZ="$(cal FC_FZ)"
for v in F0 TAU FZ FZT C1 FZD FBASE FCAP PRECON; do
  [[ -n "${!v}" ]] || { echo "[0909] could not read CAL_$v from $V — re-run validate_0909.sh" >&2; exit 1; }
done
echo "[0909] calibration: F0=$F0 TAU=$TAU FZ_OFF=$FZ FZ_TAU=$FZT C1=$C1"
echo "[0909] probe model : fz_delta0=$FZD f_base=$FBASE f_cap=$FCAP pre_contact=$PRECON"
FB=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1
    FILM_F0="$F0" FILM_TAU="$TAU" FILM_FZ_TAU="$FZT" FILM_FZ_OFF="$FZ"
    FILM_DATASET_ROOT="$RT")
mkdir -p "$DIR/logs" "$DIR/probes"

# ---- 2. smoke: 4 steps of GR00T, to prove this dataset loads ----------------------
# 0816's root came from snapshot_download (it has .gitattributes/README.md); 0909 was
# rsynced in, so --dataset.repo_id has no hub counterpart. If lerobot still reaches for
# the hub, it fails HERE (2 min) instead of 9 arms deep.
CUDA_VISIBLE_DEVICES=0 env "${FB[@]}" FILM_INJECT=state "$PY" "$DIR/train_groot.py" \
  --policy.type=groot --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=2 --steps=4 --save_freq=4 --log_freq=1 --num_workers=4 \
  --output_dir="$DIR/outputs/smoke_0909" --job_name=smoke_0909 \
  > "$DIR/logs/smoke_0909.out" 2>&1
[[ $? == 0 ]] || { echo "[0909] SMOKE FAILED — see logs/smoke_0909.out" >&2; exit 1; }
rm -rf "$DIR/outputs/smoke_0909"
echo "[0909] smoke OK"

BASE_SMOLVLA="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"

# chain: train -> select_best (val-best, --prune) -> full probe battery, one GPU each.
#   chain <run_name> <gpu> <battery_kind> <inject|-> -- <train cmd...>
chain() {
  local name=$1 gpu=$2 kind=$3 inject=$4; shift 5   # 5th arg is the '--' separator
  "$@" > "$DIR/logs/orch_${name}.out" 2>&1
  local rc=$?
  echo "[0909] $name train rc=$rc"
  [[ $rc == 0 ]] || return $rc
  env CUDA_VISIBLE_DEVICES="$gpu" "${FB[@]}" FILM_INJECT="${inject/-/state}" \
    "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$name" \
    --val-root "$RV" --repo-id "$REPO_VAL_ID" --prune >> "$DIR/logs/best_0909.out" 2>&1
  echo "[0909] best($name): $(readlink "$DIR/outputs/$name/checkpoints/best" 2>/dev/null)"
  # F0/TAU/C1 must match training exactly — probing at another round's calibration is
  # the mismatch run_battery_0729cal.sh's header forbids (and EXPERIMENTS_PI0_FILM_0729
  # §6.2 records happening to the pi0.5 eval cells).
  VAL_ROOT="$RV" REPO_VAL="$REPO_VAL_ID" FZ_OFF="$FZ" FZ_TAU="$FZT" F0="$F0" TAU="$TAU" C1="$C1" \
    FZ_DELTA0="$FZD" F_BASE="$FBASE" F_CAP="$FCAP" PRE_CONTACT="$PRECON" FC_FZ="$FCFZ" PFX=0909 \
    "$DIR/run_battery_0729cal.sh" "$name" "$kind" "$inject" \
    "$DIR/outputs/$name/checkpoints/best" "$gpu" >> "$DIR/logs/battery_0909.out" 2>&1
  echo "[0909] battery($name) rc=$?"
}

# ---- 3. pi0 (GPUs 0/1/2, bs8 20k save5k, grad-ckpt, pi0 rename map) --------------
# train_pi05.py for the naive arm too — it carries the relative_actions_processor shim
# lerobot 0.5.1 needs for pi0 (the fix the r2 round adopted).
pi0_train() { # gpu extra_env... -- entry args...
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" "$@" \
    --policy.path=lerobot/pi0_base --policy.device=cuda --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME_PI0" \
    --batch_size=8 --steps=20000 --save_freq=5000 --log_freq=100 --num_workers=24
}
chain pi0_naive_0909       0 naive     -      -- pi0_train 0 FILM_INJECT=state  "$PY" "$DIR/train_pi05.py" \
  --output_dir="$DIR/outputs/pi0_naive_0909" --job_name=pi0_naive_0909 & P1=$!
chain pi0_film_state_0909  1 film-pi0  state  -- pi0_train 1 FILM_INJECT=state  "$PY" "$DIR/train_film_pi0.py" \
  --output_dir="$DIR/outputs/pi0_film_state_0909" --job_name=pi0_film_state_0909 & P2=$!
chain pi0_film_layers_0909 2 film-pi0  layers -- pi0_train 2 FILM_INJECT=layers "$PY" "$DIR/train_film_pi0.py" \
  --output_dir="$DIR/outputs/pi0_film_layers_0909" --job_name=pi0_film_layers_0909 & P3=$!

# ---- 4. GR00T (GPUs 3/3/4, bs8 50k save5k) ---------------------------------------
groot_train() { # gpu extra_env... -- entry args...
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" "$@" \
    --policy.type=groot --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id="$REPO" --dataset.root="$RT" \
    --batch_size=8 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=24
}
chain groot_naive_0909       3 naive      -      -- groot_train 3 FILM_INJECT=state  "$PY" "$DIR/train_groot.py" \
  --output_dir="$DIR/outputs/groot_naive_0909" --job_name=groot_naive_0909 & P4=$!
chain groot_film_state_0909  3 film-groot state  -- groot_train 3 FILM_INJECT=state  "$PY" "$DIR/train_film_groot.py" \
  --output_dir="$DIR/outputs/groot_film_state_0909" --job_name=groot_film_state_0909 & P5=$!
chain groot_film_layers_0909 4 film-groot layers -- groot_train 4 FILM_INJECT=layers "$PY" "$DIR/train_film_groot.py" \
  --output_dir="$DIR/outputs/groot_film_layers_0909" --job_name=groot_film_layers_0909 & P6=$!

# ---- 5. SmolVLA (GPU 4 x3, bs32 50k save5k, from smolvla_base) -------------------
# train_smolvla.sh / train_film.sh already rename head->camera1, head_depth->camera2.
chain smolvla_naive_0909 4 naive - -- env CUDA_VISIBLE_DEVICES=4 \
  HF_DATASET_REPO="$REPO" HF_CACHE_DIR="$RT" RUN_NAME=smolvla_naive_0909 \
  "$DIR/train_smolvla.sh" --steps=50000 --save_freq=5000 --num_workers=32 \
  --policy.scheduler_decay_steps=50000 & P7=$!
# camera3 is declared-but-absent by the lges_suction-schema convention train_smolvla.sh
# defaults to — kept so all three smolvla arms share one input_features spec.
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
smolvla_film() { # inject run_name
  env CUDA_VISIBLE_DEVICES=4 "${FB[@]}" FILM_INJECT="$1" RUN_NAME="$2" \
    INIT_CKPT="$BASE_SMOLVLA" DATASET_REPO="$REPO" DATASET_ROOT="$RT" NUM_WORKERS=32 \
    "$DIR/train_film.sh" --policy.input_features="$IF15" \
    --steps=50000 --save_freq=5000 --policy.scheduler_decay_steps=50000
}
chain smolvla_film_state_0909  4 film prefix -- smolvla_film prefix smolvla_film_state_0909 & P8=$!
chain smolvla_film_layers_0909 4 film layers -- smolvla_film layers smolvla_film_layers_0909 & P9=$!

echo "[0909] launched 9 arms: pi0=$P1/$P2/$P3 groot=$P4/$P5/$P6 smolvla=$P7/$P8/$P9"
FAIL=0
for p in $P1 $P2 $P3 $P4 $P5 $P6 $P7 $P8 $P9; do
  wait "$p" || FAIL=$((FAIL + 1))
done
echo "[0909] DONE (failures: $FAIL)"
[[ $FAIL == 0 ]]
