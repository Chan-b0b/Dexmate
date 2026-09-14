#!/usr/bin/env bash
# 0909 mask ablation (2026-09-13): film-layers x pi0 / GR00T / SmolVLA with
# FILM_MASK_FORCE=0, i.e. the raw wrench stays visible in observation.state while c-hat
# also feeds the FiLM branch. Everything else is byte-identical to the 0909 round
# (same dataset, same derived calibration, same steps/bs/workers, same chain).
#
# Why: the 0909 arms all ran mask=1, which is what makes dRaw = +0.00 (EXPERIMENTS §7) —
# the single-path bottleneck that licenses the causal claim. Dropping the mask opens a
# second path and asks whether the FiLM gate earns anything over just handing the policy
# the force in its state vector. §8 makes this sharper: c-hat's 'contact' channel
# saturates at 18N while raw |F| does not, so mask=0 is also the only arm family that can
# see force above the c-hat window.
#
# Expect dRaw != 0 here. That is the point, not a regression.
#
# 'layers' is the chosen inject: the only 0909 variant that passed both press-sim timings
# (groot_film_layers 10/10 at off1 AND off30), so it is the strongest baseline to ablate.
#
# GPU map (0/1/3 are the free ones as of 2026-09-13; 2 is partly used, 4-7 hold another
# user's pi0/pi05 jobs):
#   0 pi0_film_layers_nomask | 1 groot_film_layers_nomask | 3 smolvla_film_layers_nomask
#
# SmolVLA runs 15k, not 50k. All three mask=1 SmolVLA arms picked best=005000 and their
# val_loss rises monotonically from there (film_layers 0.131@5k -> 0.343@50k, 2.6x), so
# steps past ~15k only produce checkpoints select_best will never choose. save_freq stays
# 5000 so the candidate grid (5k/10k/15k) is aligned with the mask=1 round's grid — those
# runs were --prune'd down to best+last, so they cannot be re-scored on a finer grid.
# GR00T keeps the full 50k: its best landed at 40k/45k/50k, and while that curve is flat
# enough (+-8% jitter) that the late minimum is partly luck, the paper's headline arm
# (groot_film_layers, press-sim 10/10 at both offsets) was measured at 45k — matching the
# budget is what keeps "mask" the only variable.
#
# ONLY=pi0|groot|smolvla|all  runs a subset (default all), for restarting one arm.
#
# Prereq: ./validate_0909.sh must have passed (it writes logs/validate_0909.out).
set -uo pipefail
ONLY=${ONLY:-all}
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

# ---- 1. gate: same derived calibration as the mask=1 round ------------------------
V="$DIR/logs/validate_0909.out"
grep -q '\[0909\] validation OK' "$V" 2>/dev/null \
  || { echo "[nomask] run ./validate_0909.sh first (no 'validation OK' in $V)" >&2; exit 1; }
cal() { grep -o "^CAL_$1=.*" "$V" | tail -1 | cut -d= -f2-; }
F0="$(cal F0)"; TAU="$(cal TAU)"; FZ="$(cal FZ_OFF)"; FZT="$(cal FZ_TAU)"; C1="$(cal C1)"
FZD="$(cal FZ_DELTA0)"; FBASE="$(cal F_BASE)"; FCAP="$(cal F_CAP)"; PRECON="$(cal PRE_CONTACT)"; FCFZ="$(cal FC_FZ)"
for v in F0 TAU FZ FZT C1 FZD FBASE FCAP PRECON; do
  [[ -n "${!v}" ]] || { echo "[nomask] could not read CAL_$v from $V — re-run validate_0909.sh" >&2; exit 1; }
done
echo "[nomask] calibration: F0=$F0 TAU=$TAU FZ_OFF=$FZ FZ_TAU=$FZT C1=$C1  (MASK_FORCE=0)"

# The one flag that differs from run_case_pick_0909_all.sh.
FB=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=0
    FILM_F0="$F0" FILM_TAU="$TAU" FILM_FZ_TAU="$FZT" FILM_FZ_OFF="$FZ"
    FILM_DATASET_ROOT="$RT")
mkdir -p "$DIR/logs" "$DIR/probes"

BASE_SMOLVLA="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"

# chain: train -> select_best (val-best, --prune) -> full probe battery, one GPU each.
# CONTACT_N="$F0" fixes the P1 cell: run_battery_0729cal.sh defaults --contact-n 6, but
# 0909's pre-touch |F| never drops below 6.55N, so the 'frames below contact' filter
# matched 0/107 frames and the mask=1 round's Tier-1 tables came out empty at rc=0
# (EXPERIMENTS §9.2 territory). contact_n == F0 is the 0729/0816 invariant.
chain() { # <run_name> <gpu> <battery_kind> <inject> -- <train cmd...>
  local name=$1 gpu=$2 kind=$3 inject=$4; shift 5
  "$@" > "$DIR/logs/orch_${name}.out" 2>&1
  local rc=$?
  echo "[nomask] $name train rc=$rc"
  [[ $rc == 0 ]] || return $rc
  env CUDA_VISIBLE_DEVICES="$gpu" "${FB[@]}" FILM_INJECT="${inject/-/state}" \
    "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$name" \
    --val-root "$RV" --repo-id "$REPO_VAL_ID" --prune >> "$DIR/logs/best_nomask.out" 2>&1
  echo "[nomask] best($name): $(readlink "$DIR/outputs/$name/checkpoints/best" 2>/dev/null)"
  VAL_ROOT="$RV" REPO_VAL="$REPO_VAL_ID" FZ_OFF="$FZ" FZ_TAU="$FZT" F0="$F0" TAU="$TAU" C1="$C1" \
    FZ_DELTA0="$FZD" F_BASE="$FBASE" F_CAP="$FCAP" PRE_CONTACT="$PRECON" FC_FZ="$FCFZ" \
    CONTACT_N="$F0" PFX=0909nm FILM_MASK_FORCE=0 \
    "$DIR/run_battery_0729cal.sh" "$name" "$kind" "$inject" \
    "$DIR/outputs/$name/checkpoints/best" "$gpu" >> "$DIR/logs/battery_nomask.out" 2>&1
  echo "[nomask] battery($name) rc=$?"
}

# ---- 2. pi0 (GPU 0, bs8 20k save5k, grad-ckpt, pi0 rename map) -------------------
pi0_train() { # gpu extra_env... -- entry args...
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" "$@" \
    --policy.path=lerobot/pi0_base --policy.device=cuda --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME_PI0" \
    --batch_size=8 --steps=20000 --save_freq=5000 --log_freq=100 --num_workers=24
}
P1=
if [[ $ONLY == all || $ONLY == pi0 ]]; then
chain pi0_film_layers_nomask_0909 0 film-pi0 layers -- pi0_train 0 FILM_INJECT=layers \
  "$PY" "$DIR/train_film_pi0.py" \
  --output_dir="$DIR/outputs/pi0_film_layers_nomask_0909" \
  --job_name=pi0_film_layers_nomask_0909 & P1=$!
fi

# ---- 3. GR00T (GPU 1, bs8 50k save5k) --------------------------------------------
groot_train() { # gpu extra_env... -- entry args...
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" "$@" \
    --policy.type=groot --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id="$REPO" --dataset.root="$RT" \
    --batch_size=8 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=24
}
P2=
if [[ $ONLY == all || $ONLY == groot ]]; then
chain groot_film_layers_nomask_0909 1 film-groot layers -- groot_train 1 FILM_INJECT=layers \
  "$PY" "$DIR/train_film_groot.py" \
  --output_dir="$DIR/outputs/groot_film_layers_nomask_0909" \
  --job_name=groot_film_layers_nomask_0909 & P2=$!
fi

# ---- 4. SmolVLA (GPU 3, bs32 15k save5k, from smolvla_base — see header) ---------
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
smolvla_film() { # inject run_name
  env CUDA_VISIBLE_DEVICES=3 "${FB[@]}" FILM_INJECT="$1" RUN_NAME="$2" \
    INIT_CKPT="$BASE_SMOLVLA" DATASET_REPO="$REPO" DATASET_ROOT="$RT" NUM_WORKERS=32 \
    "$DIR/train_film.sh" --policy.input_features="$IF15" \
    --steps=15000 --save_freq=5000 --policy.scheduler_decay_steps=15000
}
P3=
if [[ $ONLY == all || $ONLY == smolvla ]]; then
chain smolvla_film_layers_nomask_0909 3 film layers -- \
  smolvla_film layers smolvla_film_layers_nomask_0909 & P3=$!
fi

echo "[nomask] launched (ONLY=$ONLY): pi0=$P1 groot=$P2 smolvla=$P3"
FAIL=0
for p in $P1 $P2 $P3; do
  wait "$p" || FAIL=$((FAIL + 1))
done
echo "[nomask] DONE (failures: $FAIL)"
[[ $FAIL == 0 ]]
