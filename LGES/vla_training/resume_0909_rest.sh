#!/usr/bin/env bash
# Restart the six 0909 arms that died when the SSH session dropped (2026-09-09 20:49:34,
# sshd "Disconnected from user maverick"). run_case_pick_0909_all.sh had been started in
# the foreground, so SIGHUP took the orchestrator and all its chains; only the two arms
# that had been re-launched under nohup survived. The three pi0 arms had already finished
# (20k + select_best + battery), so only groot (died at 5k/50k) and SmolVLA (20k/50k) need
# resuming — from their last checkpoints, which all carry training_state/.
#
# Everything runs under setsid + nohup so a disconnect cannot repeat this.
# GPUs 2/3/4 only (0 and 1 must stay free, per the user's 2026-09-09 request); one groot
# + one SmolVLA per GPU. num_workers is deliberately NOT overridden — CPU capacity freed
# by the finished pi0 arms is enough, and keeping the recipe byte-identical preserves the
# round's comparability.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/logs"

#     name                      gpu kind        inject entry
ARMS=(
  "groot_naive_0909            2  naive       -      train_groot.py"
  "smolvla_naive_0909          2  naive       -      -m lerobot.scripts.lerobot_train"
  "groot_film_state_0909       3  film-groot  state  train_film_groot.py"
  "smolvla_film_state_0909     3  film        prefix train_film.py"
  "groot_film_layers_0909      4  film-groot  layers train_film_groot.py"
  "smolvla_film_layers_0909    4  film        layers train_film.py"
)
for a in "${ARMS[@]}"; do
  read -r name gpu kind inject entry <<<"$a"
  [[ -f "$DIR/outputs/$name/checkpoints/last/pretrained_model/train_config.json" ]] \
    || { echo "[rest] $name has no resumable checkpoint" >&2; exit 1; }
  setsid nohup "$DIR/resume_0909_arm.sh" "$name" "$gpu" "$kind" "$inject" "$entry" \
    > "$DIR/logs/resume_${name}.out" 2>&1 &
  echo "[rest] $name -> GPU$gpu (from $(readlink "$DIR/outputs/$name/checkpoints/last")) pid=$!"
  sleep 2
done
echo "[rest] 6 arms launched detached; watch logs/resume_*_0909.out"
