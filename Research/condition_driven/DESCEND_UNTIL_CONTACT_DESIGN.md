# Learned Descend-Until-Contact — Design Spec

Status: DRAFT for review (no training started). Method step following the diagnosis +
gate-oracle confirmation (see RESEARCH_LOG / memory). Items marked **[CONFIRM]** need your
input before implementation.

## 0. Background (one paragraph)
Diagnosis (CORRECT weight `smolvla_20260624_081946`, 2026-06-25) established that the vanilla
SmolVLA under-reaches: on the depletion sweep its descent stalls at a habitual depth (~0.82–0.83 m)
regardless of where the object is, so seal% collapses with depth (ungated **70/60/0/20/0** over
layers 0–4; failures pin at ~0.82–0.83 while the objects descend). The privileged contact-gate
(force descent until contact, same model, no retrain) recovered the deep layers — **L2 0→80%,
L3 20→100%, L4 0→90%** — reaching the true objects (0.77–0.80, tracking down with the layer) —
proving the failure is **condition-ignoring** (not capability, data, or OOD). The hard gate
**over-presses** across layers (|F| ~16–19 N vs demo 15.8); its residual gated failures (incl.
L1 73% and L4 90%) were **over-press — "pushing too hard" — not a distinct failure mode** (user-
observed). So **L1 is under-reach like the deeper layers** (gate would recover more if graded), the
gated seal% is a **lower bound**, and **this sweep contains NO alignment failure — the unobservable
negative control (S5) must come from a separate alignment-gated transition (e.g. insertion), not L1.**
This spec is the *learned* fix: a policy whose descend→grasp transition is grounded in the contact
condition, graded (no over-press), and — the scientific claim — fixed by **grounding the condition**,
not by extra capacity or data.

## 1. Goal & falsifiable success criteria
- **S1 — recovers under-reach:** V2 seal% across the sweep ≈ the gate oracle (L3/L4 high), vs V0
  collapse, **without** the gate's over-press (graded |F| at contact, no ~36 N spike).
- **S2 — grounding is the active ingredient:** **V2 > V1** (equal-capacity, ungrounded). If V1 also
  recovers, the win is capacity/architecture, not grounding → claim fails.
- **S3 — the action uses ĉ (no bypass):** counterfactual — intervening on ĉ at test flips the
  descend/stop behavior; ablating ĉ re-breaks it.
- **S4 — low-data signature:** V2's gain over V0 is **largest at few (deep) demos** and shrinks as
  deep data is added (else it's "nicer architecture," not a low-data causal fix).
- **S5 — specificity / no harm:** on a genuinely *unobservable* condition (lateral alignment — an
  insertion/alignment-gated transition; **NOT** L1, which turned out to be under-reach + gate over-press)
  grounding contact should **not** help = the negative control. And V2 must **not** degrade the easy
  layers or other sub-tasks. NOTE: the case_pick depletion sweep has no alignment failure — source the
  negative control elsewhere (e.g. battery insertion).

## 2. Scope
One transition (approach→descend→grasp, observable condition = **contact force**), one task/
embodiment/model (case_pick suction, SmolVLA). Generality (battery, 2nd model) is later. The
depletion sweep is both the diagnosis rig and the eval rig; the gate oracle (~100%) is the upper bound.

## 3. Condition representation `ĉ`
- **Definition:** `ĉ_t` = contact level, computed from the wrench already in the obs:
  `contact = |F[:3]| − baseline_f` (per-episode resting/gravity baseline), then
  `ĉ_t = clip(contact / τ, 0, 1)` (continuous soft-contact; **[CONFIRM]** binary vs continuous —
  recommend continuous to preserve within-phase force info). τ calibrated from demo force traces
  (contact rises ~14→16 N; τ ≈ a few N above baseline, same threshold family as the gate's `--contact-n`).
- **Computed, not a learned head.** Contact is *observable*, so ĉ needs no inference network — and a
  learned head would be one more thing the action can bypass. We compute ĉ from force directly.
  (The whole failure is that the policy *ignored* the raw force; §4's bottleneck forces it to use ĉ.)
- **Auto-labeled at train** from the recorded force trace; **computed live at test** from the wrench
  (`run_policy` already tracks `baseline_f` per task).

## 4. Architecture (SmolVLA-specific)
Repo-side subclass (cannot edit the venv lerobot). Three pieces:
1. **Force-mask (the bottleneck):** zero the wrench dims (state idx **9:15**) in the state token that
   feeds the action expert, so the action's *only* contact channel is ĉ. This is what makes
   "transition depends on ĉ, not around it" real (requirement #2 in the thesis). Force is still read
   to *compute* ĉ.
2. **ĉ → FiLM:** small MLP `ĉ → (γ, β)`, modulating `suffix_out` `[B, chunk, expert_hidden=240]`
   **before `action_out_proj`** in `VLAFlowMatching.denoise_step()` (and `forward()` for training) —
   the injection point already mapped.
3. **Train the action expert + FiLM; freeze the VLM backbone** **[CONFIRM]** (cheaper; the change is
   in action generation, not perception). Init from `smolvla_depthseal_fixed`.

**The three variants (the experiment):**
| | force in action path | ĉ → FiLM | meaning |
|---|---|---|---|
| **V0** | yes (raw) | none | existing vanilla policy — the failure |
| **V1** | masked | **decorrelated ĉ** (contact label shuffled across episodes; same marginal, no info) | equal capacity + same conditioning mechanism, grounding REMOVED |
| **V2** | masked | **true ĉ** | the method |

V1 isolates *information/grounding* from architecture+capacity (the decorrelation requirement #3).
**S2 = V2 > V1 ≈ V0.**

## 5. Training
- **Finetune from `smolvla_20260624_081946/checkpoints/last`** (the CORRECT weight) on the **same
  demos** (we want to fix the *same* learned model's failure, and show grounding ≠ more data).
- Loss = the existing flow-matching BC. ĉ is computed → **no extra loss term** for V2; V1 uses the
  shuffled ĉ. Force-mask applied identically train+test.
- **[CONFIRM — blocking]** the training command/config that produced `smolvla_depthseal_fixed`, where
  it runs (Thor? workstation?), and the time/compute budget per run. The subclass + mask + FiLM have
  to hook into that pipeline; this is the main engineering unknown.

## 6. Evaluation (on the depletion sweep)
- **Primary (S1/S2):** seal% vs layer for {V0, V1, V2, gate-oracle}. Predict V2 ≈ oracle, V1 ≈ V0.
  Reuse `lift_condition_probe.py --by-height`.
- **deepest_z vs layer:** V2 tracks the object (descends to contact); V0/V1 pin at ~0.82.
- **Over-press (S1):** V2's |F| at contact graded (no 36 N spike) — the learned advantage over the gate.
- **Counterfactual (S3):** at test force ĉ=0 while truly in contact (→ should keep descending) and
  ĉ=1 prematurely (→ should stop); behavior must flip. Ablate ĉ → under-reach returns.
- **Low-data curve (S4):** retrain V0 & V2 on demo subsets (few / few-deep); plot gain vs #demos.
- **Specificity (S5):** V2 still fails L1 at-depth alignment; L0–L2 and other sub-tasks unharmed.

## 7. Risks / open questions
- **Force-mask may hurt within-phase force control** (press/lift) → use *continuous* ĉ; monitor S5.
- **ĉ noisy at the exact contact instant** (demos show F drop→rebuild at the deepest point) → use
  sustained/soft ĉ, not a single-frame hard threshold.
- **Modest architectural novelty:** computed-ĉ + FiLM is close to "give the policy a clean contact
  signal it's forced to use." That's fine — per positioning, the contribution is the *demonstration +
  the V1/V2 + low-data causal result*, not the FiLM. Don't oversell the architecture.
- **Pipeline integration (lerobot training)** is the main unknown — pending §5 [CONFIRM].

## 8. Decisions to confirm before coding
1. **[blocking]** training command/config + compute location + budget (§5).
2. ĉ binary vs **continuous** (rec: continuous).
3. VLM **frozen** vs full finetune (rec: freeze, train action-expert + FiLM).
4. V1 control = **decorrelated/shuffled ĉ** (rec) vs free-latent FiLM.
5. Build order: V0(exists) → **V2** → V1 → low-data curve, eval on the sweep each time.

## 9. Running it (implemented)
Files in `LGES/vla_training/`: `film_contact.py` (patch), `train_film.py`/`train_film.sh` (train),
`self_test_film.py` (unit check); `run_policy.py --film` (eval). Don't train/roll while the robot
demo uses the GPU.

```bash
cd LGES/vla_training
# Smoke test (verify full forward + checkpoint roundtrip before the full run):
FILM_VARIANT=v2 RUN_NAME=smoke ./train_film.sh --steps=4 --save_freq=2
#   check: finite loss, no crash, contact_film.* keys in outputs/smoke/.../model.safetensors

# Full runs (finetune from the corrected vanilla ckpt; FiLM starts as identity):
FILM_VARIANT=v2 RUN_NAME=film_v2 ./train_film.sh    # the method
FILM_VARIANT=v1 RUN_NAME=film_v1 ./train_film.sh    # decorrelated control

# Eval on the depletion sweep (deplete top-first, ~10 rounds/layer), with --film:
python run_policy.py --go --task case_pick --goto-start <take> \
  --checkpoint outputs/film_v2/checkpoints/last --film \
  --log-dir ../../Research/condition_driven/rollouts_v2/case_pick_<N> --loop
#   V1: --checkpoint outputs/film_v1/checkpoints/last --film -> rollouts_v1/
#   V0: the vanilla ckpt, NO --film -> rollouts/ (already collected)

# Compare (S2 = V2 > V1 ~ V0 on deep-layer seal%, V2 ~ gate oracle WITHOUT over-press):
cd ../../Research/condition_driven
python lift_condition_probe.py --by-height rollouts_v2/case_pick_*
```
