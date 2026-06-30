# Related Work — condition-driven transitions

Literature map for the thesis: *make VLA phase transitions CAUSAL (gated on a physically-grounded
condition representation `ĉ`) rather than imitative (driven by pose/time/vision correlates)*, with
an observable-vs-latent predicate split as the spine and FiLM contact-gating as the prototype.

Compiled 2026-06-29. PDFs in this folder = the bullseye set (A/B). Other clusters are links only.

**The gap to defend:** PhaForce + ForceVLA + DPTG already gate on force/contact. Novelty must rest on
what none of them have: (1) the **observability gradient as a falsifiable control** (fix observable
conditions, leave latent alignment as the NEGATIVE control); (2) **transition-specificity** (only
transition-bound failures, not within-phase quality); (3) the **low-data signature** (gain largest at
few demos, shrinks with data).

## A. Bullseye — contact/force/phase-gated transitions (closest prior art) [PDFs here]
- **PhaForce: Phase-Scheduled Visual-Force Policy Learning with Slow Planning and Fast Correction**
  (Wang et al.) — arxiv 2603.08342 — `PhaForce_2603.08342.pdf`. NEAREST NEIGHBOR: phase transitions
  gated on contact/force, NOT elapsed time; slow-plan/fast-correct. Gap: designed phase schedule, not
  a learned grounded predicate w/ no-bypass/decorrelation; no negative-control/observability story.
- **ForceVLA: Force-aware MoE for Contact-rich Manipulation** — arxiv 2505.22159 —
  `ForceVLA_2505.22159.pdf`. Force as dedicated tokens fused via MoE into a VLA. It's feature fusion,
  not causal transition gating — our 3-condition (grounded supervision / no bypass / decorrelation)
  argument separates us.
- Tactile-Conditioned Diffusion Policy / FARM — arxiv 2510.13324
- TacForeSight (force-conditioned tactile world model) — arxiv 2606.11184
- DPTG (tactile = feasibility classifier that GATES vision-driven actions) — Frontiers Robotics & AI 2026
- FILIC (Dual-Loop Force-Guided IL w/ impedance) — arxiv 2509.17053

## B. Intermediate representations between obs and action (the `ĉ` architecture) [PDF here]
- **RT-Affordance: Affordances are Versatile Intermediate Representations** — arxiv 2411.02704 —
  `RT-Affordance_2411.02704.pdf`. Predict affordance plan -> affordance-conditioned policy. Same
  SHAPE as ĉ; ours is physically grounded + supervised toward measurable quantities, aimed at
  transition causality not web-transfer generalization.
- FiLM (Perez et al., feature-wise linear modulation) — the mechanism. Robotics uses: MoE-ACT
  (arxiv 2603.15265, FiLM on action tokens); "Gated FiLM" cerebellum regulating physical-context
  influence (arxiv 2601.14628).

## C. Failure mechanism — causal confusion / copycat / shortcut learning
- Causal Confusion in Imitation Learning (de Haan, Jayaraman, Levine) — arxiv 1905.11979
- Invariant Causal Imitation Learning (ICIL) (Bica et al., NeurIPS 2021)
- Fighting Copycat Agents in BC from Observation Histories (Wen et al.) — arxiv 2010.14876
  [relevant to the prev-action thread in reactive-chunking-research]
- Object-Aware Regularization for Causal Confusion — arxiv 2110.14118
- Shortcut Learning in Deep Neural Networks (Geirhos et al., 2020, origin); On the Foundations of
  Shortcut Learning — arxiv 2310.16228

## D. Termination / precondition grounding (observable-vs-latent predicate spine)
- Grounding Predicates through Actions (Migimatsu & Bohg) — arxiv 2109.14718
  [language NAMES ĉ, doesn't carry it]
- Relational Learning for Skill Preconditions (Sharma et al.) — NSF par.nsf.gov/10293068
- Neuro-Symbolic Imitation Learning — arxiv 2503.21406
- SymSkill (symbol+skill co-invention) — arxiv 2510.01661
- Reactive Long Horizon Task Execution via Visual Skill and Precondition Models — arxiv 2011.08694
- Learning Options from Demonstration using Skill Segmentation — arxiv 2001.06793
  [learns WHEN to switch as a binary boundary classifier = the transition predicate, learned]

## E. Reasoning VLAs (language-level transitions — what we ground instead)
- Embodied Chain-of-Thought (ECoT) — arxiv 2407.08693
- RT-H (action hierarchy / action language)
- Emma-X (grounded CoT + look-ahead) — arxiv 2412.11974
- Position: reason at LANGUAGE level, too coarse for a continuous threshold (23N vs 46N), untested
  for causal gating. "reasoning != grounded causal transition."

## F. Action chunking / reactive control (the chunk-latency confound on the lift side)
- Real-Time Chunking (RTC) — arxiv 2506.07339
- Bidirectional Decoding (BID)
- Leave No Observation Behind: Real-Time Correction for VLA Action Chunks — arxiv 2509.23224
- TIDAL — arxiv 2601.14945
- FASTER — arxiv 2603.19199
- Position: these re-ground FASTER on VISION; distinguish from gating on a cheap CONTACT condition.

## G. Privileged distillation (the privileged contact-gate = oracle/upper-bound we already ran)
- Teacher-student privileged distillation (full-state/contact teacher -> obs-only student, via DAgger)
- Learning Long-Horizon Robot Manipulation Skills via Privileged Action — arxiv 2502.15442
- Frame the learned method as distilling our forced descend-until-contact gate (the oracle).
