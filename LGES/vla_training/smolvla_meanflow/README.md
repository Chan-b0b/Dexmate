# smolvla_meanflow

SmolVLA trained with **MeanFlow** (average-velocity flow matching) for **one-step action
generation**, packaged as a LeRobot third-party plugin. **No file in the installed
`lerobot` package is modified** — everything is a subclass, registered under the new
policy type `smolvla_meanflow`.

References:
- MeanFlow: Geng et al., *Mean Flows for One-step Generative Modeling*, arXiv:2505.13447
- MF-VLA: *Mean-Flow based One-Step Vision-Language-Action*, arXiv:2603.01469

## How it works

Stock SmolVLA learns the instantaneous velocity `v(x_t, t) = noise - actions` and
integrates 10 Euler steps at inference. This package learns the **average velocity**
over an interval `[r, t]` (SmolVLA convention: `t=1` noise, `t=0` data):

```
x_r = x_t - (t - r) * u(x_t, r, t)      =>  one-step: actions = noise - u(noise, 0, 1)
```

Training uses the MeanFlow identity `u = v - (t - r) * du/dt`, where the total
derivative is computed exactly with one `torch.func.jvp` (tangent `(v, 0, 1)`), and the
target is stop-gradiented. The interval `(t - r)` is injected as an extra sinusoidal
embedding through a **zero-initialized** projection, so a warm-started model is exactly
equivalent to the SmolVLA checkpoint it came from at step 0 (verified to ~2e-6).

## Install

```bash
VIRTUAL_ENV=~/vla_venv uv pip install -e ~/smolvla_meanflow --no-deps
```

## Warm-start from a pretrained SmolVLA (recommended)

```bash
python scripts/init_from_smolvla.py \
  --src lerobot/smolvla_base \        # or a local fine-tuned smolvla checkpoint dir
  --out ~/checkpoints/smolvla_meanflow_base
```

Only the two zero-init interval-projection tensors are new; all other weights (and the
pre/post-processor files) are carried over.

## Train

```bash
lerobot-train \
  --policy.path=$HOME/checkpoints/smolvla_meanflow_base \
  --policy.discover_packages_path=smolvla_meanflow \
  --dataset.repo_id=Chanho-Lee/lges_case_pick_0708 \
  --batch_size=64 --steps=200000
```

`--policy.discover_packages_path=smolvla_meanflow` is what makes lerobot import this
package and register the policy type; add it to any lerobot CLI call (`lerobot-eval`,
`lerobot-record`, ...) that touches a `smolvla_meanflow` checkpoint. From-scratch
training also works with `--policy.type=smolvla_meanflow`.

## Config knobs (defaults follow the papers)

| field | default | meaning |
|---|---|---|
| `num_steps` | 1 | inference NFEs; raise to 2–5 for quality/latency trade-off |
| `meanflow_time_diff_ratio` | 0.25 | fraction of samples trained with `r < t` (rest are plain FM) |
| `meanflow_adaptive_gamma` | 0.5 | adaptive loss `w = 1/(||Δ||² + c)^(1-γ)`; `1.0` disables |
| `meanflow_adaptive_c` | 1e-3 | stabilizer in the adaptive weight |
| `zero_init_interval_proj` | True | keep True when warm-starting |

Not supported (rejected at config time): RTC (`rtc_config`), `use_cache=False`,
`compile_model=True` (torch.compile does not compose with the training-time JVP).

## Inference

Identical API to SmolVLA:

```python
import torch
from smolvla_meanflow import SmolVLAMeanFlowPolicy

policy = SmolVLAMeanFlowPolicy.from_pretrained("~/checkpoints/.../pretrained_model")
chunk = policy.predict_action_chunk(batch)   # one forward pass per chunk
```

## Notes for RL fine-tuning afterwards

- The policy keeps `SmolVLAPolicy`'s full interface (`forward`, `predict_action_chunk`,
  `select_action`, queues), so anything that drives a SmolVLA policy drives this one.
- One-step generation makes online rollouts and replay-buffer relabeling ~10x cheaper —
  the usual blocker for RL on flow-matching VLAs.
- `forward(batch, reduction="none")` returns per-sample losses, ready for
  advantage-weighted regression / RA-BC-style weighting.
- The chunk is a differentiable function of the noise: `a = ε - u(ε, 0, 1)` with `ε` the
  sampled latent, so reparameterized / DPG-style policy-gradient updates through the
  actor are possible. Exact log-probs are *not* tractable (as with any distilled
  one-step sampler) — prefer value-weighted or reparameterized objectives over PPO-style
  likelihood ratios on the raw actions.

## Tests

```bash
HF_HUB_OFFLINE=1 python tests/test_smoke.py            # JVP, grads, FM-equivalence, sampling, registry
HF_HUB_OFFLINE=1 python tests/test_checkpoint_load.py  # converted-checkpoint end-to-end inference
```

Implementation notes: the training JVP runs only over the action expert; the VLM prefix
is computed once per batch and reused via the KV cache (its output does not depend on
`(x_t, r, t)`), exactly mirroring the inference path. SmolVLA's attention is eager
(plain matmul + softmax), which is why `torch.func.jvp` composes with it.
