# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""CPU smoke test for smolvla_meanflow with a truncated (2-layer) VLM.

Run:  HF_HUB_OFFLINE=1 python tests/test_smoke.py
Requires HuggingFaceTB/SmolVLM2-500M-Video-Instruct config/tokenizer in the HF cache.

Checks:
1. Training forward + backward works (torch.func.jvp through the expert stack).
2. Gradients reach the expert, the fusion MLPs, and the new interval projection.
3. With identical weights and r == t, the MeanFlow per-element loss is numerically
   identical to stock SmolVLA flow matching (warm-start equivalence).
4. One-step and two-step sampling produce chunks of the right shape.
5. Plugin registration resolves the policy/config/processors through lerobot's factory.
"""

import dataclasses

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from smolvla_meanflow import SmolVLAMeanFlowConfig, SmolVLAMeanFlowPolicy

IMG_KEY = "observation.images.cam"
ACTION_DIM = 6
STATE_DIM = 6
CHUNK = 5
BSIZE = 2

TINY_KWARGS = dict(
    chunk_size=CHUNK,
    n_action_steps=CHUNK,
    max_state_dim=32,
    max_action_dim=32,
    resize_imgs_with_padding=(128, 128),
    load_vlm_weights=False,
    num_vlm_layers=2,
    num_expert_layers=2,
    self_attn_every_n_layers=2,  # layer 0: self-attn path, layer 1: cross-attn path
    tokenizer_max_length=16,
    device="cpu",
)

FEATURES = dict(
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        IMG_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    },
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
)


def make_batch():
    torch.manual_seed(0)
    return {
        IMG_KEY: torch.rand(BSIZE, 3, 128, 128),
        OBS_STATE: torch.randn(BSIZE, STATE_DIM),
        OBS_LANGUAGE_TOKENS: torch.randint(4, 1000, (BSIZE, 16)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(BSIZE, 16, dtype=torch.bool),
        ACTION: torch.randn(BSIZE, CHUNK, ACTION_DIM),
    }


def main():
    torch.manual_seed(0)

    print("[1/5] Building tiny MeanFlow policy ...")
    mf_cfg = SmolVLAMeanFlowConfig(**TINY_KWARGS, **FEATURES, num_steps=1)
    mf_policy = SmolVLAMeanFlowPolicy(mf_cfg).float()
    mf_policy.train()

    batch = make_batch()

    print("[2/5] Training forward/backward through torch.func.jvp ...")
    loss, loss_dict = mf_policy.forward(dict(batch))
    assert loss.ndim == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    loss.backward()
    grads_expected = {
        "model.action_interval_proj.weight": False,
        "model.action_time_mlp_in.weight": False,
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight": False,
        "model.vlm_with_expert.lm_expert.layers.1.self_attn.k_proj.weight": False,  # cross-attn over cached prefix KV
        "model.state_proj.weight": False,
    }
    for name, param in mf_policy.named_parameters():
        if name in grads_expected:
            grads_expected[name] = param.grad is not None and param.grad.abs().sum() > 0
    missing_grads = [k for k, ok in grads_expected.items() if not ok]
    assert not missing_grads, f"no gradient reached: {missing_grads}"
    print(f"    loss={loss.item():.4f}, adaptive_weight={loss_dict['meanflow_adaptive_weight']:.4f}, "
          "gradients reach expert/fusion/interval/state projections")
    mf_policy.zero_grad(set_to_none=True)

    print("[3/5] Warm-start equivalence vs stock SmolVLA (r == t, zero-init interval proj) ...")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    fm_cfg = SmolVLAConfig(**TINY_KWARGS, **FEATURES)
    fm_policy = SmolVLAPolicy(fm_cfg).float()
    state_dict = {k: v for k, v in mf_policy.state_dict().items() if "action_interval_proj" not in k}
    missing, unexpected = fm_policy.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    fm_policy.eval()
    mf_policy.eval()

    actions = torch.nn.functional.pad(batch[ACTION], (0, 32 - ACTION_DIM))
    noise = torch.randn_like(actions)
    time = torch.rand(BSIZE) * 0.8 + 0.1
    images, img_masks = mf_policy.prepare_images(dict(batch))
    with torch.no_grad():
        args = (images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                torch.nn.functional.pad(batch[OBS_STATE], (0, 32 - STATE_DIM)))
        losses_mf = mf_policy.model.forward(*args, actions, noise=noise, time=time)
        losses_fm = fm_policy.model.forward(*args, actions, noise=noise, time=time)
    max_diff = (losses_mf - losses_fm).abs().max().item()
    assert max_diff < 1e-4, f"MeanFlow(r=t) != stock flow matching, max diff {max_diff}"
    print(f"    per-element losses match stock SmolVLA (max diff {max_diff:.2e})")

    print("[4/5] One-step and two-step sampling ...")
    with torch.no_grad():
        chunk = mf_policy.predict_action_chunk(dict(batch))
        assert chunk.shape == (BSIZE, CHUNK, ACTION_DIM), chunk.shape
        mf_policy.config.num_steps = 2
        chunk2 = mf_policy.predict_action_chunk(dict(batch))
        assert chunk2.shape == (BSIZE, CHUNK, ACTION_DIM)
        mf_policy.config.num_steps = 1
    assert torch.isfinite(chunk).all() and torch.isfinite(chunk2).all()
    print(f"    action chunk shape {tuple(chunk.shape)} for num_steps=1 and 2")

    print("[5/5] Factory/plugin resolution ...")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    assert PreTrainedConfig.get_choice_class("smolvla_meanflow") is SmolVLAMeanFlowConfig
    assert get_policy_class("smolvla_meanflow") is SmolVLAMeanFlowPolicy
    pre, post = make_pre_post_processors(mf_cfg)
    assert pre is not None and post is not None
    round_trip = SmolVLAMeanFlowConfig(**{
        f.name: getattr(mf_cfg, f.name) for f in dataclasses.fields(mf_cfg) if f.init
    })
    assert round_trip.type == "smolvla_meanflow"
    print("    config registry, policy class, and processors all resolve")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
