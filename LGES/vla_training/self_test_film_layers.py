"""Self-test for inject='layers' (per-layer FiLM on the action decoder/expert).

Verifies, per architecture:
  [A] identity-at-init : with all films zero-init, film-ON output == film-OFF output
                         (mask_force=0 in the smolvla arm so ON/OFF differ ONLY by FiLM)
  [B] fire count       : ContactFiLM.forward fires once per decoder/expert layer per pass
  [C] authority        : perturbing ONE layer's film bias changes the output

Run each arch in its own process (apply() is structural per process):
  python self_test_film_layers.py act
  python self_test_film_layers.py smolvla   # loads lerobot/smolvla_base (cached)
  python self_test_film_layers.py pi0       # loads lerobot/pi0_base; checks BOTH forward
                                            # paths (mixed train loop + cached denoise)
  python self_test_film_layers.py groot     # loads outputs/groot_naive_0729 best; drives
                                            # select_action through the real processors
"""
import sys

import torch

import film_contact

# count ContactFiLM invocations (the hooks skip the film entirely when c is None)
CALLS = {"n": 0}
_orig_film_fwd = film_contact.ContactFiLM.forward


def _counting_fwd(self, x, c):
    CALLS["n"] += 1
    return _orig_film_fwd(self, x, c)


film_contact.ContactFiLM.forward = _counting_fwd

WM, WS = torch.zeros(6), torch.ones(6)
SM, SS = torch.tensor(0.0), torch.tensor(1.0)
COND = ("contact", "fz", "seal")


def test_act():
    import film_contact_act as fca
    fca.apply("v2", WM, WS, seal_mean=SM, seal_std=SS, cond=COND,
              mask_force=True, inject="layers")

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACT

    n_dec = 3
    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (15,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=10, n_action_steps=10, dim_model=64, n_heads=4, dim_feedforward=128,
        n_encoder_layers=2, n_decoder_layers=n_dec, use_vae=False,
    )
    torch.manual_seed(0)
    model = ACT(cfg).eval()
    assert isinstance(model.contact_film, torch.nn.ModuleList) and len(model.contact_film) == n_dec

    B = 2
    batch = {"observation.state": torch.randn(B, 15),
             "observation.environment_state": torch.randn(B, 8)}
    c = torch.rand(B, len(COND))

    with torch.no_grad():
        model._cur_contact = None
        out_off, _ = model(batch)
        CALLS["n"] = 0
        model._cur_contact = c
        out_on, _ = model(batch)
    assert CALLS["n"] == n_dec, f"expected {n_dec} film fires, got {CALLS['n']}"
    assert torch.allclose(out_on, out_off, atol=1e-5), \
        f"not identity at init (max diff {(out_on - out_off).abs().max():.2e})"
    print(f"[act A/B] identity-at-init OK, fire count {CALLS['n']} == n_decoder_layers {n_dec}")

    with torch.no_grad():
        # perturb GAMMA, not beta: a uniform beta shift is exactly nullified by the
        # decoder layer's LayerNorm (mean subtraction); gamma scales x, which survives.
        model.contact_film[1].scale[-1].bias.add_(1.0)
        out_pert, _ = model(batch)
        model._cur_contact = None
        out_off2, _ = model(batch)
    assert not torch.allclose(out_pert, out_on, atol=1e-5), "perturbed film did not move the output"
    assert torch.allclose(out_off2, out_off, atol=1e-5), "c=None opt-out leaked the perturbed film"
    print("[act C] perturbed layer-1 film moves the output; c=None opt-out clean")


def test_smolvla():
    # mask_force=0 so film-ON vs film-OFF (instance opt-out) differ ONLY by the film hooks
    film_contact.apply("v2", WM, WS, seal_mean=SM, seal_std=SS, cond=COND,
                       mask_force=False, inject="layers")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    model = policy.model.eval()
    films = model.contact_film
    n_exp = len(model.vlm_with_expert.lm_expert.layers)
    assert isinstance(films, torch.nn.ModuleList) and len(films) == n_exp

    torch.manual_seed(0)
    # mixed-dtype checkpoint: vision tower may be bf16 while the proj heads stay fp32
    dev = model.state_proj.weight.device
    dt_img = next(model.vlm_with_expert.get_vlm_model().vision_model.parameters()).dtype
    dt_st = model.state_proj.weight.dtype
    B, cs, ad = 1, policy.config.chunk_size, policy.config.max_action_dim
    images = [torch.rand(B, 3, 512, 512, device=dev, dtype=dt_img)]
    img_masks = [torch.ones(B, dtype=torch.bool, device=dev)]
    lang_tokens = torch.randint(5, 1000, (B, 16), device=dev)
    lang_masks = torch.ones(B, 16, dtype=torch.bool, device=dev)
    state = torch.randn(B, policy.config.max_state_dim, device=dev, dtype=dt_st)
    actions = torch.randn(B, cs, ad, device=dev, dtype=dt_st)
    noise = torch.randn(B, cs, ad, device=dev, dtype=dt_st)
    time = torch.full((B,), 0.5, device=dev, dtype=dt_st)

    with torch.no_grad():
        cond_bak = model._film_cond
        model._film_cond = None          # instance opt-out => no c-hat, hooks no-op
        model._cur_contact = None
        loss_off = model(images, img_masks, lang_tokens, lang_masks, state, actions,
                         noise=noise, time=time)
        model._film_cond = cond_bak
        CALLS["n"] = 0
        loss_on = model(images, img_masks, lang_tokens, lang_masks, state, actions,
                        noise=noise, time=time)
    n_engaged = sum(1 for _ in model.vlm_with_expert.lm_expert.layers)
    assert CALLS["n"] == n_engaged, f"expected {n_engaged} film fires, got {CALLS['n']}"
    assert torch.allclose(loss_on, loss_off, atol=1e-5), \
        f"not identity at init (max diff {(loss_on - loss_off).abs().max():.2e})"
    print(f"[smolvla A/B] identity-at-init OK, fire count {CALLS['n']} == expert layers {n_engaged}")

    with torch.no_grad():
        films[n_exp // 2].scale[-1].bias.add_(1.0)
        loss_pert = model(images, img_masks, lang_tokens, lang_masks, state, actions,
                          noise=noise, time=time)
    assert not torch.allclose(loss_pert, loss_on, atol=1e-5), "perturbed film did not move the loss"
    print("[smolvla C] perturbed mid-layer film moves the loss")


def test_pi0():
    # mask_force=0 so film-ON vs instance opt-out differ ONLY by the film hooks
    import film_contact_pi0 as fc0
    fc0.apply("v2", WM, WS, seal_mean=SM, seal_std=SS, cond=COND,
              mask_force=False, inject="layers")

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained("lerobot/pi0_base")
    model = policy.model.eval()
    n_exp = len(model.paligemma_with_expert.gemma_expert.model.layers)
    films = model.contact_film
    assert isinstance(films, torch.nn.ModuleList) and len(films) == n_exp

    torch.manual_seed(0)
    dev = model.state_proj.weight.device
    dt_st = model.state_proj.weight.dtype
    dt_img = next(model.paligemma_with_expert.paligemma.model.vision_tower.parameters()).dtype
    B, cs, ad = 1, policy.config.chunk_size, policy.config.max_action_dim
    images = [torch.rand(B, 3, 224, 224, device=dev, dtype=dt_img)]
    img_masks = [torch.ones(B, dtype=torch.bool, device=dev)]
    lang_tokens = torch.randint(5, 1000, (B, 16), device=dev)
    lang_masks = torch.ones(B, 16, dtype=torch.bool, device=dev)
    state = torch.randn(B, policy.config.max_state_dim, device=dev, dtype=dt_st)
    actions = torch.randn(B, cs, ad, device=dev, dtype=dt_st)
    noise = torch.randn(B, cs, ad, device=dev, dtype=dt_st)
    time = torch.full((B,), 0.5, device=dev, dtype=dt_st)

    with torch.no_grad():
        # path 1: the mixed train forward (hand-rolled compute_layer_complete)
        cond_bak = model._film_cond
        model._film_cond = None
        model._cur_contact = None
        loss_off = model(images, img_masks, lang_tokens, lang_masks, state, actions,
                         noise=noise, time=time)
        model._film_cond = cond_bak
        CALLS["n"] = 0
        loss_on = model(images, img_masks, lang_tokens, lang_masks, state, actions,
                        noise=noise, time=time)
    assert CALLS["n"] == n_exp, f"train path: expected {n_exp} fires, got {CALLS['n']}"
    assert torch.allclose(loss_on, loss_off, atol=1e-5), \
        f"train path not identity at init (max diff {(loss_on - loss_off).abs().max():.2e})"
    print(f"[pi0 train-path] identity OK, fires {CALLS['n']} == expert layers {n_exp}")

    with torch.no_grad():
        # path 2: cached-inference denoise (suffix-only HF forward per step)
        model._film_cond = None
        model._cur_contact = None
        a_off = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                     noise=noise)
        model._film_cond = cond_bak
        CALLS["n"] = 0
        a_on = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                    noise=noise)
    steps = policy.config.num_inference_steps
    assert CALLS["n"] == n_exp * steps, \
        f"denoise path: expected {n_exp}*{steps} fires, got {CALLS['n']}"
    assert torch.allclose(a_on, a_off, atol=1e-5), \
        f"denoise path not identity at init (max diff {(a_on - a_off).abs().max():.2e})"
    print(f"[pi0 denoise-path] identity OK, fires {CALLS['n']} == {n_exp}x{steps} steps")

    with torch.no_grad():
        films[n_exp // 2].scale[-1].bias.add_(1.0)
        a_pert = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                      noise=noise)
    assert not torch.allclose(a_pert, a_on, atol=1e-5), "perturbed film did not move actions"
    print("[pi0 C] perturbed mid-layer film moves the sampled actions")


def test_pi05():
    # pi05 computes c-hat at the POLICY level (state is discretized into the prompt),
    # so the model-level test sets _cur_contact directly. mask_force=0 (tokenizer-step
    # masking is orthogonal here) — film-ON vs OFF differ only by the hooks.
    import film_contact_pi05 as fcp
    q01 = torch.linspace(-1.0, 0.0, 32)
    q99 = torch.linspace(1.0, 2.0, 32)
    fcp.apply("v2", q01, q99, cond=COND, mask_force=False, inject="layers")

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained("lerobot/pi05_base")
    model = policy.model.eval()
    n_exp = len(model.paligemma_with_expert.gemma_expert.model.layers)
    films = model.contact_film
    assert isinstance(films, torch.nn.ModuleList) and len(films) == n_exp

    torch.manual_seed(0)
    dev = model.action_in_proj.weight.device
    dt = model.action_in_proj.weight.dtype
    dt_img = next(model.paligemma_with_expert.paligemma.model.vision_tower.parameters()).dtype
    B, cs, ad = 1, policy.config.chunk_size, policy.config.max_action_dim
    images = [torch.rand(B, 3, 224, 224, device=dev, dtype=dt_img)]
    img_masks = [torch.ones(B, dtype=torch.bool, device=dev)]
    tokens = torch.randint(5, 1000, (B, 32), device=dev)
    masks = torch.ones(B, 32, dtype=torch.bool, device=dev)
    actions = torch.randn(B, cs, ad, device=dev, dtype=dt)
    noise = torch.randn(B, cs, ad, device=dev, dtype=dt)
    time = torch.full((B,), 0.5, device=dev, dtype=dt)
    c = torch.rand(B, len(COND), device=dev)

    with torch.no_grad():
        model._cur_contact = None
        loss_off = model(images, img_masks, tokens, masks, actions, noise=noise, time=time)
        model._cur_contact = c
        CALLS["n"] = 0
        loss_on = model(images, img_masks, tokens, masks, actions, noise=noise, time=time)
    assert CALLS["n"] == n_exp, f"train path: expected {n_exp} fires, got {CALLS['n']}"
    assert torch.allclose(loss_on, loss_off, atol=1e-5), \
        f"train path not identity at init (max diff {(loss_on - loss_off).abs().max():.2e})"
    print(f"[pi05 train-path] identity OK, fires {CALLS['n']} == expert layers {n_exp}")

    with torch.no_grad():
        model._cur_contact = None
        a_off = model.sample_actions(images, img_masks, tokens, masks, noise=noise)
        model._cur_contact = c
        CALLS["n"] = 0
        a_on = model.sample_actions(images, img_masks, tokens, masks, noise=noise)
    steps = policy.config.num_inference_steps
    assert CALLS["n"] == n_exp * steps, \
        f"denoise path: expected {n_exp}*{steps} fires, got {CALLS['n']}"
    assert torch.allclose(a_on, a_off, atol=1e-5), \
        f"denoise path not identity at init (max diff {(a_on - a_off).abs().max():.2e})"
    print(f"[pi05 denoise-path] identity OK, fires {CALLS['n']} == {n_exp}x{steps} steps")

    with torch.no_grad():
        films[n_exp // 2].scale[-1].bias.add_(1.0)
        a_pert = model.sample_actions(images, img_masks, tokens, masks, noise=noise)
    assert not torch.allclose(a_pert, a_on, atol=1e-5), "perturbed film did not move actions"
    print("[pi05 C] perturbed mid-layer film moves the sampled actions")


def test_groot():
    from pathlib import Path
    import train_groot  # noqa: F401  sdpa fallback (no flash-attn on this box)
    import film_contact_groot as fcg

    root = Path(__file__).resolve().parent
    rv = root / "datasets/lges_case_pick_0729_val"
    mn, mx = fcg.load_state_minmax(rv)
    fcg.apply("v2", mn, mx, cond=COND, mask_force=False, inject="layers")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    md = root / "outputs/groot_naive_0729/checkpoints/best/pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(md)
    # fresh construction, not from_pretrained: GrootPolicy's safetensors load is STRICT, so
    # a film-less naive ckpt + film patch raises on the missing contact_film keys. Random
    # head weights are fine for a plumbing test (zero-init films => identity regardless).
    policy = get_policy_class(cfg.type)(cfg)
    policy.to(cfg.device if cfg.device else "cuda")
    policy.eval()
    policy.config.n_action_steps = 1
    blocks = policy._groot_model.action_head.model.transformer_blocks
    n_blk = len(blocks)
    assert isinstance(policy.contact_film, torch.nn.ModuleList) and len(policy.contact_film) == n_blk
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(md),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})

    ds = LeRobotDataset("Chanho-Lee/lges_case_pick_0729_val", root=rv)
    frame = ds[0]
    obs = {"observation.state": frame["observation.state"].unsqueeze(0), "task": frame["task"]}
    for k in frame:
        if k.startswith("observation.images."):
            obs[k] = frame[k].unsqueeze(0)

    def predict():
        policy.reset()
        torch.manual_seed(0)
        with torch.inference_mode():
            return policy.select_action(pre(dict(obs)))

    cond_bak = policy._film_cond
    policy._film_cond = None          # opt-out: no c-hat, hooks no-op
    policy._cur_contact = None
    a_off = predict()
    policy._film_cond = cond_bak
    CALLS["n"] = 0
    a_on = predict()
    assert CALLS["n"] > 0 and CALLS["n"] % n_blk == 0, \
        f"expected a positive multiple of {n_blk} fires, got {CALLS['n']}"
    assert torch.allclose(a_on, a_off, atol=1e-5), \
        f"not identity at init (max diff {(a_on - a_off).abs().max():.2e})"
    print(f"[groot A/B] identity OK, fires {CALLS['n']} = {CALLS['n'] // n_blk} denoise "
          f"steps x {n_blk} DiT blocks")

    with torch.no_grad():
        policy.contact_film[n_blk // 2].scale[-1].bias.add_(1.0)
    a_pert = predict()
    assert not torch.allclose(a_pert, a_on, atol=1e-5), "perturbed film did not move actions"
    print("[groot C] perturbed mid-block film moves the action")


if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "act"
    {"act": test_act, "smolvla": test_smolvla, "pi0": test_pi0, "groot": test_groot,
     "pi05": test_pi05}[arch]()
    print(f"[{arch}] ALL OK")
