# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""SmolVLA trained with the MeanFlow objective for one-step action generation.

Instead of the instantaneous velocity v(x_t, t) = noise - actions used by SmolVLA's flow
matching, the action expert predicts the average velocity over an interval [r, t]:

    u(x_t, r, t) = 1/(t - r) * integral_r^t v(x_tau, tau) dtau        (r <= t)

which satisfies x_r = x_t - (t - r) * u(x_t, r, t). One-step inference is therefore
x_0 = x_1 - u(x_1, r=0, t=1) with x_1 pure noise. Training uses the MeanFlow identity

    u(x_t, r, t) = v(x_t, t) - (t - r) * d/dt u(x_t, r, t)

where d/dt is the total derivative along the flow, computed exactly with a single
jacobian-vector product with tangent (dx/dt, dr/dt, dt/dt) = (v_t, 0, 1).

Everything is a subclass of the stock SmolVLA classes; no lerobot sources are modified.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from smolvla_meanflow.configuration_smolvla_meanflow import SmolVLAMeanFlowConfig


class VLAMeanFlow(VLAFlowMatching):
    """VLAFlowMatching with an average-velocity head conditioned on (t, t - r).

    Time convention follows SmolVLA: t=1 is pure noise, t=0 is data, and
    x_t = t * noise + (1 - t) * actions.
    """

    def __init__(self, config: SmolVLAMeanFlowConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size
        # Extra conditioning on the interval length (t - r), added to the timestep embedding
        # before the existing action/time fusion MLP. Zero-init keeps u(x, t, t) identical to
        # the instantaneous velocity of a warm-start SmolVLA checkpoint at initialization.
        self.action_interval_proj = nn.Linear(expert_hidden_size, expert_hidden_size)
        if config.zero_init_interval_proj:
            nn.init.zeros_(self.action_interval_proj.weight)
            nn.init.zeros_(self.action_interval_proj.bias)

    def embed_suffix(self, noisy_actions, timestep, interval):
        """Same as SmolVLA's suffix embedding, with an extra (t - r) interval conditioning."""
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).type(dtype=dtype)
        interval_emb = create_sinusoidal_pos_embedding(
            interval,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).type(dtype=dtype)
        time_emb = time_emb + self.action_interval_proj(interval_emb)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs = action_time_emb
        pad_masks = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=device)
        att_masks = torch.tensor(
            [1] * self.config.chunk_size, dtype=action_time_emb.dtype, device=device
        )
        att_masks = att_masks[None, :].expand(bsize, self.config.chunk_size)
        return embs, pad_masks, att_masks

    def sample_time_pair(self, bsize, device):
        """Sample (t, r) with r <= t, forcing r == t (plain flow matching) on a fraction
        1 - meanflow_time_diff_ratio of the batch."""
        t1 = self.sample_time(bsize, device)
        t2 = self.sample_time(bsize, device)
        t = torch.maximum(t1, t2)
        r = torch.minimum(t1, t2)
        keep_fm = torch.rand(bsize, device=device) >= self.config.meanflow_time_diff_ratio
        r = torch.where(keep_fm, t, r)
        return t, r

    def embed_prefix_and_fill_kv_cache(self, images, img_masks, lang_tokens, lang_masks, state):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def predict_average_velocity(self, x_t, r, t, prefix_pad_masks, past_key_values):
        """One expert forward predicting u(x_t, r, t), attending to the cached prefix KV."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, t, t - r)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, r_timestep):
        # Signature intentionally differs from the parent: any caller expecting the
        # instantaneous-velocity denoise_step (e.g. RTC) must fail loudly.
        return self.predict_average_velocity(x_t, r_timestep, timestep, prefix_pad_masks, past_key_values)

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """MeanFlow training loss, per element (batch_size x chunk_size x num_motors).

        Returns squared errors between u(x_t, r, t) and the stop-gradient MeanFlow target,
        with the same shape contract as the parent so the policy wrapper's padding/masking
        logic applies unchanged.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            t, r = self.sample_time_pair(actions.shape[0], actions.device)
        else:
            # A caller-provided time degenerates to r == t (plain flow matching), keeping the
            # parent's (noise, time) reproducibility contract.
            t = time
            r = time

        # The prefix does not depend on (x_t, r, t): compute it once outside the JVP and reuse
        # its KV cache, exactly like inference does. Gradients still flow through the cache and
        # through the trainable expert projections applied on top of the cached states.
        prefix_pad_masks, past_key_values = self.embed_prefix_and_fill_kv_cache(
            images, img_masks, lang_tokens, lang_masks, state
        )

        time_expanded = t[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        v_t = noise - actions

        def u_fn(x, r_, t_):
            return self.predict_average_velocity(x, r_, t_, prefix_pad_masks, past_key_values)

        # MeanFlow identity: u = v - (t - r) * du/dt, with du/dt the total derivative along the
        # flow, i.e. the JVP of u with tangent (dx/dt, dr/dt, dt/dt) = (v_t, 0, 1).
        u_pred, du_dt = torch.func.jvp(
            u_fn, (x_t, r, t), (v_t, torch.zeros_like(r), torch.ones_like(t))
        )
        u_target = v_t - (t - r)[:, None, None] * du_dt
        losses = F.mse_loss(u_target.detach(), u_pred, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs) -> Tensor:
        """Generate an action chunk in config.num_steps evaluations (1 by default):
        x_{t-1/N} = x_t - (1/N) * u(x_t, t - 1/N, t)."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_pad_masks, past_key_values = self.embed_prefix_and_fill_kv_cache(
            images, img_masks, lang_tokens, lang_masks, state
        )

        num_steps = self.config.num_steps
        x_t = noise
        for step in range(num_steps):
            t_val = 1.0 - step / num_steps
            r_val = 1.0 - (step + 1) / num_steps
            t_tensor = torch.full((bsize,), t_val, dtype=torch.float32, device=device)
            r_tensor = torch.full((bsize,), r_val, dtype=torch.float32, device=device)
            u = self.predict_average_velocity(x_t, r_tensor, t_tensor, prefix_pad_masks, past_key_values)
            x_t = x_t - (t_val - r_val) * u
        return x_t


class SmolVLAMeanFlowPolicy(SmolVLAPolicy):
    """SmolVLAPolicy wrapper around VLAMeanFlow. Same batch/loss/action-chunk interfaces
    as SmolVLAPolicy, so training scripts and downstream RL wrappers drive it identically."""

    config_class = SmolVLAMeanFlowConfig
    name = "smolvla_meanflow"

    def __init__(self, config: SmolVLAMeanFlowConfig, **kwargs):
        # Skip SmolVLAPolicy.__init__ so the parent's VLAFlowMatching (and a second copy of
        # the VLM weights) is never built.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAMeanFlow(config, rtc_processor=self.rtc_processor)
        self.reset()

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Same as SmolVLAPolicy.forward plus the MeanFlow adaptive loss weighting
        w = 1 / (||delta||^2 + c)^(1 - gamma), applied per sample."""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        delta_sq = losses.mean(dim=(1, 2))
        weights = (delta_sq.detach() + self.config.meanflow_adaptive_c) ** (
            self.config.meanflow_adaptive_gamma - 1.0
        )
        losses = losses * weights[:, None, None]
        loss_dict["meanflow_adaptive_weight"] = weights.mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
            "|action_interval_proj"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
