"""FiLM condition-conditioning port for lerobot ACT — descend-until-contact,
prefix-style (state-token) injection at the transformer ENCODER input.

Architecture mapping vs the SmolVLA patch (film_contact.py, inject='prefix'):
  - ACT builds one state token via `encoder_robot_state_input_proj` (Linear(state_dim,
    dim_model)), appended alongside the latent + image-patch tokens into the main
    transformer ENCODER's input sequence (see modeling_act.py ACT.forward). This is the
    direct analogue of SmolVLA's state_proj output that inject='prefix' modulates.
  - use_vae=True (ACT's CVAE) ALSO reads observation.state, via a SEPARATE projection
    (`vae_encoder_robot_state_input_proj`), to build the training-time latent posterior.
    FiLM only hooks `encoder_robot_state_input_proj` (not the VAE one), but mask_force
    zeroes the conditioned dims of the shared `batch[OBS_STATE]` BEFORE either projection
    ever sees it (same batch key, same tensor) — same bottleneck guarantee as SmolVLA's
    embed_prefix masking.
  - c-hat is computed at the POLICY level (ACTPolicy.forward / predict_action_chunk,
    before self.model(batch) is called) since that's where the raw batch dict — with
    OBS_STATE still a plain tensor — is available. Same design as film_contact_pi05.
  - ACT's observation.state normalization is MEAN_STD (confirmed via a trained config),
    same as SmolVLA — so film_contact's `_condition_from_state` (mean/std un-normalize)
    is reused as-is, no pi05-style quantile un-normalization needed.

Usage (BEFORE from_pretrained):
  import film_contact, film_contact_act as fca
  wm, ws = film_contact.load_wrench_stats(dataset_root)
  sm, ss = film_contact.load_seal_stats(dataset_root)
  fca.apply('v2', wm, ws, seal_mean=sm, seal_std=ss, cond=('contact', 'fz', 'seal'),
            mask_force=True)
"""
from __future__ import annotations

import torch

from film_contact import (ContactFiLM, _canon, _condition_from_state, WRENCH_LO, WRENCH_HI,
                          FZ_IDX, SEAL_IDX, DFMAG_IDX)

_CFG = {"variant": "v0", "mask_force": True, "cond": ("contact", "fz", "seal")}


def apply(variant: str, wrench_mean: torch.Tensor, wrench_std: torch.Tensor,
          seal_mean: torch.Tensor = None, seal_std: torch.Tensor = None,
          cond=("contact", "fz", "seal"), contact_F0: float = 6.0, contact_tau: float = 4.0,
          fz_tau: float = 5.0, fz_off: float = 2.6, fmag_off: float = 5.1, fmag_tau: float = 5.0,
          dfmag_mean: torch.Tensor = None, dfmag_std: torch.Tensor = None,
          dfmag_tau: float = 5.0, mask_force: bool = True) -> None:
    """Patch ACT (state-token FiLM) + ACTPolicy (c-hat from batch state, mask_force).
    Idempotent like film_contact.apply. `cond` is structural (fixed at first call)."""
    from lerobot.policies.act.modeling_act import ACT, ACTPolicy
    from lerobot.utils.constants import OBS_STATE

    cond = _canon(cond)
    if "seal" in cond and (seal_mean is None or seal_std is None):
        raise ValueError("cond includes 'seal' but seal_mean/seal_std were not provided")
    if "dfmag" in cond and (dfmag_mean is None or dfmag_std is None):
        raise ValueError("cond includes 'dfmag' but dfmag_mean/dfmag_std were not provided "
                         "(needs a *_dF dataset with state dim 16)")

    _CFG.update(variant=variant, mask_force=mask_force, cond=cond)
    if getattr(ACT, "_film_patched", False):
        return

    orig_init = ACT.__init__

    def new_init(self, config):
        orig_init(self, config)
        self._film_cond = cond
        self.contact_film = ContactFiLM(self.encoder_robot_state_input_proj.out_features,
                                         cond_dim=len(cond))
        self.register_buffer("_wrench_mean", wrench_mean.clone())
        self.register_buffer("_wrench_std", wrench_std.clone())
        if "seal" in cond:
            self.register_buffer("_seal_mean", seal_mean.clone())
            self.register_buffer("_seal_std", seal_std.clone())
        if "dfmag" in cond:
            self.register_buffer("_dfmag_mean", dfmag_mean.clone())
            self.register_buffer("_dfmag_std", dfmag_std.clone())
            self.register_buffer("_dfmag_tau", torch.tensor(float(dfmag_tau)), persistent=False)
        for name, val in [("_contact_F0", contact_F0), ("_contact_tau", contact_tau),
                          ("_fz_tau", fz_tau), ("_fz_off", fz_off),
                          ("_fmag_off", fmag_off), ("_fmag_tau", fmag_tau)]:
            # non-persistent: runtime eval hyperparameters, not learned/trained values.
            self.register_buffer(name, torch.tensor(float(val)), persistent=False)
        self._cur_contact = None
        owner = self  # closure ref (NOT a submodule -> not in state_dict, no recursion)

        def _state_film_hook(_module, _inp, out):
            c = owner._cur_contact
            if c is None:
                return out
            return owner.contact_film(out[:, None, :], c)[:, 0, :]  # (B,H) -> film -> (B,H)

        self.encoder_robot_state_input_proj.register_forward_hook(_state_film_hook)

    ACT.__init__ = new_init

    def _set_cond_and_mask(policy, batch, training: bool):
        st = batch.get(OBS_STATE)
        if st is None:
            policy.model._cur_contact = None
            return batch
        c = _condition_from_state(policy.model, st)
        if _CFG["variant"] == "v1" and training:  # decorrelated control
            c = c[torch.randperm(c.shape[0], device=c.device)]
        policy.model._cur_contact = c
        if _CFG["mask_force"]:  # bottleneck: mask conditioned dims for BOTH proj layers that read it
            st = st.clone()
            if "contact" in cond or "fmag" in cond:
                st[..., WRENCH_LO:WRENCH_HI] = 0.0
            if "fz" in cond:
                st[..., FZ_IDX] = 0.0
            if "seal" in cond:
                st[..., SEAL_IDX] = 0.0
            if "dfmag" in cond:
                st[..., DFMAG_IDX] = 0.0
            batch = dict(batch)
            batch[OBS_STATE] = st
        return batch

    orig_forward = ACTPolicy.forward
    orig_predict = ACTPolicy.predict_action_chunk

    def new_forward(self, batch):
        batch = _set_cond_and_mask(self, batch, self.training)
        return orig_forward(self, batch)

    def new_predict(self, batch):
        batch = _set_cond_and_mask(self, batch, False)
        return orig_predict(self, batch)

    ACTPolicy.forward = new_forward
    ACTPolicy.predict_action_chunk = new_predict
    ACT._film_patched = True
