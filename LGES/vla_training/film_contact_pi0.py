"""FiLM condition-conditioning port for lerobot PI0 (π0) — descend-until-contact.

Why π0 and not π0.5 for the injection-point question (2026-08-06): π0.5 has NO state
token at all — observation.state is quantile-normalized, discretized into 256 bins and
written into the language prompt as TEXT ("Task: ..., State: 128 130 99 ...;\nAction: "),
so SmolVLA's winning injection point (FiLM on the state token, where ee_z lives) has no
analogue there. π0 keeps a real dense `state_proj: Linear(max_state_dim -> width)`, and
π0's STATE is MEAN_STD-normalized exactly like SmolVLA's — so film_contact.py's channel
formulas and wrench/seal stats load unchanged, and only the injection plumbing differs.

Architecture note (read before comparing to SmolVLA): in π0 BOTH candidate points live in
embed_suffix — the state token and the action tokens are neighbours in the same expert
sequence, whereas in SmolVLA the state token sits in the VLM *prefix* and the action tokens
in the *suffix*. So this port tests "does WHICH TOKEN you condition matter" (state vs
action), not literally "prefix vs suffix". Read it as the token-level half of the SmolVLA
result, and do not compare magnitudes across architectures.

inject:
  'state'  — FiLM the state token (state_proj output). The SmolVLA-'prefix' analogue.
  'action' — FiLM the action tokens (action_time_mlp output). The 'suffix' analogue.

mask_force zeroes the conditioned dims of `state` BEFORE state_proj (after c-hat is read),
same bottleneck as film_contact.py: the raw wrench cannot reach the model except through
c-hat.

Usage (BEFORE from_pretrained):
  import film_contact_pi0 as fc0
  wm, ws = film_contact.load_wrench_stats(dataset_root)
  sm, ss = film_contact.load_seal_stats(dataset_root)
  fc0.apply('v2', wm, ws, seal_mean=sm, seal_std=ss, cond=('contact','fz','seal'),
            inject='state', mask_force=True)
"""
from __future__ import annotations

import torch

from film_contact import (ContactFiLM, _canon, _condition_from_state, WRENCH_LO, WRENCH_HI,
                         FZ_IDX, SEAL_IDX, DFMAG_IDX)

INJECTS = ("state", "action")
_CFG = {"variant": "v0", "mask_force": True, "cond": ("contact", "fz", "seal"),
        "inject": "state"}


def apply(variant: str, wrench_mean: torch.Tensor, wrench_std: torch.Tensor,
          seal_mean: torch.Tensor | None = None, seal_std: torch.Tensor | None = None,
          cond=("contact", "fz", "seal"), contact_F0: float = 6.0, contact_tau: float = 4.0,
          fz_tau: float = 5.0, fz_off: float = 2.6, fmag_off: float = 5.1,
          fmag_tau: float = 5.0, dfmag_mean: torch.Tensor | None = None,
          dfmag_std: torch.Tensor | None = None, dfmag_tau: float = 5.0,
          inject: str = "state", mask_force: bool = True) -> None:
    """Patch PI0Pytorch.embed_suffix. Idempotent, like film_contact.apply."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch

    if inject not in INJECTS:
        raise ValueError(f"unknown inject {inject!r}; allowed: {INJECTS}")
    cond = _canon(cond)
    _CFG.update(variant=variant, mask_force=mask_force, cond=cond, inject=inject)
    if getattr(PI0Pytorch, "_film_patched", False):
        return

    orig_init = PI0Pytorch.__init__
    orig_suffix = PI0Pytorch.embed_suffix

    def new_init(self, *a, **k):
        orig_init(self, *a, **k)
        self._film_cond = cond
        # 'state' modulates the state token (state_proj width), 'action' the action tokens
        # (action_in_proj width) — in π0 both are the expert width, but keep it explicit.
        hidden = (self.state_proj.out_features if inject == "state"
                  else self.action_in_proj.out_features)
        self.contact_film = ContactFiLM(hidden, cond_dim=len(cond))
        self.register_buffer("_wrench_mean", wrench_mean.clone())
        self.register_buffer("_wrench_std", wrench_std.clone())
        if seal_mean is not None:
            self.register_buffer("_seal_mean", seal_mean.clone())
            self.register_buffer("_seal_std", seal_std.clone())
        if dfmag_mean is not None:
            self.register_buffer("_dfmag_mean", dfmag_mean.clone())
            self.register_buffer("_dfmag_std", dfmag_std.clone())
        # runtime eval hyperparameters — non-persistent, same policy as film_contact.py
        for name, val in [("_contact_F0", contact_F0), ("_contact_tau", contact_tau),
                          ("_fz_tau", fz_tau), ("_fz_off", fz_off), ("_fmag_off", fmag_off),
                          ("_fmag_tau", fmag_tau), ("_dfmag_tau", dfmag_tau)]:
            self.register_buffer(name, torch.tensor(float(val)), persistent=False)
        self._cur_contact = None

    def new_embed_suffix(self, state, noisy_actions, timestep):
        # c-hat from the UNMASKED normalized state, exactly like film_contact's embed_prefix.
        # Instance opt-out (_film_cond falsy) keeps a vanilla policy built after the class
        # patch — e.g. a naive baseline in the same process — un-masked and un-conditioned.
        if state is not None and getattr(self, "_film_cond", None):
            c = _condition_from_state(self, state)
            if _CFG["variant"] == "v1" and self.training:   # decorrelated control
                c = c[torch.randperm(c.shape[0], device=c.device)]
            self._cur_contact = c
            if _CFG["mask_force"]:
                state = state.clone()                       # c was already read above
                if "contact" in self._film_cond or "fmag" in self._film_cond:
                    state[..., WRENCH_LO:WRENCH_HI] = 0.0
                if "fz" in self._film_cond:
                    state[..., FZ_IDX] = 0.0
                if "seal" in self._film_cond:
                    state[..., SEAL_IDX] = 0.0
                if "dfmag" in self._film_cond and state.shape[-1] > DFMAG_IDX:
                    state[..., DFMAG_IDX] = 0.0
        else:
            self._cur_contact = None

        embs, pad_masks, att_masks, adarms_cond = orig_suffix(
            self, state, noisy_actions, timestep)
        c = self._cur_contact
        if c is not None:
            c = c.to(embs.device)
            if _CFG["inject"] == "state":
                # embed_suffix concatenates [state_token, action_tokens]: index 0 is the
                # state token, so modulating embs[:, :1] conditions exactly the token
                # state_proj produced — the π0 analogue of SmolVLA's state-token hook.
                head = self.contact_film(embs[:, :1, :], c)
                embs = torch.cat([head, embs[:, 1:, :]], dim=1)
            else:                                    # 'action': the chunk_size action tokens
                tail = self.contact_film(embs[:, 1:, :], c)
                embs = torch.cat([embs[:, :1, :], tail], dim=1)
        return embs, pad_masks, att_masks, adarms_cond

    PI0Pytorch.__init__ = new_init
    PI0Pytorch.embed_suffix = new_embed_suffix
    PI0Pytorch._film_patched = True
