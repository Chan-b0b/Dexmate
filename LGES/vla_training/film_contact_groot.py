"""FiLM condition-conditioning port for lerobot GR00T N1.5 — descend-until-contact,
state-token injection at the action head's state_encoder output.

Architecture mapping vs the other ports (2026-08-14):
  - GR00T N1.5 = Eagle VLM backbone (frozen by default) + flow-matching DiT action head.
    The head embeds state via `state_encoder` (CategorySpecificMLP, (B,1,64) ->
    (B,1,input_embedding_dim=1536)) and concatenates [state_token, future_tokens,
    action_tokens] into the DiT sequence — so state_encoder's output IS a real dense
    state token, the direct analogue of SmolVLA-'prefix' / π0-'state' / ACT-encoder.
    FiLM hooks that output. No 'action' arm here (π0 already answered the token question);
    this port tests whether the state-token result carries to a 4th architecture.
  - GR00T min-max normalizes state: x_norm = 2*(x-min)/(max-min)-1, per-dim, inside the
    groot_pack_inputs_v3 processor step (NOT mean/std) — so c-hat un-normalizes with the
    dataset's min/max vectors (load_state_minmax), like the pi05 port did with quantiles.
  - c-hat + mask_force live at the POLICY level (GrootPolicy.forward /
    predict_action_chunk), where batch['state'] is the packed normalized (B,1,64) tensor —
    same design as film_contact_pi05/act. mask_force zeroes the conditioned dims in
    NORMALIZED space (0 = the min-max midpoint, a constant carrying no information),
    the same convention as the pi05 quantile port.
  - Probes: forced-c must hook film_contact_groot._cond_from_state (module-level, resolved
    at call time) — patching film_contact alone is a silent no-op, as with pi0/pi05.

Usage (BEFORE from_pretrained / policy construction):
  import film_contact_groot as fcg
  mn, mx = fcg.load_state_minmax(dataset_root)
  fcg.apply('v2', mn, mx, cond=('contact', 'fz', 'seal'), mask_force=True)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from film_contact import (ContactFiLM, _canon, WRENCH_LO, WRENCH_HI, FZ_IDX, SEAL_IDX,
                          DFMAG_IDX)

INJECTS = ("state", "layers")
_CFG = {"variant": "v0", "mask_force": True, "cond": ("contact", "fz", "seal"),
        "inject": "state"}


def load_state_minmax(dataset_root):
    """(state_min, state_max) float32 tensors (D,) from a lerobot dataset stats.json —
    the SAME stats groot_pack_inputs_v3 normalizes with."""
    st = json.loads((Path(dataset_root) / "meta" / "stats.json").read_text())["observation.state"]
    return (torch.tensor(st["min"], dtype=torch.float32),
            torch.tensor(st["max"], dtype=torch.float32))


def _raw_state(policy, state_norm: torch.Tensor) -> torch.Tensor:
    """Invert groot's min-max map on the real (un-padded) dims: (B,1,64) -> raw (B,D)."""
    mn = policy._film_min.to(state_norm.device)
    mx = policy._film_max.to(state_norm.device)
    n = mn.shape[-1]
    x = state_norm[:, 0, :n].float()
    return (x + 1.0) * (mx - mn) / 2.0 + mn


def _cond_from_state(policy, state_norm: torch.Tensor) -> torch.Tensor:
    """c-hat (B, cond_dim) from the packed NORMALIZED state. Same channel formulas as
    film_contact, computed on the min-max-un-normalized RAW state."""
    raw = _raw_state(policy, state_norm)                       # (B, D) raw units
    fmag = torch.linalg.norm(raw[:, WRENCH_LO:WRENCH_LO + 3], dim=-1, keepdim=True)
    cols = {}
    cols["contact"] = torch.clamp((fmag - policy._contact_F0) / policy._contact_tau, 0.0, 1.0)
    cols["fz"] = (raw[:, FZ_IDX:FZ_IDX + 1] - policy._fz_off) / policy._fz_tau
    cols["fmag"] = (fmag - policy._fmag_off) / policy._fmag_tau
    cols["seal"] = torch.clamp(raw[:, SEAL_IDX:SEAL_IDX + 1], 0.0, 1.0)
    if "dfmag" in policy._film_cond:
        cols["dfmag"] = raw[:, DFMAG_IDX:DFMAG_IDX + 1] / policy._dfmag_tau
    return torch.cat([cols[ch] for ch in policy._film_cond], dim=-1)


def apply(variant: str, state_min: torch.Tensor, state_max: torch.Tensor,
          cond=("contact", "fz", "seal"), contact_F0: float = 6.0, contact_tau: float = 4.0,
          fz_tau: float = 5.0, fz_off: float = 2.6, fmag_off: float = 5.1,
          fmag_tau: float = 5.0, dfmag_tau: float = 5.0, mask_force: bool = True,
          inject: str = "state") -> None:
    """Patch GrootPolicy (c-hat + mask_force at the batch level; FiLM on the action head).
    inject='state' hooks state_encoder's output (the state token); inject='layers' gates
    EVERY DiT block's feed-forward branch (block.ff output, pre-residual) with one
    zero-init ContactFiLM per block — the architecture-generic per-layer gate shared with
    film_contact.py/film_contact_pi0.py inject='layers'. Idempotent like film_contact.apply."""
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    cond = _canon(cond)
    if inject not in INJECTS:
        raise ValueError(f"unknown inject {inject!r}; allowed: {INJECTS}")
    if "dfmag" in cond and state_min.shape[-1] <= DFMAG_IDX:
        raise ValueError("cond includes 'dfmag' but the dataset state has no dfmag dim")
    _CFG.update(variant=variant, mask_force=mask_force, cond=cond, inject=inject)
    if getattr(GrootPolicy, "_film_patched", False):
        return

    orig_init = GrootPolicy.__init__

    def new_init(self, config, **kwargs):
        orig_init(self, config, **kwargs)
        self._film_cond = cond
        head = self._groot_model.action_head
        if inject == "layers":  # one zero-init film per DiT block (ff branch width = blk.dim)
            self.contact_film = nn.ModuleList(
                [ContactFiLM(blk.dim, cond_dim=len(cond))
                 for blk in head.model.transformer_blocks])
        else:
            self.contact_film = ContactFiLM(head.config.input_embedding_dim, cond_dim=len(cond))
        self.register_buffer("_film_min", state_min.clone())
        self.register_buffer("_film_max", state_max.clone())
        for name, val in [("_contact_F0", contact_F0), ("_contact_tau", contact_tau),
                          ("_fz_tau", fz_tau), ("_fz_off", fz_off), ("_fmag_off", fmag_off),
                          ("_fmag_tau", fmag_tau), ("_dfmag_tau", dfmag_tau)]:
            # non-persistent: runtime eval hyperparameters, not learned values
            self.register_buffer(name, torch.tensor(float(val)), persistent=False)
        self._cur_contact = None
        owner = self  # closure ref (NOT a submodule -> no state_dict recursion)

        if inject == "layers":
            # Per-block gate on the DiT feed-forward branch: block.ff's output, BEFORE the
            # residual add — the same pre-residual MLP-branch spec as SmolVLA/π0 'layers'.
            # DiT.forward calls each BasicTransformerBlock (and its .ff) as a module, so
            # hooks fire in both training and the denoise inference loop.
            def _mk_ff_hook(idx):
                def _ff_film_hook(_module, _inp, out):
                    c = owner._cur_contact
                    if c is None:
                        return out
                    return owner.contact_film[idx](out, c.to(out.device))
                return _ff_film_hook
            for i, blk in enumerate(head.model.transformer_blocks):
                blk.ff.register_forward_hook(_mk_ff_hook(i))
        else:
            def _state_film_hook(_module, _inp, out):
                c = owner._cur_contact
                if c is None:
                    return out
                return owner.contact_film(out, c.to(out.device))  # (B,1,H) -> film -> (B,1,H)

            head.state_encoder.register_forward_hook(_state_film_hook)

    GrootPolicy.__init__ = new_init

    def _set_cond_and_mask(policy, batch, training: bool):
        st = batch.get("state")
        if st is None or not getattr(policy, "_film_cond", None):
            policy._cur_contact = None
            return batch
        c = _cond_from_state(policy, st)                       # from the UNMASKED state
        if _CFG["variant"] == "v1" and training:               # decorrelated control
            c = c[torch.randperm(c.shape[0], device=c.device)]
        policy._cur_contact = c
        if _CFG["mask_force"]:  # bottleneck: the force reaches the model ONLY via c-hat
            st = st.clone()
            if "contact" in policy._film_cond or "fmag" in policy._film_cond:
                st[..., WRENCH_LO:WRENCH_HI] = 0.0
            if "fz" in policy._film_cond:
                st[..., FZ_IDX] = 0.0
            if "seal" in policy._film_cond:
                st[..., SEAL_IDX] = 0.0
            if "dfmag" in policy._film_cond:
                st[..., DFMAG_IDX] = 0.0
            batch = dict(batch)
            batch["state"] = st
        return batch

    orig_forward = GrootPolicy.forward
    orig_predict = GrootPolicy.predict_action_chunk

    def new_forward(self, batch):
        batch = _set_cond_and_mask(self, batch, self.training)
        return orig_forward(self, batch)

    def new_predict(self, batch):
        batch = _set_cond_and_mask(self, batch, False)
        return orig_predict(self, batch)

    GrootPolicy.forward = new_forward
    GrootPolicy.predict_action_chunk = new_predict
    GrootPolicy._film_patched = True
