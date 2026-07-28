"""FiLM condition-conditioning port for lerobot PI05 (π0.5) — descend-until-contact.

Architecture differences vs the SmolVLA patch (film_contact.py):
  - π0.5 has NO continuous state token: observation.state is QUANTILE-normalized to
    [-1,1] and DISCRETIZED (256 bins) into the language prompt by
    Pi05PrepareStateTokenizerProcessorStep. The model forward never sees the state
    tensor -> c-hat is computed at the POLICY level (PI05Policy.forward /
    predict_action_chunk read batch['observation.state']) and stashed on the model.
  - inject: 'suffix' ONLY (action-token embeddings at the expert input, all layers).
    There is no state_proj/state token, so SmolVLA's winning 'prefix' point has no
    analogue here.
  - mask_force: wrench (+conditioned) dims are zeroed inside the TOKENIZER step's
    local copy, so the discretized prompt carries no force info while the batch's
    observation.state stays intact for c-hat.
  - un-normalization is quantile-based: raw = (norm+1)/2 * (q99-q01) + q01.

Channel formulas and calibration defaults are film_contact.py's (contact rise,
fz-2.6, fmag-5.1; env-overridable in train_film.py-style launchers).

Usage (BEFORE from_pretrained):
  import film_contact_pi05 as fcp
  q01, q99 = fcp.load_state_quantiles(dataset_root)
  fcp.apply('v2', q01, q99, cond=('contact','fz','seal'), mask_force=True, ...)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from film_contact import (ContactFiLM, _canon, WRENCH_LO, WRENCH_HI, FZ_IDX, SEAL_IDX,
                          DFMAG_IDX)

_CFG = {"variant": "v0", "mask_force": True, "cond": ("contact", "fz", "seal")}


def load_state_quantiles(dataset_root):
    """(q01, q99) float32 tensors over observation.state — the quantile normalizer's
    parameters, needed to recover raw Newtons from the [-1,1]-normalized batch state."""
    st = json.loads((Path(dataset_root) / "meta" / "stats.json").read_text())["observation.state"]
    return (torch.tensor(st["q01"], dtype=torch.float32),
            torch.tensor(st["q99"], dtype=torch.float32))


def _cond_from_state(model, state_norm: torch.Tensor) -> torch.Tensor:
    """c-hat (B, cond_dim) from the QUANTILE-normalized batch state."""
    q01, q99 = model._film_q01, model._film_q99
    n = state_norm.shape[-1]
    raw = (state_norm + 1.0) / 2.0 * (q99[:n] - q01[:n]) + q01[:n]
    cols = []
    for ch in model._film_cond:
        if ch == "contact":
            fmag = torch.linalg.norm(raw[..., WRENCH_LO:WRENCH_LO + 3], dim=-1, keepdim=True)
            cols.append(torch.clamp((fmag - model._contact_F0) / model._contact_tau, 0.0, 1.0))
        elif ch == "fz":
            cols.append((raw[..., FZ_IDX:FZ_IDX + 1] - model._fz_off) / model._fz_tau)
        elif ch == "fmag":
            fmag = torch.linalg.norm(raw[..., WRENCH_LO:WRENCH_LO + 3], dim=-1, keepdim=True)
            cols.append((fmag - model._fmag_off) / model._fmag_tau)
        elif ch == "seal":
            cols.append(torch.clamp(raw[..., SEAL_IDX:SEAL_IDX + 1], 0.0, 1.0))
        elif ch == "dfmag":
            cols.append(raw[..., DFMAG_IDX:DFMAG_IDX + 1] / model._dfmag_tau)
    return torch.cat(cols, dim=-1)


def apply(variant: str, q01: torch.Tensor, q99: torch.Tensor,
          cond=("contact", "fz", "seal"), contact_F0: float = 6.0, contact_tau: float = 4.0,
          fz_tau: float = 5.0, fz_off: float = 2.6, fmag_off: float = 5.1,
          fmag_tau: float = 5.0, dfmag_tau: float = 5.0, mask_force: bool = True) -> None:
    """Patch PI05FlowMatching (suffix FiLM) + PI05Policy (c-hat from batch state) +
    the state tokenizer step (mask_force). Idempotent like film_contact.apply."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, PI05Policy
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.utils.constants import OBS_STATE

    cond = _canon(cond)
    _CFG.update(variant=variant, mask_force=mask_force, cond=cond)
    if getattr(PI05Pytorch, "_film_patched", False):
        return

    orig_init = PI05Pytorch.__init__
    orig_suffix = PI05Pytorch.embed_suffix

    def new_init(self, *a, **k):
        orig_init(self, *a, **k)
        self._film_cond = cond
        self.contact_film = ContactFiLM(self.action_in_proj.out_features, cond_dim=len(cond))
        self.register_buffer("_film_q01", q01.clone())
        self.register_buffer("_film_q99", q99.clone())
        # runtime eval hyperparameters — non-persistent, same policy as film_contact.py
        for name, val in [("_contact_F0", contact_F0), ("_contact_tau", contact_tau),
                          ("_fz_tau", fz_tau), ("_fz_off", fz_off), ("_fmag_off", fmag_off),
                          ("_fmag_tau", fmag_tau), ("_dfmag_tau", dfmag_tau)]:
            self.register_buffer(name, torch.tensor(float(val)), persistent=False)
        self._cur_contact = None

    def new_embed_suffix(self, noisy_actions, timestep):
        # pi05's embed_suffix returns (embs, pad_masks, att_masks, adarms_cond)
        embs, pad_masks, att_masks, adarms_cond = orig_suffix(self, noisy_actions, timestep)
        if self._cur_contact is not None:
            embs = self.contact_film(embs, self._cur_contact.to(embs.device))
        return embs, pad_masks, att_masks, adarms_cond

    def _set_cond(policy, batch, training: bool):
        st = batch.get(OBS_STATE)
        if st is None:
            policy.model._cur_contact = None
            return
        c = _cond_from_state(policy.model, st)
        if _CFG["variant"] == "v1" and training:   # decorrelated control
            c = c[torch.randperm(c.shape[0], device=c.device)]
        policy.model._cur_contact = c

    orig_forward = PI05Policy.forward
    orig_predict = PI05Policy.predict_action_chunk

    def new_forward(self, batch, reduction="mean"):
        _set_cond(self, batch, self.training)
        return orig_forward(self, batch, reduction=reduction)

    def new_predict(self, batch, **kw):
        _set_cond(self, batch, False)
        return orig_predict(self, batch, **kw)

    # mask_force: zero conditioned dims in the tokenizer step's LOCAL state copy so
    # the discretized prompt loses the force info; batch state stays intact for c-hat.
    orig_step = Pi05PrepareStateTokenizerProcessorStep.__call__

    def new_step(self, transition):
        if _CFG["mask_force"]:
            obs = transition.get(TransitionKey.OBSERVATION, {}) if TransitionKey else None
            st = obs.get(OBS_STATE) if obs else None
            if st is not None:
                st = st.clone()
                if "contact" in cond or "fmag" in cond:
                    st[..., WRENCH_LO:WRENCH_HI] = 0.0
                if "fz" in cond:
                    st[..., FZ_IDX] = 0.0
                if "seal" in cond:
                    st[..., SEAL_IDX] = 0.0
                if "dfmag" in cond and st.shape[-1] > DFMAG_IDX:
                    st[..., DFMAG_IDX] = 0.0
                obs = dict(obs)
                obs[OBS_STATE] = st
                transition = dict(transition)
                transition[TransitionKey.OBSERVATION] = obs
        return orig_step(self, transition)

    try:
        from lerobot.processor import TransitionKey
    except ImportError:
        from lerobot.processor.pipeline import TransitionKey  # lerobot 0.5.x layout

    PI05Pytorch.__init__ = new_init
    PI05Pytorch.embed_suffix = new_embed_suffix
    PI05Policy.forward = new_forward
    PI05Policy.predict_action_chunk = new_predict
    Pi05PrepareStateTokenizerProcessorStep.__call__ = new_step
    PI05Pytorch._film_patched = True
