"""FiLM condition-conditioning for SmolVLA — learned descend-until-contact (V1/V2).

See Research/condition_driven/DESCEND_UNTIL_CONTACT_DESIGN.md.

Monkey-patches lerobot 0.5.1 `VLAFlowMatching` (installed in the venv, NOT edited) to:
  - build a condition vector c-hat from the observation.state. Channels are configurable
    (`cond=`); currently two are available:
        'contact' : clip((|F[:3]| - F0) / tau, 0, 1)   from the un-normalized wrench (idx 9:15)
        'seal'    : the vacuum_sealed bit (idx 8), un-normalized to ~{0,1}
    stats (wrench / seal mean,std) are baked into buffers so c-hat is identical train+test.
  - optionally FORCE-MASK the conditioned dims out of the state that feeds the action expert
    (the bottleneck: the condition reaches the action ONLY via c-hat) — toggle with mask_force.
  - FiLM-modulate the action-expert features (the input to action_out_proj) on c-hat, via a
    forward_pre_hook -> no forward()/denoise_step() replication, and the base-checkpoint keys
    are preserved (contact_film is the only new param group).

Variants:
  'v2' : true c-hat (the method).
  'v1' : c-hat shuffled across the batch at TRAIN time (decorrelated control: same
         capacity + mechanism, grounding removed).
  'v0' : no-op (use the vanilla checkpoint; do not call apply()).

Usage (call BEFORE the policy is built, i.e. before make_policy / from_pretrained):
  import film_contact
  wm, ws = film_contact.load_wrench_stats(dataset_root)
  sm, ss = film_contact.load_seal_stats(dataset_root)
  film_contact.apply('v2', wm, ws, seal_mean=sm, seal_std=ss, cond=('contact', 'seal'))
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

WRENCH_LO, WRENCH_HI = 9, 15   # wrench (fx,fy,fz,tx,ty,tz) dims in observation.state
SEAL_IDX = 8                   # vacuum_sealed bit in observation.state
CHANNELS = ("contact", "seal")  # available condition channels (canonical order)
_CFG = {"variant": "v0", "mask_force": True, "cond": ("contact", "seal"),
        "inject": "suffix"}  # mutated by apply()
INJECTS = ("suffix", "output")  # where FiLM modulates the action features


def load_wrench_stats(dataset_root):
    """(wrench_mean, wrench_std) float32 tensors (6,) from a lerobot dataset stats.json."""
    st = json.loads((Path(dataset_root) / "meta" / "stats.json").read_text())["observation.state"]
    m = torch.tensor(st["mean"][WRENCH_LO:WRENCH_HI], dtype=torch.float32)
    sd = torch.tensor(st["std"][WRENCH_LO:WRENCH_HI], dtype=torch.float32)
    return m, sd


def load_seal_stats(dataset_root):
    """(seal_mean, seal_std) float32 scalar tensors from a lerobot dataset stats.json."""
    st = json.loads((Path(dataset_root) / "meta" / "stats.json").read_text())["observation.state"]
    return (torch.tensor(st["mean"][SEAL_IDX], dtype=torch.float32),
            torch.tensor(st["std"][SEAL_IDX], dtype=torch.float32))


def _canon(cond):
    """Validate + return cond channels in canonical CHANNELS order (deduped)."""
    bad = [c for c in cond if c not in CHANNELS]
    if bad:
        raise ValueError(f"unknown FiLM cond channels {bad}; allowed: {CHANNELS}")
    out = tuple(c for c in CHANNELS if c in cond)
    if not out:
        raise ValueError("FiLM cond must contain >=1 channel")
    return out


class ContactFiLM(nn.Module):
    """c-hat (B, cond_dim) -> (gamma, beta) modulating features (B,T,H).
    Final layers zero-init -> identity at start, so a finetune begins exactly at the
    base policy's behavior and then learns the modulation."""

    def __init__(self, hidden: int, cond_dim: int = 1, hdim: int = 64):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(cond_dim, hdim), nn.SiLU(), nn.Linear(hdim, hidden))
        self.shift = nn.Sequential(nn.Linear(cond_dim, hdim), nn.SiLU(), nn.Linear(hdim, hidden))
        for mlp in (self.scale, self.shift):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = c.float()
        g = self.scale(c)[:, None, :]
        b = self.shift(c)[:, None, :]
        return (x.float() * (1.0 + g) + b).to(x.dtype)


def _contact_from_state(self, state: torch.Tensor) -> torch.Tensor:
    """contact c-hat in [0,1], shape (B,1), from the NORMALIZED wrench in `state` (B, >=15)."""
    w = state[..., WRENCH_LO:WRENCH_HI] * self._wrench_std + self._wrench_mean  # un-normalize
    fmag = torch.linalg.norm(w[..., :3], dim=-1, keepdim=True)                  # raw |F| (B,1)
    return torch.clamp((fmag - self._contact_F0) / self._contact_tau, 0.0, 1.0)


def _seal_from_state(self, state: torch.Tensor) -> torch.Tensor:
    """seal bit in [0,1], shape (B,1), un-normalized from the NORMALIZED state."""
    seal = state[..., SEAL_IDX:SEAL_IDX + 1] * self._seal_std + self._seal_mean
    return torch.clamp(seal, 0.0, 1.0)


def _condition_from_state(self, state: torch.Tensor) -> torch.Tensor:
    """Concatenate the enabled channels (canonical order) -> c-hat (B, cond_dim)."""
    cols = [(_contact_from_state if ch == "contact" else _seal_from_state)(self, state)
            for ch in self._film_cond]
    return torch.cat(cols, dim=-1)


def apply(variant: str, wrench_mean: torch.Tensor, wrench_std: torch.Tensor,
          seal_mean: torch.Tensor = None, seal_std: torch.Tensor = None,
          cond=("contact", "seal"), contact_F0: float = 14.0, contact_tau: float = 3.0,
          mask_force: bool = True, inject: str = "suffix") -> None:
    """Patch VLAFlowMatching for the given variant + condition channels.

    cond: which channels feed FiLM — any subset of CHANNELS ('contact', 'seal').
    inject: WHERE FiLM modulates —
      'suffix' : the action-token embedding at the EXPERT INPUT (embed_suffix) — every expert
                 layer computes the action conditioned on c-hat (strong authority).
      'output' : the input to action_out_proj (the final projection) — a weak last-layer tap.
    mask_force=False keeps the conditioned dims in the action path (FiLM added ON TOP of the
    existing signal) — an ablation to isolate the effect of adding FiLM before committing to
    the bottleneck. When True, the conditioned dims (wrench for 'contact', seal for 'seal')
    are zeroed from the action path so the condition reaches the action ONLY via c-hat.

    Idempotent: re-calling only swaps variant/mask_force in _CFG. `cond` and `inject` (the FiLM
    input width, buffers, and hook placement) are STRUCTURAL — fixed at the first call, before
    the model is built. NOTE: eval must use the SAME cond + inject + mask_force the checkpoint
    was trained with."""
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    cond = _canon(cond)
    if inject not in INJECTS:
        raise ValueError(f"unknown inject {inject!r}; allowed: {INJECTS}")
    if "seal" in cond and (seal_mean is None or seal_std is None):
        raise ValueError("cond includes 'seal' but seal_mean/seal_std were not provided")

    _CFG["variant"] = variant
    _CFG["mask_force"] = mask_force
    _CFG["cond"] = cond
    _CFG["inject"] = inject
    if getattr(VLAFlowMatching, "_film_patched", False):
        return

    orig_init = VLAFlowMatching.__init__
    orig_embed_prefix = VLAFlowMatching.embed_prefix
    orig_embed_suffix = VLAFlowMatching.embed_suffix

    def new_init(self, *a, **k):
        orig_init(self, *a, **k)
        hidden = self.vlm_with_expert.expert_hidden_size
        self._film_cond = cond
        self.contact_film = ContactFiLM(hidden, cond_dim=len(cond))
        if "contact" in cond:
            self.register_buffer("_wrench_mean", wrench_mean.clone())
            self.register_buffer("_wrench_std", wrench_std.clone())
            self.register_buffer("_contact_F0", torch.tensor(float(contact_F0)))
            self.register_buffer("_contact_tau", torch.tensor(float(contact_tau)))
        if "seal" in cond:
            self.register_buffer("_seal_mean", seal_mean.clone())
            self.register_buffer("_seal_std", seal_std.clone())
        self._cur_contact = None
        owner = self  # closure ref (NOT a submodule -> not in state_dict, no recursion)

        if inject == "output":   # weak last-layer tap (kept for the injection-point ablation)
            def _film_pre_hook(_module, inp):
                c = owner._cur_contact
                if c is None:
                    return None
                return (owner.contact_film(inp[0], c),)
            self.action_out_proj.register_forward_pre_hook(_film_pre_hook)

    def new_embed_suffix(self, noisy_actions, timestep):
        embs, pad_masks, att_masks = orig_embed_suffix(self, noisy_actions, timestep)
        if inject == "suffix" and self._cur_contact is not None:  # condition the action tokens
            embs = self.contact_film(embs, self._cur_contact)     # at the expert input
        return embs, pad_masks, att_masks

    def new_embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        if state is not None:
            c = _condition_from_state(self, state)                # from the UNMASKED state
            if _CFG["variant"] == "v1" and self.training:         # decorrelate (control)
                c = c[torch.randperm(c.shape[0], device=c.device)]
            self._cur_contact = c
            if _CFG["mask_force"]:                                 # bottleneck: mask conditioned dims
                state = state.clone()
                if "contact" in self._film_cond:
                    state[..., WRENCH_LO:WRENCH_HI] = 0.0
                if "seal" in self._film_cond:
                    state[..., SEAL_IDX] = 0.0
        return orig_embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=state)

    VLAFlowMatching.__init__ = new_init
    VLAFlowMatching.embed_prefix = new_embed_prefix
    VLAFlowMatching.embed_suffix = new_embed_suffix
    VLAFlowMatching._film_patched = True
