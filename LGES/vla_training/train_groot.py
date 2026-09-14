#!/usr/bin/env python3
"""lerobot-train wrapper for GR00T N1.5 on this box (no flash-attn).

The vendored Eagle backbone config pins _attn_implementation=flash_attention_2 (and
modeling_eagle2_5_vl.py hard-codes it again for the SigLIP vision tower), so transformers
hard-fails at model construction when flash_attn isn't importable. This box has no CUDA
toolkit (no nvcc) and no FA2 wheel exists for torch 2.10+cu130/cp312, so instead we patch
transformers' attn-implementation check to FALL BACK TO SDPA whenever flash-attn is the
only problem. sdpa computes the same attention (different kernels); the Eagle backbone is
FROZEN in this port (tune_llm/visual=False), so this cannot change what is trained —
only feature-extraction numerics at the kernel level.

  python train_groot.py --policy.type=groot --dataset.repo_id=... ...
Also imported by train_film_groot.py / select_best_ckpt.py for the same patch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import modeling_utils as _mu  # noqa: E402

_orig_check = _mu.PreTrainedModel._check_and_adjust_attn_implementation


def _sdpa_fallback_check(self, attn_implementation, *args, **kwargs):
    try:
        return _orig_check(self, attn_implementation, *args, **kwargs)
    except ImportError as e:
        if "flash" not in str(e).lower():
            raise
        print(f"[groot-shim] flash-attn unavailable -> sdpa for {type(self).__name__}",
              file=sys.stderr)
        return _orig_check(self, "sdpa", *args, **kwargs)


if not getattr(_mu.PreTrainedModel, "_sdpa_fallback_patched", False):
    _mu.PreTrainedModel._check_and_adjust_attn_implementation = _sdpa_fallback_check
    _mu.PreTrainedModel._sdpa_fallback_patched = True

# Second shim: transformers 5.x runs from_pretrained module construction under a META
# device context, and the action head builds torch.distributions.Beta in __init__ —
# whose arg validation calls Tensor.item() (impossible on meta), and which, being no
# nn.Module, would never be materialized by the weight load anyway. Force it onto CPU;
# sample_time() already moves samples to the model device afterwards.
import torch  # noqa: E402
from lerobot.policies.groot.action_head import flow_matching_action_head as _fmah  # noqa: E402

_OrigBeta = _fmah.Beta


class _CpuBeta(_OrigBeta):
    # device AND dtype must be explicit: construction happens under transformers'
    # meta-device context with default dtype set to bf16, and torch._sample_dirichlet
    # is not implemented for BFloat16.
    def __init__(self, concentration1, concentration0, validate_args=None):
        super().__init__(
            torch.as_tensor(float(concentration1), dtype=torch.float32, device="cpu"),
            torch.as_tensor(float(concentration0), dtype=torch.float32, device="cpu"),
            validate_args)


if _fmah.Beta is not _CpuBeta:
    _fmah.Beta = _CpuBeta

# Third shim: the vendored GR00TN15.__init__ (written for older transformers) never calls
# post_init(), but transformers 5.3's from_pretrained finalization reads
# `all_tied_weights_keys`, which post_init() would have set. Register it ourselves —
# GR00TN15 ties nothing at the top level, so the expanded mapping (or {}) is correct.
from lerobot.policies.groot.groot_n1 import GR00TN15 as _G15  # noqa: E402

if not getattr(_G15, "_tied_keys_patched", False):
    _orig_g15_init = _G15.__init__

    def _g15_init_with_tied_keys(self, *args, **kwargs):
        _orig_g15_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            try:
                self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
            except Exception:
                self.all_tied_weights_keys = {}

    _G15.__init__ = _g15_init_with_tied_keys
    _G15._tied_keys_patched = True

# Fourth shim: transformers 5.x's ProcessorMixin._merge_kwargs no longer propagates a
# flat `return_tensors=` into the per-modality kwargs, so the vendored Eagle processor's
# INNER image_processor call returns a python list and `pixel_values.shape` crashes
# (processing_eagle2_5_vl.py:225). Restore the 4.x propagation, setdefault-only.
from transformers import processing_utils as _pu  # noqa: E402

_orig_merge = _pu.ProcessorMixin._merge_kwargs


def _merge_kwargs_with_rt(self, ModelProcessorKwargs, tokenizer_init_kwargs=None, **kwargs):
    out = _orig_merge(self, ModelProcessorKwargs,
                      tokenizer_init_kwargs=tokenizer_init_kwargs, **kwargs)
    rt = kwargs.get("return_tensors") or kwargs.get("common_kwargs", {}).get("return_tensors")
    if rt is not None:
        for mod in ("text_kwargs", "images_kwargs", "videos_kwargs"):
            out.setdefault(mod, {}).setdefault("return_tensors", rt)
    return out


if not getattr(_pu.ProcessorMixin, "_rt_merge_patched", False):
    _pu.ProcessorMixin._merge_kwargs = _merge_kwargs_with_rt
    _pu.ProcessorMixin._rt_merge_patched = True

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import train
    train()
