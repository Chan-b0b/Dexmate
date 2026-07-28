#!/usr/bin/env python3
"""lerobot-train wrapper for X-VLA on lerobot 0.5.1.

The 2toINF/X-VLA-* HF repos are transformers-format (no draccus 'type' key), so
`--policy.path=` can't parse their config. Instead run with
  --policy.type=xvla --policy.pretrained_path=2toINF/X-VLA-Pt
and this wrapper fills the CLI-built XVLAConfig's `florence_config` (vision/text
backbone spec) from the repo's own config.json, which the fresh config lacks
("vision_config is required" otherwise). Weights then load via lerobot's custom
XVLAPolicy.from_pretrained (prefix remap + skip list).

  python train_xvla.py --policy.type=xvla --policy.pretrained_path=2toINF/X-VLA-Pt ...
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

XVLA_REPO = os.environ.get("XVLA_REPO", "2toINF/X-VLA-Pt")

from huggingface_hub import hf_hub_download  # noqa: E402
from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: E402

_orig_get_florence = XVLAConfig.get_florence_config


def _patched_get_florence(self):
    fc = dict(self.florence_config) if self.florence_config else {}
    if not fc.get("vision_config") or not fc.get("text_config"):
        repo_cfg = json.load(open(hf_hub_download(XVLA_REPO, "config.json")))
        self.florence_config = repo_cfg["florence_config"]
        self._florence_config_obj = None
        print(f"[xvla] florence_config filled from {XVLA_REPO}/config.json", file=sys.stderr)
    return _orig_get_florence(self)


XVLAConfig.get_florence_config = _patched_get_florence

# lerobot 0.5.1's XVLAPolicy.from_pretrained docstring promises a 'model.' key
# remap + skip list but implements neither (TODO in source) -> strict load fails
# on the 2toINF checkpoints. Wrap the state_dict load: prefix keys with 'model.'
# and load with strict=False so new-domain heads stay randomly initialized.
import safetensors.torch as _st  # noqa: E402

_orig_load_file = _st.load_file


def _remapped_load_file(path, *a, **k):
    sd = _orig_load_file(path, *a, **k)
    if any(key.startswith("model.") for key in sd):
        return sd
    return {f"model.{key}": v for key, v in sd.items()}


# modeling_xvla does `import safetensors.torch` INSIDE from_pretrained, so patch
# the function on the safetensors.torch module itself (pass-through for any
# checkpoint whose keys already carry the 'model.' prefix).
_st.load_file = _remapped_load_file

from lerobot.policies.xvla import modeling_xvla as _mx  # noqa: E402

_orig_lsd = _mx.XVLAPolicy.load_state_dict


def _lenient_lsd(self, state_dict, strict=True, **kw):
    own = self.state_dict()
    dropped = [k for k, v in state_dict.items() if k in own and own[k].shape != v.shape]
    for k in dropped:
        state_dict.pop(k)
    if dropped:
        print(f"[xvla] dropped {len(dropped)} size-mismatched keys (stay fresh-init, e.g. "
              f"pretrain action dim != ours): {dropped[:4]}", file=sys.stderr)
    ret = _orig_lsd(self, state_dict, strict=False, **kw)
    missing = [m for m in ret.missing_keys if "soft_prompt" not in m and "action_hub" not in m]
    print(f"[xvla] load: {len(ret.missing_keys)} missing ({len(missing)} outside "
          f"soft_prompt/action_hub), {len(ret.unexpected_keys)} unexpected", file=sys.stderr)
    if len(missing) > 200:
        raise RuntimeError(f"[xvla] too many missing backbone keys ({len(missing)}) — remap wrong; "
                           f"first: {missing[:5]}")
    return ret


_mx.XVLAPolicy.load_state_dict = _lenient_lsd

# The 2toINF repo has no lerobot processor configs (policy_preprocessor.json),
# so build the processor pipeline from config DEFAULTS instead of the repo.
import lerobot.scripts.lerobot_train as _lt  # noqa: E402

_orig_mppp = _lt.make_pre_post_processors


def _mppp_no_repo(policy_cfg, pretrained_path=None, **kw):
    if pretrained_path and "X-VLA" in str(pretrained_path):
        pretrained_path = None
    return _orig_mppp(policy_cfg, pretrained_path=pretrained_path, **kw)


_lt.make_pre_post_processors = _mppp_no_repo

if __name__ == "__main__":
    _lt.train()
