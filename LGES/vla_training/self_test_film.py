"""Fast unit check of film_contact.py (no model load). Verifies the FiLM math,
identity-at-init, the contact scalar (force-DROP), and that apply() patches VLAFlowMatching."""
import torch
import film_contact as fc

# 1. ContactFiLM is identity at init (zero-init last layers), 1D and 2D cond.
H = 16
for cond_dim in (1, 2):
    film = fc.ContactFiLM(H, cond_dim=cond_dim)
    x = torch.randn(2, 5, H)
    c = torch.rand(2, cond_dim)
    y = film(x, c)
    assert y.shape == x.shape, y.shape
    assert torch.allclose(y, x, atol=1e-5), "FiLM should be identity at init"
print("[1] ContactFiLM identity-at-init OK (cond_dim 1 & 2)")

# 2. contact-from-state: graded c-hat from a normalized wrench. Contact = a force DROP
#    below the fixed baseline F0, so c^ fires when |F| FALLS (the press unloads the tool),
#    NOT when it rises. |F|~14N (free hover) -> 0; |F|~2N (pressing) -> ~1.
class _M:
    pass
m = _M()
m._wrench_mean = torch.tensor([-1.828, 0.264, 16.585, -0.118, -0.095, 0.039])
m._wrench_std = torch.tensor([1.282, 0.914, 5.895, 0.159, 0.154, 0.082])
m._contact_F0 = torch.tensor(12.0)    # threshold 2N below the ~14N baseline (deadband)
m._contact_tau = torch.tensor(10.0)   # graded contact-DROP (saturates near the full press)
m._fz_tau = torch.tensor(30.0)        # continuous fz scale (fz_raw/30)

def norm_state(fz_raw):
    raw = torch.tensor([-1.8, 0.26, fz_raw, -0.1, -0.09, 0.04])
    s = torch.zeros(1, 32)
    s[0, 9:15] = (raw - m._wrench_mean) / m._wrench_std   # normalize like the pipeline
    return s

c_rest = float(fc._contact_from_state(m, norm_state(14.0)))     # free hover -> 0
c_press = float(fc._contact_from_state(m, norm_state(2.0)))     # pressing |F|~2 -> 1 (sharp)
c_loaded = float(fc._contact_from_state(m, norm_state(20.0)))   # loaded/lift (|F| rises) -> 0
print(f"[2] contact c-hat (tau=10): |F|~14N -> {c_rest:.3f}   pressing |F|~2N -> {c_press:.3f}   "
      f"loaded |F|~20N -> {c_loaded:.3f}")
assert c_rest < 0.2 and c_press > 0.8 and c_loaded < 0.2, "contact-DROP scalar not as expected"
print("[2] contact-from-state (force DROP) OK")

# 2b. fz channel = raw fz / fz_tau (continuous, signed). Captures descent/contact/loaded by level.
fz_press = float(fc._fz_from_state(m, norm_state(2.0)))     # ~2/30
fz_descent = float(fc._fz_from_state(m, norm_state(14.0)))  # ~14/30
fz_loaded = float(fc._fz_from_state(m, norm_state(17.0)))   # ~17/30
print(f"[2b] fz channel (fz/30): press~2N -> {fz_press:.3f}  descent~14N -> {fz_descent:.3f}  "
      f"loaded~17N -> {fz_loaded:.3f}")
assert abs(fz_press - 2.0 / 30) < 1e-3 and abs(fz_descent - 14.0 / 30) < 1e-3, "fz should be fz_raw/fz_tau"
print("[2b] fz-from-state OK")

# 2c. seal channel + the full 3-channel condition vector (contact, fz, seal).
m._seal_mean = torch.tensor(0.45711)
m._seal_std = torch.tensor(0.49816)
m._film_cond = ("contact", "fz", "seal")

def with_seal(s, seal_raw):
    s = s.clone()
    s[0, 8] = (seal_raw - m._seal_mean) / m._seal_std   # normalize like the pipeline
    return s

seal_on = float(fc._seal_from_state(m, with_seal(norm_state(2.0), 1.0)))
seal_off = float(fc._seal_from_state(m, with_seal(norm_state(14.0), 0.0)))
print(f"[2c] seal: raw=1 -> {seal_on:.3f}   raw=0 -> {seal_off:.3f}")
assert seal_on > 0.9 and seal_off < 0.1, "seal not recovered to {0,1}"
cond = fc._condition_from_state(m, with_seal(norm_state(2.0), 1.0))
assert cond.shape == (1, 3), cond.shape
print(f"[2c] condition vector [contact, fz, seal] = {[round(x,3) for x in cond[0].tolist()]} OK")

# 3. apply() patches VLAFlowMatching (imports lerobot, no weights), with cond=(contact,fz,seal).
import os
from pathlib import Path
ds = os.environ.get("FILM_DATASET_ROOT",
                    str(Path(__file__).resolve().parent / "datasets/lges_case_pick_0708"))
wm, ws = fc.load_wrench_stats(ds)
sm, ss = fc.load_seal_stats(ds)
fc.apply("v2", wm, ws, seal_mean=sm, seal_std=ss, cond=("contact", "fz", "seal"))
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching  # noqa: E402
assert getattr(VLAFlowMatching, "_film_patched", False)
assert fc._CFG["cond"] == ("contact", "fz", "seal"), fc._CFG["cond"]
assert fc._CFG["inject"] == "suffix", fc._CFG["inject"]  # default injection point
assert VLAFlowMatching.embed_suffix.__name__ == "new_embed_suffix", "embed_suffix not patched"
print(f"[3] apply() patched VLAFlowMatching OK; cond={fc._CFG['cond']} inject={fc._CFG['inject']} "
      f"seal_mean={float(sm):.3f} seal_std={float(ss):.3f}")
print("ALL OK")
