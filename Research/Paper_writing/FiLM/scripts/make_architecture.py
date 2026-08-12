#!/usr/bin/env python3
"""Fig: conditioned-policy architecture (paper §IV, fig:arch).

08-12 user decision: the mask is NOT depicted — the wrench is drawn as not
entering the state at all; c-hat through FiLM is the only force pathway.
(§IV text still explains the mask mechanism.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = "/home/maverick/Humanoid/Dexmate/Research/Paper_writing/FiLM/paper/figs"

INK = "#333333"
GRAY_EDGE = "#BBBBB8"
GRAY_FILL = "#F5F5F3"
BLUE = "#0072B2"
BLUE_FILL = "#E8F1F8"

fig, ax = plt.subplots(figsize=(3.5, 2.3), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis("off")

def box(x0, y0, x1, y1, text, color=False, fs=6.2):
    fc, ec = (BLUE_FILL, BLUE) if color else (GRAY_FILL, GRAY_EDGE)
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.6,rounding_size=1.6",
                 fc=fc, ec=ec, lw=0.9))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, linespacing=1.25)

def arrow(x0, y0, x1, y1, color=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=6, lw=lw,
                 color=color, shrinkA=0, shrinkB=0))

# ── inputs (left) ────────────────────────────────────────────────
box(2, 52, 19, 62, "head camera\n(RGB-D)")
box(2, 40, 19, 48, "instruction")
box(2, 26, 19, 36, "state\nEE pose, suction,\nseal", fs=5.4)
box(2, 6, 16, 16, "F/T sensor\n(wrench)", color=True)

# ── force pathway (bottom, blue) ─────────────────────────────────
box(20, 6, 42, 16, "computed conditions\n$\\hat{c}$: contact, |F|, f$_z$, seal", color=True, fs=5.2)
box(45, 6, 55, 16, "FiLM\n$\\gamma,\\beta$", color=True)

arrow(16, 11, 20, 11, color=BLUE)
arrow(42, 11, 45, 11, color=BLUE)

# modulation node on the state-token path (prefix injection)
mod_x, mod_y = 50, 31
arrow(50, 16, 50, 28.6, color=BLUE)          # FiLM up to the node
ax.add_patch(Circle((mod_x, mod_y), 2.3, fc="white", ec=BLUE, lw=1.0))
ax.text(mod_x, mod_y, "$\\times$", ha="center", va="center", fontsize=7, color=BLUE)
ax.text(54.5, 34.2, "FiLM\ninjection", fontsize=5.2, color=BLUE,
        ha="center", va="bottom", linespacing=1.1)

# ── backbone and head (right) ────────────────────────────────────
box(60, 22, 78, 62, "VLM\nbackbone")
box(81, 32, 95, 50, "action\nexpert")

arrow(19, 57, 60, 57)                         # image
arrow(19, 44, 60, 44)                         # language
arrow(19, 31, 47.7, 31)                       # state -> modulation
arrow(52.3, 31, 60, 31)                       # modulated state -> VLM
arrow(78, 42, 81, 42)                         # VLM -> expert
arrow(88, 32, 88, 24)                         # expert -> action
ax.text(88, 21.5, "action chunk\n$\\Delta$pose + suction ($\\times$5)",
        ha="center", va="top", fontsize=5.2, color=INK, linespacing=1.2)

# annotation: only route
ax.text(35.5, 1.2, "the only pathway from force to action", fontsize=5.4,
        color=BLUE, ha="center", va="bottom", style="italic")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(f"{OUT}/architecture.pdf")
fig.savefig(f"{OUT}/architecture.png")
print("saved", OUT)
