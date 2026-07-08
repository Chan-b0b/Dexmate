"""Barcode scan + divert decision for ik_demo.

Scan-during-descent: a BackgroundScanner (drivers.bcr) fires reads in a daemon
thread while the suction arm descends onto a battery; after the seal, the agreed
code is checked against cfg.TARGET_BARCODES. A match diverts the battery to the
right-hand gripper (gripper.py) instead of the case slot.

No scan-gate / spiral search — scanning happens during the normal pick descent.
"""

from __future__ import annotations

try:
    from . import config as cfg
    from .drivers.bcr import BackgroundScanner  # noqa: F401 (re-exported)
except ImportError:  # allow running a module directly from ik_demo/
    import config as cfg
    from drivers.bcr import BackgroundScanner  # noqa: F401


def is_target(code: str | None) -> bool:
    """True if a decoded barcode should be diverted to the gripper."""
    return code is not None and code in cfg.TARGET_BARCODES
