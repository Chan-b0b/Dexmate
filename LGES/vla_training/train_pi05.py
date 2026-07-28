#!/usr/bin/env python3
"""lerobot-train wrapper for π0.5 on lerobot 0.5.1.

The HF `lerobot/pi05_base` preprocessor was saved by a NEWER lerobot whose registry
has a 'relative_actions_processor' step; 0.5.1 doesn't. In this checkpoint the step
is saved with enabled=false (a no-op), so we register a same-name passthrough shim
before the pipeline loads. Aborts if a checkpoint ever sets enabled=true.

  python train_pi05.py --policy.path=lerobot/pi05_base --dataset.repo_id=... ...
Also imported by film-pi05 launchers for the same shim.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.processor import ProcessorStep, ProcessorStepRegistry  # noqa: E402


@ProcessorStepRegistry.register(name="relative_actions_processor")
@dataclass
class _RelativeActionsShim(ProcessorStep):
    enabled: bool = False
    exclude_joints: list = field(default_factory=list)
    action_names: object = None

    def __call__(self, transition):
        if self.enabled:
            raise NotImplementedError(
                "relative_actions_processor shim is passthrough-only (enabled=false); "
                "this checkpoint wants enabled=true — upgrade lerobot instead.")
        return transition

    def get_config(self):
        return {"enabled": self.enabled, "exclude_joints": self.exclude_joints,
                "action_names": self.action_names}

    def transform_features(self, features):
        return features


if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import train
    train()
