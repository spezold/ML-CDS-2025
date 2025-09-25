# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from math import cos, pi

from torch.optim import Optimizer


@dataclass(frozen=True, kw_only=True)
class WarmupCosineScheduleArgs:
    steps_warmup: int
    steps_total: int
    start_lr: float
    ref_lr: float
    final_lr: float


class WarmupCosineSchedule:  # TODO: Provide property that holds or calculates new value

    def __init__(self, optimizer: Optimizer, args: WarmupCosineScheduleArgs):
        self._optimizer = optimizer
        self._args = args
        self._step = 0

    def _new_lr_at_warmup(self) -> float:
        progress = self._step / max(1, self._args.steps_warmup)
        return self._args.start_lr + progress * (self._args.ref_lr - self._args.start_lr)

    def _new_lr_after_warmup(self) -> float:
        a = self._args
        progress = (self._step - a.steps_warmup) / max(1, a.steps_total - a.steps_warmup)
        return max(a.final_lr, a.final_lr + (a.ref_lr - a.final_lr) * 0.5 * (1. + cos(pi * progress)))

    def step(self, num_steps: int = 1) -> "WarmupCosineSchedule":
        self._step += num_steps
        new_lr = self._new_lr_at_warmup() if self._step < self._args.steps_warmup else self._new_lr_after_warmup()
        for group in self._optimizer.param_groups:
            group["lr"] = new_lr
        return self


@dataclass(frozen=True, kw_only=True)
class CosineWDScheduleArgs:
    steps_total: int
    ref_wd: float
    final_wd: float


class CosineWDSchedule:  # TODO: Provide property that holds or calculates new value

    def __init__(self, optimizer: Optimizer, args: CosineWDScheduleArgs):
        self._optimizer = optimizer
        self._args = args
        self._step = 0

    def step(self, num_steps: int = 1) -> "CosineWDSchedule":
        self._step += num_steps
        a = self._args
        progress = self._step / a.steps_total
        new_wd = a.final_wd + (a.ref_wd - a.final_wd) * 0.5 * (1. + cos(pi * progress))
        new_wd = max(a.final_wd, new_wd) if a.final_wd <= a.ref_wd else min(a.final_wd, new_wd)

        for group in self._optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return self
