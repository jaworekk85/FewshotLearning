"""An EMA teacher with explicit warmup, buffer policy, and resumable state."""
import copy
import math

import torch


def validate_teacher(config):
    if not isinstance(config.get("ema_teacher", False), bool):
        raise ValueError("ema_teacher must be boolean")
    if not config.get("ema_teacher"):
        return
    if config.get("method") != "ce_2v_pseudo_soft" or config.get("selection") != "last":
        raise ValueError("Teacher pilot requires soft SSL and last-checkpoint selection")
    decay = config.get("ema_teacher_decay", .99)
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("Invalid EMA decay")
    warmup = config.get("ema_teacher_warmup_epochs", 3)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("Invalid teacher warmup")
    if config.get("neighbor_reference") or config.get("contrastive_weight", 0) or "active_strategy" in config:
        raise ValueError("Teacher pilot isolates EMA from neighbor/contrastive changes")


class EMATeacher:
    def __init__(self, config):
        self.decay = float(config.get("ema_teacher_decay", .99))
        self.model = None
        self.updates = 0
        self.pending = None

    def initialize(self, student):
        if self.model is None:
            self.model = copy.deepcopy(student).eval().requires_grad_(False)
            if self.pending is not None:
                self.model.load_state_dict(self.pending)
                self.pending = None
        return self.model

    @torch.no_grad()
    def update(self, student):
        teacher = self.initialize(student)
        self.updates += 1
        # Starts as a true average, then caps memory at roughly 100 updates.
        alpha = min(self.decay, 1 - 1 / (self.updates + 1))
        for target, source in zip(teacher.parameters(), student.parameters()):
            target.mul_(alpha).add_(source.detach(), alpha=1-alpha)
        # Explicit policy: teacher performs eval-only forwards; no second BN fit.
        for target, source in zip(teacher.buffers(), student.buffers()):
            target.copy_(source)

    def state_dict(self):
        return dict(updates=self.updates, model=self.model.state_dict() if self.model is not None else self.pending)

    def load_state_dict(self, state):
        self.updates = state["updates"]
        if self.model is None:
            self.pending = state["model"]
        else:
            self.model.load_state_dict(state["model"])
