"""Label-only reference bank and detached, mass-preserving SSL reweighting."""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from .active_selection import js_divergence, neighbor_probabilities


DEFAULTS = dict(neighbor_reference=False, neighbor_pseudo_weighting=False,
                neighbor_warmup_epochs=3, neighbor_confidence_threshold=.95,
                neighbor_k=15, neighbor_temperature=.2, neighbor_js_scale=.2,
                neighbor_weight_floor=.25)


def validate_neighbor(config):
    cfg = {**DEFAULTS, **config}
    for key in ("neighbor_reference", "neighbor_pseudo_weighting"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be boolean")
    if cfg["neighbor_pseudo_weighting"] and not cfg["neighbor_reference"]:
        raise ValueError("Neighbor weighting requires a reference bank")
    if not cfg["neighbor_reference"]:
        return
    if cfg.get("pseudo_weighting") != "epoch_gaussian" or cfg.get("pseudo_target") != "soft" or not cfg.get("pseudo_label_weight", 0):
        raise ValueError("Neighbor pilot requires soft targets and epoch Gaussian weights")
    if cfg.get("reconstruction_pilot"):
        raise ValueError("Neighbor pilot does not support reconstruction")
    for key, minimum in (("neighbor_warmup_epochs", 0), ("neighbor_k", 1)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {key}")
    for key in ("neighbor_temperature", "neighbor_js_scale"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["neighbor_confidence_threshold"] <= 1 or not 0 < cfg["neighbor_weight_floor"] <= 1:
        raise ValueError("Invalid neighbor threshold or floor")


@contextmanager
def isolated_eval(model):
    """Preserve every module's mode and CPU/CUDA RNG, including on exceptions."""
    modes = [(m, m.training) for m in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        for module, mode in modes:
            module.training = mode


@torch.no_grad()
def agreement_scores(probabilities, features, reference, labels, config):
    cfg = {**DEFAULTS, **config}
    q = neighbor_probabilities(features.cpu(), reference.cpu(), labels.cpu(), probabilities.shape[1],
                               cfg["neighbor_k"], cfg["neighbor_temperature"])
    return js_divergence(probabilities.cpu(), q).to(probabilities.device)


@torch.no_grad()
def reweight(base_weights, confidence, disagreement, config, epoch):
    cfg = {**DEFAULTS, **config}
    eligible = confidence >= cfg["neighbor_confidence_threshold"]
    factor = torch.ones_like(base_weights)
    enabled = cfg["neighbor_pseudo_weighting"] and epoch >= cfg["neighbor_warmup_epochs"]
    if enabled:
        floor = cfg["neighbor_weight_floor"]
        factor[eligible] = floor + (1 - floor) * (-disagreement[eligible] / cfg["neighbor_js_scale"]).exp()
    weighted = base_weights * factor
    if enabled and float(weighted.sum()) > 0:
        weighted = weighted * (base_weights.sum() / weighted.sum())
    return weighted, {"neighbor_eligible": int(eligible.sum()),
                      "neighbor_reweighted": int(eligible.sum()) if enabled else 0,
                      "neighbor_js_sum": float(disagreement.sum()),
                      "neighbor_raw_factor_sum": float(factor.sum()),
                      "pseudo_base_weight_sum": float(base_weights.sum())}
