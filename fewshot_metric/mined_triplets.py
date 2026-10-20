"""New triplet mining: labeled anchors, distant positives and confidence weights.

Selectors see revealed L labels and model predictions only, never hidden U labels.
Mining is detached; the hinge loss is always computed in trainable embeddings.
"""
import math
import time

import torch
from torch.nn import functional as F


DEFAULTS = dict(metric_triplet_design="mixed", metric_triplet_distance="embedding",
                metric_triplet_warmup_epochs=3, metric_triplet_positive_fraction=.25,
                metric_triplet_confidence_threshold=.95, metric_triplet_confidence_power=1.,
                metric_triplet_anchor_limit=128, metric_triplet_mask_same_pseudo=False)


def validate_mined(config):
    if "metric_triplet_design" not in config:
        return
    cfg = {**DEFAULTS, **config}
    if cfg.get("method") != "ce_2v_pseudo_soft_mined_triplet":
        raise ValueError("Mined triplets require their explicit method preset")
    if cfg["metric_triplet_design"] not in {"lul", "mixed"} or cfg["metric_triplet_distance"] not in {"embedding", "pixels"}:
        raise ValueError("Unknown triplet design or distance")
    for key, minimum in (("metric_triplet_warmup_epochs", 0), ("metric_triplet_anchor_limit", 1)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if not 0 < cfg["metric_triplet_positive_fraction"] <= 1 or not 0 <= cfg["metric_triplet_confidence_threshold"] <= 1:
        raise ValueError("Invalid mining fraction or confidence threshold")
    if not math.isfinite(cfg["metric_triplet_confidence_power"]) or cfg["metric_triplet_confidence_power"] < 0:
        raise ValueError("Invalid triplet confidence exponent")
    if not isinstance(cfg["metric_triplet_mask_same_pseudo"], bool):
        raise ValueError("Instance mask flag must be boolean")
    if cfg.get("neighbor_reference") or cfg.get("reconstruction_pilot"):
        raise ValueError("Mined triplet pilot keeps other extensions disabled")


def conflicting_instance_pairs(predictions, confidence, threshold):
    accepted = confidence.detach() >= threshold
    return accepted[:, None] & accepted[None, :] & (predictions[:, None] == predictions[None, :])


@torch.no_grad()
def mine_indices(first, labeled_labels, predictions, confidence, generator, config, weak_pixels=None):
    cfg = {**DEFAULTS, **config}
    labeled_labels, predictions, confidence = (v.detach().cpu() for v in (labeled_labels, predictions, confidence))
    n_l, n_u = len(labeled_labels), len(predictions)
    known = torch.arange(n_l + n_u) < n_l
    labels = torch.cat([labeled_labels, predictions])
    trusted = torch.cat([torch.ones(n_l, dtype=torch.bool), confidence >= cfg["metric_triplet_confidence_threshold"]])
    reliability = torch.cat([torch.ones(n_l), confidence]).pow(cfg["metric_triplet_confidence_power"])
    if len(first) != len(labels) or (labeled_labels < 0).any():
        raise ValueError("Misaligned features or unrevealed L labels")
    if cfg["metric_triplet_distance"] == "embedding":
        geometry = F.normalize(first.detach().cpu(), dim=1)
    else:
        if weak_pixels is None or len(weak_pixels) != len(first):
            raise ValueError("Pixel mining needs the existing weak views")
        # Undo input normalization. Geometry is RGB [0,1] of the weak crop/flip.
        mean = torch.tensor(config["mean"])[None, :, None, None]
        std = torch.tensor(config["std"])[None, :, None, None]
        geometry = (weak_pixels.detach().cpu() * std + mean).flatten(1)
    distances = torch.cdist(geometry, geometry)
    anchors = (known if cfg["metric_triplet_design"] == "lul" else trusted).nonzero().flatten()
    limit = cfg["metric_triplet_anchor_limit"]
    anchors = anchors[torch.randperm(len(anchors), generator=generator)[:limit]]
    selected = []
    for a in anchors.tolist():
        positive = trusted & (labels == labels[a])
        positive[a] = False
        # A labeled anchor must connect to U, rather than duplicate L-only loss.
        if a < n_l:
            positive &= ~known
        negative = trusted & (labels != labels[a])
        if cfg["metric_triplet_design"] == "lul":
            negative &= known
        p = positive.nonzero().flatten()
        n = negative.nonzero().flatten()
        if not len(p) or not len(n):
            continue
        ranked = p[torch.argsort(distances[a, p], descending=True, stable=True)]
        tail = max(1, math.ceil(len(ranked) * cfg["metric_triplet_positive_fraction"]))
        p = int(ranked[torch.randint(tail, (1,), generator=generator)])
        n = int(n[torch.argmin(distances[a, n])])
        selected.append((a, p, n))
    triples = torch.tensor(selected, dtype=torch.long).reshape(-1, 3)
    weights = reliability[triples].prod(1) if len(triples) else torch.empty(0)
    return triples, weights


def mined_triplet_loss(l_first, l_second, u_first, u_second, labels, predictions,
                       confidence, generator, config, epoch, weak_pixels=None):
    tick = time.perf_counter()
    cfg = {**DEFAULTS, **config}
    first = F.normalize(torch.cat([l_first, u_first]), dim=1)
    second = F.normalize(torch.cat([l_second, u_second]), dim=1)
    limit = cfg["metric_triplet_anchor_limit"]
    counts = dict(pseudo_triplet_valid=0, pseudo_triplet_attempted=limit,
        pseudo_triplet_labeled_positives=0, pseudo_triplet_labeled_negatives=0,
        pseudo_triplet_pseudo_positives=0, pseudo_triplet_pseudo_negatives=0,
        pseudo_triplet_active=0, metric_anchor_l=0, metric_anchor_u=0,
        metric_triplet_weight_sum=0., metric_positive_distance_sum=0., metric_negative_distance_sum=0.)
    def finish(loss):
        counts["metric_class_seconds"] = time.perf_counter() - tick
        return loss, counts
    zero = (first.sum() + second.sum()) * 0.
    if epoch < cfg["metric_triplet_warmup_epochs"]:
        return finish(zero)
    triples, weights = mine_indices(first, labels, predictions, confidence, generator, config, weak_pixels)
    if not len(triples):
        return finish(zero)
    a, p, n = triples.to(first.device).unbind(1)
    positive_distance = (first[a] - second[p]).norm(dim=1)
    negative_distance = (first[a] - first[n]).norm(dim=1)
    losses = (positive_distance - negative_distance + float(config.get("triplet_margin", .2))).relu()
    n_l = len(labels)
    counts.update(pseudo_triplet_valid=len(a), pseudo_triplet_active=int((losses.detach() > 0).sum()),
        pseudo_triplet_labeled_positives=int((p < n_l).sum()), pseudo_triplet_pseudo_positives=int((p >= n_l).sum()),
        pseudo_triplet_labeled_negatives=int((n < n_l).sum()), pseudo_triplet_pseudo_negatives=int((n >= n_l).sum()),
        metric_anchor_l=int((a < n_l).sum()), metric_anchor_u=int((a >= n_l).sum()),
        metric_triplet_weight_sum=float(weights.sum()),
        metric_positive_distance_sum=float(positive_distance.detach().sum()),
        metric_negative_distance_sum=float(negative_distance.detach().sum()))
    # Fixed attempted-anchor denominator; low coverage does not amplify its loss.
    return finish((weights.to(losses) * losses).sum() / limit)
