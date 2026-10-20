"""A single weighted multi-positive graph on existing L1/L2/Uw/Us embeddings."""
import math

import torch
from torch.nn import functional as F


def validate_contrastive(config):
    weight = config.get("contrastive_weight", 0.)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Invalid contrastive weight")
    if not weight:
        return
    if config.get("method") != "ce_2v_pseudo_soft" or config.get("neighbor_reference") or config.get("ema_teacher"):
        raise ValueError("Mixed contrastive pilot requires plain soft SSL")
    temperature = config.get("contrastive_temperature", .2)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid contrastive temperature")
    if not isinstance(config.get("contrastive_pseudo_pairs", False), bool):
        raise ValueError("Pseudo-pair flag must be boolean")
    if not 0 <= config.get("contrastive_pseudo_threshold", .95) <= 1:
        raise ValueError("Invalid contrastive confidence threshold")
    warmup = config.get("contrastive_warmup_epochs", 3)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("Invalid contrastive warmup")


@torch.no_grad()
def pair_graph(labels, predictions, confidence, config):
    """No hidden U labels. Same-image positives override class decisions."""
    n_l, n_u = len(labels), len(predictions)
    device = labels.device
    image_ids = torch.cat([torch.arange(n_l,device=device).repeat(2),
                           (n_l+torch.arange(n_u,device=device)).repeat(2)])
    known = torch.arange(2*n_l+2*n_u,device=device) < 2*n_l
    classes = torch.cat([labels.repeat(2),predictions.repeat(2)])
    certainty = torch.cat([torch.ones(2*n_l,device=device),confidence.detach().repeat(2)])
    trusted = known.clone()
    if config.get("contrastive_pseudo_pairs",False):
        trusted |= certainty >= config.get("contrastive_pseudo_threshold",.95)
    same_image = image_ids[:,None] == image_ids[None,:]
    self_mask = torch.eye(len(image_ids),device=device,dtype=torch.bool)
    same_image &= ~self_mask
    class_positive = trusted[:,None] & trusted[None,:] & (classes[:,None] == classes[None,:]) & ~same_image & ~self_mask
    weights = class_positive.to(certainty) * certainty[:,None] * certainty[None,:]
    weights[same_image] = 1.
    known_pair = known[:,None] & known[None,:]
    cross_pair = known[:,None] != known[None,:]
    stats = dict(contrastive_instance_pairs=int(same_image.sum()),
                 contrastive_ll_pairs=int((class_positive & known_pair).sum()),
                 contrastive_lu_pairs=int((class_positive & cross_pair).sum()),
                 contrastive_uu_pairs=int((class_positive & ~known[:,None] & ~known[None,:]).sum()),
                 contrastive_negative_pairs=int((~self_mask & (weights == 0)).sum()),
                 contrastive_positive_weight_sum=float(weights.sum()))
    return weights, stats


def mixed_contrastive_loss(features, labels, predictions, confidence, config, epoch):
    weights, stats = pair_graph(labels, predictions, confidence, config)
    stats.update(contrastive_active_anchors=0,contrastive_loss=0.)
    if epoch < config.get("contrastive_warmup_epochs",3):
        return features.sum()*0., stats
    z = F.normalize(features,dim=1)
    similarity = z @ z.T / config.get("contrastive_temperature",.2)
    self_mask = torch.eye(len(z),device=z.device,dtype=torch.bool)
    log_denominator = torch.logsumexp(similarity.masked_fill(self_mask,-torch.inf),dim=1)
    # The positive graph excludes self and every node has its paired view.
    log_probability = similarity - log_denominator[:,None]
    positive_mass = weights.sum(1)
    valid = positive_mass > 0
    if not bool(valid.any()):
        return features.sum()*0., stats
    loss = -((weights * log_probability).sum(1)[valid] / positive_mass[valid]).mean()
    stats.update(contrastive_active_anchors=int(valid.sum()),contrastive_loss=float(loss.detach()))
    return loss, stats
