from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """Supervised contrastive loss for features shaped [batch, views, dim]."""
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, views, dim].")

    device = features.device
    batch_size, views, dim = features.shape
    labels = labels.reshape(-1).to(device)
    contrast_labels = labels.repeat_interleave(views).reshape(-1, 1)
    mask = torch.eq(contrast_labels, contrast_labels.T).float()

    contrast = F.normalize(features.reshape(batch_size * views, dim), dim=1)
    logits = torch.div(torch.matmul(contrast, contrast.T), temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size * views, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives = mask.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives
    return -mean_log_prob_pos.mean()
