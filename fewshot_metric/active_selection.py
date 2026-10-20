"""Prepared acquisition rules. Inputs contain no hidden candidate labels.

BADGE uses raw penultimate features and factorized last-layer CE gradients.
Other rules are explicit pilot heuristics, not claims of new algorithms.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


NEW_STRATEGIES = {
    "badge", "entropy_density", "neighbor_disagreement_density",
    "view_disagreement", "neighbor_disagreement_density_mix",
    "confident_neighbor",
}
DEFAULTS = {
    "density_neighbors": 10,
    "density_temperature": 0.1,
    "label_neighbors": 15,
    "label_temperature": 0.2,
    "random_fraction": 0.5,
    "confident_threshold": 0.95,
    "confident_query_fraction": 0.5,
}


def validate_settings(settings):
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown acquisition settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **settings}
    for key in ("density_neighbors", "label_neighbors"):
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("density_temperature", "label_temperature"):
        if not 0 < float(cfg[key]) < float("inf"):
            raise ValueError(f"{key} must be finite and positive.")
    if not 0 <= float(cfg["random_fraction"]) <= 1:
        raise ValueError("random_fraction must be in [0, 1].")
    for key in ("confident_threshold", "confident_query_fraction"):
        if not 0 <= float(cfg[key]) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    return cfg


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)


def js_divergence(first, second):
    return (entropy((first + second) / 2) - (entropy(first) + entropy(second)) / 2).clamp_min(0)


def local_density(features, neighbors=10, temperature=0.1, chunk_size=256):
    """Mean exp(-cosine_distance / temperature) over k other candidates."""
    z = F.normalize(features, dim=1)
    if len(z) == 1:
        return torch.ones(1, dtype=z.dtype)
    k = min(int(neighbors), len(z) - 1)
    values = []
    for start in range(0, len(z), chunk_size):
        similarities = z[start:start + chunk_size] @ z.T
        row = torch.arange(len(similarities))
        similarities[row, start + row] = -torch.inf
        nearest = similarities.topk(k, dim=1).values.clamp(max=1)
        values.append(((nearest - 1) / temperature).exp().mean(1))
    return torch.cat(values)


def neighbor_probabilities(features, known_features, known_labels, classes, neighbors=15, temperature=0.2):
    if known_labels is None or len(known_features) == 0 or len(known_features) != len(known_labels):
        raise ValueError("Neighbor acquisition needs only the currently revealed labels, aligned with features.")
    labels = torch.as_tensor(known_labels, dtype=torch.long)
    if (labels < 0).any() or (labels >= classes).any():
        raise ValueError("Known labels are outside the classifier's classes.")
    similarities = F.normalize(features, dim=1) @ F.normalize(known_features, dim=1).T
    values, indices = similarities.topk(min(int(neighbors), len(known_features)), dim=1)
    result = torch.zeros(len(features), classes, dtype=features.dtype)
    result.scatter_add_(1, labels[indices], (values / temperature).softmax(1))
    return result


def badge_squared_distances(probabilities, features, center):
    """Squared distances between (p-onehot(argmax(p))) tensor-product z."""
    residual = probabilities - F.one_hot(probabilities.argmax(1), probabilities.shape[1]).to(probabilities)
    norms = residual.square().sum(1) * features.square().sum(1)
    cross = (residual @ residual[center]) * (features @ features[center])
    return (norms + norms[center] - 2 * cross).clamp_min(0), norms


def badge_queries(probabilities, features, count, generator):
    # Largest gradient norm first, then D^2-sampling, as in BADGE's k-means++.
    _, norms = badge_squared_distances(probabilities, features, 0)
    center = int(norms.argmax())
    selected = []
    nearest = torch.full_like(norms, torch.inf)
    for step in range(count):
        selected.append(center)
        distances, _ = badge_squared_distances(probabilities, features, center)
        nearest = torch.minimum(nearest, distances)
        nearest[selected] = 0
        if step + 1 == count:
            break
        if float(nearest.sum()) > 0:
            center = int(torch.multinomial(nearest, 1, generator=generator))
        else:
            available = torch.ones(len(features), dtype=torch.bool)
            available[selected] = False
            remaining = available.nonzero().flatten()
            center = int(remaining[torch.randint(len(remaining), (1,), generator=generator)])
    return torch.tensor(selected), norms


def select_prepared_queries(probabilities, features, known_features, known_labels,
                            count, strategy, generator, *, view_probabilities=None, settings=None):
    """CPU acquisition, returning positions and label-free diagnostics.

    Candidate labels are deliberately absent from the interface. Known labels
    are supplied separately and are never inferred from the hidden dataset.
    """
    cfg = validate_settings(settings or {})
    if strategy not in NEW_STRATEGIES or not 0 < count <= len(probabilities):
        raise ValueError("Unknown prepared strategy or invalid acquisition size.")
    if probabilities.ndim != 2 or features.ndim != 2 or len(features) != len(probabilities):
        raise ValueError("Candidate probabilities and features must be aligned matrices.")
    if any(t.device.type != "cpu" or not torch.isfinite(t).all() for t in (probabilities, features, known_features)):
        raise ValueError("Acquisition expects finite CPU tensors.")
    if (probabilities < 0).any() or not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities.sum(1)), atol=1e-5):
        raise ValueError("Invalid class probabilities.")
    h = entropy(probabilities)
    details = {"entropy": h}
    if strategy == "badge":
        chosen, norms = badge_queries(probabilities, features, count, generator)
        details["gradient_norm_squared"] = norms
        score = norms
    elif strategy == "view_disagreement":
        if (view_probabilities is None or view_probabilities.ndim != 3
                or view_probabilities.shape[0] < 2
                or tuple(view_probabilities.shape[1:]) != tuple(probabilities.shape)
                or not torch.isfinite(view_probabilities).all()):
            raise ValueError("View disagreement needs aligned predictions from at least two views.")
        score = (entropy(view_probabilities.mean(0)) - entropy(view_probabilities).mean(0)).clamp_min(0)
        details["view_js"] = score
        chosen = torch.argsort(score, descending=True, stable=True)[:count]
    elif strategy == "confident_neighbor":
        q = neighbor_probabilities(features, known_features, known_labels, probabilities.shape[1],
                                   cfg["label_neighbors"], cfg["label_temperature"])
        score = js_divergence(probabilities, q)
        eligible = probabilities.max(1).values >= cfg["confident_threshold"]
        ordered = torch.argsort(score, descending=True, stable=True)
        confident_count = min(int(count * cfg["confident_query_fraction"]), int(eligible.sum()))
        confident = ordered[eligible[ordered]][:confident_count]
        remaining = torch.ones(len(features), dtype=torch.bool)
        remaining[confident] = False
        uncertainty_order = torch.argsort(h, descending=True, stable=True)
        chosen = torch.cat([confident, uncertainty_order[remaining[uncertainty_order]][:count-confident_count]])
        details.update(neighbor_js=score, neighbor_class=q.argmax(1).float(), confident_eligible=eligible.float())
    else:
        density = local_density(features, cfg["density_neighbors"], cfg["density_temperature"])
        details["density"] = density
        if strategy == "entropy_density":
            score = h * density
        else:
            q = neighbor_probabilities(features, known_features, known_labels, probabilities.shape[1],
                                       cfg["label_neighbors"], cfg["label_temperature"])
            details["neighbor_js"] = js_divergence(probabilities, q)
            details["neighbor_class"] = q.argmax(1).float()
            score = details["neighbor_js"] * density
        ordered = torch.argsort(score, descending=True, stable=True)
        random_count = int(count * cfg["random_fraction"]) if strategy.endswith("_mix") else 0
        ranked_count = count - random_count
        chosen = ordered[:ranked_count]
        if random_count:
            remaining = torch.ones(len(probabilities), dtype=torch.bool)
            remaining[chosen] = False
            remaining = remaining.nonzero().flatten()
            chosen = torch.cat([chosen, remaining[torch.randperm(len(remaining), generator=generator)[:random_count]]])
    details["score"] = score
    details["predicted_class"] = probabilities.argmax(1).float()
    details["confidence"] = probabilities.max(1).values
    summary = {key: {"mean": float(value.mean()), "min": float(value.min()), "max": float(value.max())}
               for key, value in details.items() if not key.endswith("class")}
    selected = [{key: int(value[i]) if key.endswith("class") else float(value[i])
                 for key, value in details.items()} for i in chosen.tolist()]
    for position, row in enumerate(selected):
        row["selection_component"] = (("confident_neighbor" if position < confident_count else "entropy") if strategy == "confident_neighbor"
                                      else "random" if strategy.endswith("_mix") and position >= ranked_count
                                      else "badge" if strategy == "badge" else "ranked")
    return chosen, {"strategy": strategy, "settings": cfg, "pool_summary": summary, "selected": selected}
