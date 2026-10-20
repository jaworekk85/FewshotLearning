"""Shared batches and additive losses for the small-CNN SSL pilot.

Pseudo-labeling follows the weak-to-strong idea of FixMatch. This is a
controlled pilot with two labeled views, not a reproduction of its recipe.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from .data import seed_worker
from .transforms import MultiViewTransform


METHODS = {
    "ce_2v": (0.0, 0.0, 0.0),
    "ce_2v_triplet_l": (1.0, 0.0, 0.0),
    "ce_2v_triplet_l_semihard": (1.0, 0.0, 0.0),
    "ce_2v_triplet_u": (0.0, 1.0, 0.0),
    "ce_2v_triplet_lu": (1.0, 1.0, 0.0),
    "ce_2v_pseudo": (0.0, 0.0, 1.0),
    "ce_2v_triplet_lu_pseudo": (1.0, 1.0, 1.0),
    "ce_2v_pseudo_random_triplet": (0.0, 1.0, 1.0),
    "ce_2v_pseudo_guided_triplet": (0.0, 0.0, 1.0),
    "ce_2v_pseudo_semihard_triplet": (0.0, 0.0, 1.0),
    "ce_2v_pseudo_soft_weight": (0.0, 0.0, 1.0),
    "ce_2v_pseudo_soft": (0.0, 0.0, 1.0),
    "ce_2v_pseudo_soft_instance_triplet": (0.0, 0.1, 1.0),
    "ce_2v_pseudo_soft_mixed_triplet": (1.0 / 30, 1.0 / 30, 1.0),
    "ce_2v_pseudo_soft_mined_triplet": (1.0 / 30, 1.0 / 30, 1.0),
    "ce_2v_u_control": (0.0, 0.0, 0.0),
}


def select_method(config: dict, method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown pilot method {method!r}.")
    labeled, unlabeled, pseudo = METHODS[method]
    selected = {**config, "pilot_mode": True, "method": method, "labeled_views": 2,
                "triplet_l_weight": labeled, "triplet_u_weight": unlabeled,
                "pseudo_label_weight": pseudo, "use_unlabeled_control": method == "ce_2v_u_control"}
    if method in {"ce_2v_pseudo_guided_triplet", "ce_2v_pseudo_semihard_triplet", "ce_2v_pseudo_soft_mixed_triplet"}:
        selected["pseudo_triplet_weight"] = 1.0 / 30 if method == "ce_2v_pseudo_soft_mixed_triplet" else 1.0
        selected.setdefault("pseudo_triplet_pool", "labeled_and_pseudo")
        if "semihard" in method or method == "ce_2v_pseudo_soft_mixed_triplet":
            selected["pseudo_triplet_mining"] = "semihard"
    elif "pseudo_triplet_weight" in selected:
        selected["pseudo_triplet_weight"] = 0.0
    if method in {"ce_2v_pseudo_soft_weight", "ce_2v_pseudo_soft", "ce_2v_pseudo_soft_instance_triplet", "ce_2v_pseudo_soft_mixed_triplet"}:
        selected.update(pseudo_weighting="epoch_gaussian", pseudo_target="hard" if method == "ce_2v_pseudo_soft_weight" else "soft")
        selected.setdefault("pseudo_temperature", 0.5)
    if method in {"ce_2v_pseudo_soft_instance_triplet", "ce_2v_pseudo_soft_mixed_triplet"}:
        selected["triplet_u_mining"] = "batch_all"
    if method in {"ce_2v_triplet_l_semihard", "ce_2v_pseudo_soft_mixed_triplet"}:
        selected["triplet_l_mining"] = "semihard"
    if method == "ce_2v_pseudo_soft_mined_triplet":
        from .mined_triplets import DEFAULTS
        selected.update(DEFAULTS)
        selected.update(pseudo_weighting="epoch_gaussian", pseudo_target="soft",
                        pseudo_triplet_weight=1.0 / 30, pseudo_triplet_pool="labeled_and_pseudo",
                        pseudo_triplet_mining="random", triplet_l_mining="semihard", triplet_u_mining="batch_all")
        selected.setdefault("pseudo_temperature", .5)
    return selected


def validate_pilot(config: dict):
    from .neighbor_reliability import validate_neighbor
    validate_neighbor(config)
    from .mined_triplets import validate_mined
    validate_mined(config)
    from .ema_teacher import validate_teacher
    from .mixed_contrastive import validate_contrastive
    validate_teacher(config)
    validate_contrastive(config)
    if int(config.get("labeled_views", 2)) != 2:
        raise ValueError("This pilot requires exactly two labeled views.")
    for field in ("triplet_l_weight", "triplet_u_weight", "pseudo_label_weight", "pseudo_triplet_weight"):
        if float(config.get(field, 0.0)) < 0:
            raise ValueError(f"{field} must be non-negative.")
    if not 0 <= float(config.get("pseudo_label_threshold", 0.95)) <= 1:
        raise ValueError("pseudo_label_threshold must be between zero and one.")
    if int(config.get("unlabeled_batch_size", config["batch_size"])) <= 0:
        raise ValueError("unlabeled_batch_size must be positive.")
    if float(config.get("triplet_margin", 0.2)) <= 0:
        raise ValueError("triplet_margin must be positive.")
    if config.get("pseudo_triplet_pool", "labeled_and_pseudo") not in {"labeled_and_pseudo", "pseudo_only"}:
        raise ValueError("Unknown pseudo_triplet_pool.")
    if int(config["num_workers"]) != 0:
        raise ValueError("Paired pilot currently requires num_workers=0 for reproducible augmentation draws.")
    if config.get("pseudo_triplet_mining", "random") not in {"random", "semihard"}:
        raise ValueError("Unknown pseudo_triplet_mining.")
    if config.get("triplet_l_mining", "random") not in {"random", "semihard"}:
        raise ValueError("Unknown triplet_l_mining.")
    if config.get("triplet_u_mining", "random") not in {"random", "batch_all"}:
        raise ValueError("Unknown triplet_u_mining.")
    if config.get("pseudo_weighting", "hard") not in {"hard", "epoch_gaussian"}:
        raise ValueError("Unknown pseudo_weighting.")
    if config.get("pseudo_target", "hard") not in {"hard", "soft"} or float(config.get("pseudo_temperature", 0.5)) <= 0:
        raise ValueError("Invalid pseudo target or temperature.")


def view_transform(config: dict, strong: bool = False):
    dataset_name = config.get("dataset", "cifar10")
    if dataset_name in {"fashion_mnist", "mnist"}:
        # Geometry-only augmentation: no flips of digits, inversion or color policy.
        degrees, translation = (15, .12) if strong else (5, .04)
        steps = [transforms.RandomAffine(degrees, translate=(translation, translation),
                 scale=(.85, 1.1) if strong else (.95, 1.05), fill=0)]
        if dataset_name == "fashion_mnist":
            steps.append(transforms.RandomHorizontalFlip())
        return transforms.Compose([*steps, transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"])])
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if strong:
        steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    return transforms.Compose([*steps, transforms.ToTensor(),
                               transforms.Normalize(config["mean"], config["std"])])


class WeakStrong:
    def __init__(self, config):
        self.weak = view_transform(config)
        self.strong = view_transform(config, strong=True)

    def __call__(self, image):
        return [self.weak(image), self.strong(image)]


class KeyedViews(Dataset):
    """Augmentation draws depend on image, visit and epoch, not other losses.

    Unlabeled instances store no labels and do not call a labeled dataset's
    __getitem__. An independent RNG context keeps U from changing L's views.
    """
    def __init__(self, images, indices, transform, seed, labels=None):
        self.images = images
        self.indices = list(indices)
        self.transform = transform
        self.seed = int(seed)
        self.labels = None if labels is None else [int(labels[i]) for i in self.indices]
        self.set_epoch(0)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.visits = {}

    def __getitem__(self, item):
        index = self.indices[item]
        visit = self.visits.get(index, 0)
        self.visits[index] = visit + 1
        key = (self.seed + 1_000_000_007 * self.epoch + 1_000_003 * index + 104_729 * visit) % (2**63)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(key)
            views = self.transform(Image.fromarray(self.images[index]))
        return views if self.labels is None else (views, self.labels[item])


def labeled_dataset(images, indices, labels, config):
    return KeyedViews(images, indices, MultiViewTransform(view_transform(config), 2),
                      int(config["seed"]) + 11_003, labels=labels)


def synthetic_datasets(config):
    rng = np.random.default_rng(724)
    shape = (120, 28, 28) if config.get("dataset") in {"fashion_mnist", "mnist"} else (120, 32, 32, 3)
    images = rng.integers(0, 256, size=shape, dtype=np.uint8)
    labels = np.arange(120) % 10
    train = labeled_dataset(images[:100], list(range(40)), labels[:100], config)
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    val = TensorDataset(torch.stack([evaluation(Image.fromarray(x)) for x in images[40:60]]), torch.tensor(labels[40:60]))
    test = TensorDataset(torch.stack([evaluation(Image.fromarray(x)) for x in images[100:120]]), torch.tensor(labels[100:120]))
    return train, val, test


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pseudo_label_loss(weak_logits, strong_logits, threshold):
    # Denominator is the entire U batch, including rejected examples.
    with torch.no_grad():
        confidence, targets = weak_logits.detach().softmax(1).max(1)
        accepted = confidence >= threshold
    loss = (F.cross_entropy(strong_logits, targets, reduction="none") * accepted).mean()
    return loss, accepted, targets, confidence


class EpochPseudoState:
    """Epoch-lagged confidence weighting; targets are refreshed every step.

    Inspired by SoftMatch's truncated Gaussian, without its full recipe or
    distribution alignment. The first epoch retains the original hard gate.
    """
    def __init__(self, config):
        self.config = config
        self.mean, self.std, self.ready = 0.95, 0.1, False
        self.reset()

    def reset(self):
        self.count, self.total, self.squares = 0, 0.0, 0.0

    def loss(self, weak, strong, weight_transform=None):
        with torch.no_grad():
            probabilities = weak.detach().softmax(1)
            confidence, targets = probabilities.max(1)
            accepted = confidence >= float(self.config.get("pseudo_label_threshold", 0.95))
            if self.ready:
                offset = (confidence - self.mean).clamp(max=0)
                weights = torch.exp(-offset.square() / (2 * self.std**2))
            else:
                weights = accepted.float()
            self.count += confidence.numel()
            self.total += float(confidence.sum())
            self.squares += float(confidence.square().sum())
            soft = (weak.detach() / float(self.config.get("pseudo_temperature", 0.5))).softmax(1)
            if weight_transform is not None:
                weights = weight_transform(weights, confidence)
        losses = -(soft * strong.log_softmax(1)).sum(1) if self.config.get("pseudo_target", "hard") == "soft" else F.cross_entropy(strong, targets, reduction="none")
        return (weights * losses).mean(), accepted, targets, confidence, weights

    def finish_epoch(self):
        if self.count:
            self.mean = self.total / self.count
            self.std = max((max(self.squares / self.count - self.mean**2, 0))**0.5, 0.05)
            self.ready = True
        return {"pseudo_weight_center_next_epoch": self.mean, "pseudo_weight_std_next_epoch": self.std}

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "ready": self.ready}

    def load_state_dict(self, state):
        self.mean, self.std, self.ready = state["mean"], state["std"], state["ready"]


def labeled_triplet_loss(first, second, labels, generator, margin, mining="random", stats=None):
    count = labels.numel()
    cpu_labels = labels.detach().cpu()
    same = cpu_labels[:, None] == cpu_labels[None, :]
    different = ~same
    other_positive = same & ~torch.eye(count, dtype=torch.bool)
    positive_scores = torch.rand(count, count, generator=generator).masked_fill(~other_positive, -1)
    positive = positive_scores.argmax(1)
    positive = torch.where(other_positive.any(1), positive, torch.arange(count))
    negative_scores = torch.rand(count, count, generator=generator).masked_fill(~different, -1)
    negative = negative_scores.argmax(1)
    valid = different.any(1).to(first.device)
    if stats is not None:
        stats.update(triplet_l_valid=int(valid.sum()), triplet_l_active=0, triplet_l_semihard=0)
    if not bool(valid.any()):
        return first.sum() * 0.0
    if mining == "semihard":
        with torch.no_grad():
            distance = torch.cdist(first.detach(), first.detach())
            positive_distance = (first.detach() - second[positive.to(first.device)].detach()).norm(dim=1)
            candidates = different.to(first.device)
            semihard = candidates & (distance > positive_distance[:, None]) & (distance < positive_distance[:, None] + margin)
            found = semihard.any(1)
            semi_choice = distance.masked_fill(~semihard, float("inf")).argmin(1)
            nearest = distance.masked_fill(~candidates, float("inf")).argmin(1)
            negative = torch.where(found, semi_choice, nearest).cpu()
            if stats is not None:
                stats["triplet_l_semihard"] = int(found.sum())
    if mining == "random" and stats is None:
        return F.triplet_margin_loss(first[valid], second[positive.to(first.device)][valid],
                                     first[negative.to(first.device)][valid], margin=margin)
    losses = F.triplet_margin_loss(first[valid], second[positive.to(first.device)][valid],
                                   first[negative.to(first.device)][valid], margin=margin, reduction="none")
    if stats is not None:
        stats["triplet_l_active"] = int((losses.detach() > 0).sum())
    return losses.mean()


def unlabeled_triplet_loss(first, second, generator, margin, mining="random", stats=None, exclude_negative_mask=None):
    count = first.shape[0]
    if mining not in {"random", "batch_all"}:
        raise ValueError("Unknown unlabeled triplet mining.")
    if stats is not None:
        stats.update(triplet_u_pairs=0, triplet_u_active=0)
    if count < 2:
        return (first.sum() + second.sum()) * 0.0
    if mining == "batch_all":
        # Anchor = weak view i; positive = strong view i. Every strong view
        # j != i is a negative, with no labels or pseudo-labels inspected.
        distances = torch.cdist(first, second, p=2)
        losses = (distances.diagonal()[:, None] - distances + margin).relu()
        mask = ~torch.eye(count, device=first.device, dtype=torch.bool)
        if exclude_negative_mask is not None:
            if exclude_negative_mask.shape != mask.shape or exclude_negative_mask.dtype != torch.bool:
                raise ValueError("Invalid instance negative exclusion mask")
            excluded = mask & exclude_negative_mask.to(mask.device)
            mask &= ~excluded
            if stats is not None:
                stats["metric_instance_excluded_pairs"] = int(excluded.sum())
        losses = losses[mask]
        if stats is not None:
            stats.update(triplet_u_pairs=losses.numel(), triplet_u_active=int((losses.detach() > 0).sum()))
        return losses.mean() if losses.numel() else (first.sum() + second.sum()) * 0.
    negative = torch.randint(count - 1, (count,), generator=generator)
    negative += (negative >= torch.arange(count)).long()
    return F.triplet_margin_loss(first, second, first[negative.to(first.device)], margin=margin)


def pseudo_triplet_indices(labeled_labels, predictions, accepted, generator, pool="labeled_and_pseudo"):
    """Select distinct images using known L labels and confident U predictions.

    Returned indices refer to the concatenation [L, U]. Only accepted U images
    are anchors. Positives and negatives may come from L or accepted U.
    """
    known = labeled_labels.detach().cpu()
    predicted = predictions.detach().cpu()
    accepted = accepted.detach().cpu().bool()
    l_size, u_size = known.numel(), predicted.numel()
    pool_labels = torch.cat([known, predicted])
    trusted = torch.cat([torch.full((l_size,), pool == "labeled_and_pseudo", dtype=torch.bool), accepted])
    positive_mask = (predicted[:, None] == pool_labels[None, :]) & trusted[None, :]
    negative_mask = (predicted[:, None] != pool_labels[None, :]) & trusted[None, :]
    # Another augmentation of the anchor is not a class-positive in this variant.
    positive_mask[torch.arange(u_size), l_size + torch.arange(u_size)] = False
    valid = accepted & positive_mask.any(1) & negative_mask.any(1)
    positive = torch.rand(u_size, l_size + u_size, generator=generator).masked_fill(~positive_mask, -1).argmax(1)
    negative = torch.rand(u_size, l_size + u_size, generator=generator).masked_fill(~negative_mask, -1).argmax(1)
    anchors = l_size + torch.arange(u_size)
    return anchors[valid], positive[valid], negative[valid]


def pseudo_triplet_loss(l_first, l_second, u_first, u_second, labels, predictions,
                        accepted, generator, margin, pool="labeled_and_pseudo", mining="random"):
    anchors, positive, negative = pseudo_triplet_indices(labels, predictions, accepted, generator, pool)
    l_size, u_size = labels.numel(), u_first.shape[0]
    counts = {"pseudo_triplet_valid": anchors.numel(), "pseudo_triplet_attempted": u_size,
              "pseudo_triplet_labeled_positives": int((positive < l_size).sum()),
              "pseudo_triplet_labeled_negatives": int((negative < l_size).sum()),
              "pseudo_triplet_pseudo_positives": int((positive >= l_size).sum()),
              "pseudo_triplet_pseudo_negatives": int((negative >= l_size).sum())}
    if mining != "random":
        counts.update(pseudo_triplet_active=0, pseudo_triplet_semihard=0)
    if not anchors.numel():
        return (l_first.sum() + u_first.sum()) * 0.0, counts
    first = F.normalize(torch.cat([l_first, u_first]), dim=1)
    second = F.normalize(torch.cat([l_second, u_second]), dim=1)
    device = first.device
    semihard_count = 0
    if mining == "semihard":
        with torch.no_grad():
            all_labels = torch.cat([labels, predictions]).to(device)
            trusted = torch.cat([torch.full_like(labels, pool == "labeled_and_pseudo", dtype=torch.bool), accepted]).to(device)
            candidates = trusted[None, :] & (all_labels[anchors.to(device), None] != all_labels[None, :])
            distance = torch.cdist(first[anchors.to(device)].detach(), first.detach())
            positive_distance = (first[anchors.to(device)].detach() - second[positive.to(device)].detach()).norm(dim=1)
            semihard = candidates & (distance > positive_distance[:, None]) & (distance < positive_distance[:, None] + margin)
            nearest = distance.masked_fill(~candidates, float("inf")).argmin(1)
            semi_choice = distance.masked_fill(~semihard, float("inf")).argmin(1)
            found = semihard.any(1)
            negative = torch.where(found, semi_choice, nearest).cpu()
            semihard_count = int(found.sum())
    losses = F.triplet_margin_loss(first[anchors.to(device)], second[positive.to(device)],
                                   first[negative.to(device)], margin=margin, reduction="none")
    # Recompute source counters after mining changes negative indices.
    counts.update(pseudo_triplet_labeled_negatives=int((negative < l_size).sum()),
                  pseudo_triplet_pseudo_negatives=int((negative >= l_size).sum()))
    if mining != "random":
        counts.update(pseudo_triplet_semihard=semihard_count, pseudo_triplet_active=int((losses.detach() > 0).sum()))
    # Rejected/missing triplets contribute zero; denominator is the full U batch.
    return losses.sum() / u_size, counts


class PilotTraining:
    def __init__(self, config, train, splits):
        self.config, self.train = config, train
        seed = int(config["seed"])
        self.l_generator = torch.Generator().manual_seed(seed + 42_101)
        self.u_generator = torch.Generator().manual_seed(seed + 42_103)
        self.loader_generator = torch.Generator().manual_seed(seed + 42_107)
        self.guided_generator = torch.Generator().manual_seed(seed + 42_109)
        self.soft_state = EpochPseudoState(config) if config.get("pseudo_weighting") == "epoch_gaussian" else None
        self.ema = None
        if config.get("ema_teacher"):
            from .ema_teacher import EMATeacher
            self.ema = EMATeacher(config)
        use_u = any(float(config.get(key, 0.0)) > 0 for key in ("triplet_u_weight", "pseudo_label_weight", "pseudo_triplet_weight")) or config.get("use_unlabeled_control", False)
        self.unlabeled = self.loader = None
        if use_u:
            used = set(splits["train"]) | set(splits["validation"])
            indices = [i for i in range(len(train.images)) if i not in used]
            if not indices:
                raise ValueError("This variant needs a nonempty unlabeled training pool.")
            self.unlabeled = KeyedViews(train.images, indices, WeakStrong(config), seed + 21_011)
            batch = int(config.get("unlabeled_batch_size", config["batch_size"]))
            self.loader = DataLoader(self.unlabeled, batch_size=batch, shuffle=True,
                                     generator=self.loader_generator, num_workers=0,
                                     worker_init_fn=seed_worker, drop_last=len(indices) >= batch)
        self.iterator = None

    def start_epoch(self, epoch):
        self.epoch = epoch
        if self.soft_state is not None:
            self.soft_state.reset()
        self.train.set_epoch(epoch)
        if self.unlabeled is not None:
            self.unlabeled.set_epoch(epoch)
            self.iterator = iter(self.loader)

    def state_dict(self):
        state = {"l": self.l_generator.get_state(), "u": self.u_generator.get_state(),
                "loader": self.loader_generator.get_state(), "guided": self.guided_generator.get_state()}
        if self.soft_state is not None:
            state["soft"] = self.soft_state.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        return state

    def after_optimizer_step(self, model):
        if self.ema is None:
            return {}
        tick = time.perf_counter()
        self.ema.update(model)
        return {"ema_updates": 1, "ema_update_seconds": time.perf_counter()-tick}

    def evaluate_teacher(self, loader, device):
        from .neighbor_reliability import isolated_eval
        from .train import evaluate
        with isolated_eval(self.ema.model):
            return evaluate(self.ema.model, loader, device, normalize=False)

    def finish_epoch(self):
        result = self.soft_state.finish_epoch() if self.soft_state is not None else {}
        if self.config.get("neighbor_reference"):
            result.update(neighbor_reference_views=len(self.reference_labels),
                          neighbor_reference_seconds=self.reference_seconds)
        return result

    def prepare_neighbor_epoch(self, model, device):
        import time
        from .neighbor_reliability import isolated_eval
        from .supervised import cifar_transform
        tick = time.perf_counter()
        # KeyedViews stores only the already revealed labels for these indices.
        reference = KeyedViews(self.train.images, self.train.indices,
                               cifar_transform(self.config, False), int(self.config["seed"]) + 61001)
        loader = DataLoader(reference, batch_size=256, shuffle=False,
                            generator=torch.Generator().manual_seed(61001))
        features = []
        with isolated_eval(model):
            for images in loader:
                z, _ = model(images.to(device), normalize=False)
                features.append(F.normalize(z, dim=1).cpu())
        self.reference_features = torch.cat(features)
        self.reference_labels = torch.tensor(self.train.labels, dtype=torch.long)
        self.reference_seconds = time.perf_counter() - tick

    def load_state_dict(self, state):
        self.l_generator.set_state(state["l"])
        self.u_generator.set_state(state["u"])
        self.loader_generator.set_state(state["loader"])
        if "guided" in state:
            self.guided_generator.set_state(state["guided"])
        if self.soft_state is not None:
            self.soft_state.load_state_dict(state["soft"])
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])

    def step(self, model, images, labels, device):
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError("Pilot training needs two labeled views.")
        labeled = [view.to(device) for view in images]
        labels = labels.to(device)
        inputs = list(labeled)
        u_size = 0
        if self.loader is not None:
            try:
                unlabeled = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                unlabeled = next(self.iterator)
            inputs.extend(view.to(device) for view in unlabeled)
            u_size = unlabeled[0].shape[0]
        teacher_logits = None
        teacher_metrics = {}
        if self.ema is not None:
            teacher = self.ema.initialize(model)
            teacher_metrics = {"ema_training_views": 0, "ema_forward_seconds": 0., "ema_target_steps": 0}
            if self.epoch >= self.config.get("ema_teacher_warmup_epochs", 3):
                tick = time.perf_counter()
                with torch.no_grad():
                    _, teacher_logits = teacher(inputs[2])
                teacher_metrics.update(ema_training_views=u_size, ema_forward_seconds=time.perf_counter()-tick,
                                       ema_target_steps=1)
        features, logits = model(torch.cat(inputs), normalize=False)
        size = labels.numel()
        labeled_logits = logits[:2 * size]
        targets = labels.repeat(2)
        ce = F.cross_entropy(labeled_logits, targets)
        zero = ce * 0.0
        triplet_l = triplet_u = pseudo = guided = zero
        guided_counts = {}
        unlabeled_counts = {} if self.config.get("triplet_u_mining", "random") != "random" else None
        labeled_counts = {} if self.config.get("triplet_l_mining", "random") != "random" else None
        guided_weight = float(self.config.get("pseudo_triplet_weight", 0.0))
        margin = float(self.config.get("triplet_margin", 0.2))
        # Normalization affects only the metric objective, never the CE head.
        if float(self.config.get("triplet_l_weight", 0.0)) > 0:
            triplet_l = labeled_triplet_loss(F.normalize(features[:size], dim=1),
                                             F.normalize(features[size:2 * size], dim=1), labels,
                                             self.l_generator, margin, self.config.get("triplet_l_mining", "random"), labeled_counts)
        accepted_count = attempted = 0
        confidence_sum = 0.0
        soft_metrics = {}
        histogram = [0] * 10
        if u_size:
            start = 2 * size
            if float(self.config.get("triplet_u_weight", 0.0)) > 0:
                exclusion = None
                if self.config.get("metric_triplet_mask_same_pseudo"):
                    from .mined_triplets import conflicting_instance_pairs
                    with torch.no_grad():
                        u_confidence, u_predictions = logits[start:start + u_size].detach().softmax(1).max(1)
                        exclusion = conflicting_instance_pairs(u_predictions, u_confidence,
                            float(self.config["metric_triplet_confidence_threshold"]))
                triplet_u = unlabeled_triplet_loss(F.normalize(features[start:start + u_size], dim=1),
                                                   F.normalize(features[start + u_size:], dim=1),
                                                   self.u_generator, margin,
                                                   self.config.get("triplet_u_mining", "random"), unlabeled_counts, exclusion)
            if float(self.config.get("pseudo_label_weight", 0.0)) > 0 or guided_weight > 0:
                if self.soft_state is None:
                    pseudo, accepted, predictions, confidence = pseudo_label_loss(
                        logits[start:start + u_size], logits[start + u_size:],
                        float(self.config.get("pseudo_label_threshold", 0.95)))
                else:
                    weight_transform = None
                    if self.config.get("neighbor_reference"):
                        from .neighbor_reliability import isolated_eval, agreement_scores, reweight
                        with isolated_eval(model):
                            z_eval, p_eval = model(inputs[2], normalize=False)
                            disagreement = agreement_scores(p_eval.softmax(1), z_eval,
                                self.reference_features, self.reference_labels, self.config)
                        soft_metrics["neighbor_query_views"] = u_size
                        def weight_transform(base_weights, confidence):
                            updated, counts = reweight(base_weights, confidence, disagreement,
                                                       self.config, self.epoch)
                            soft_metrics.update(counts)
                            return updated
                    pseudo, accepted, predictions, confidence, weights = self.soft_state.loss(
                        logits[start:start + u_size] if teacher_logits is None else teacher_logits,
                        logits[start + u_size:], weight_transform)
                    soft_metrics["pseudo_weight_sum"] = float(weights.sum())
                accepted_count, attempted = int(accepted.sum()), u_size
                confidence_sum = float(confidence.sum())
                histogram = torch.bincount(predictions[accepted], minlength=10).cpu().tolist()
                if guided_weight > 0:
                    if "metric_triplet_design" in self.config:
                        from .mined_triplets import mined_triplet_loss
                        pixels = torch.cat([inputs[0], inputs[2]]) if self.config["metric_triplet_distance"] == "pixels" else None
                        guided, guided_counts = mined_triplet_loss(
                            features[:size], features[size:2 * size], features[start:start + u_size], features[start + u_size:],
                            labels, predictions, confidence, self.guided_generator, self.config, self.epoch, pixels)
                    else:
                        guided, guided_counts = pseudo_triplet_loss(
                            features[:size], features[size:2 * size],
                            features[start:start + u_size], features[start + u_size:],
                            labels, predictions, accepted, self.guided_generator, margin,
                            self.config.get("pseudo_triplet_pool", "labeled_and_pseudo"),
                            self.config.get("pseudo_triplet_mining", "random"))
        loss = (ce + float(self.config.get("triplet_l_weight", 0.0)) * triplet_l
                + float(self.config.get("triplet_u_weight", 0.0)) * triplet_u
                + float(self.config.get("pseudo_label_weight", 0.0)) * pseudo
                + guided_weight * guided)
        contrastive_metrics = {}
        if self.config.get("contrastive_weight", 0):
            from .mixed_contrastive import mixed_contrastive_loss
            contrastive, contrastive_metrics = mixed_contrastive_loss(features, labels, predictions,
                confidence, self.config, self.epoch)
            loss = loss + float(self.config["contrastive_weight"]) * contrastive
        metrics = {"ce_loss": float(ce.detach()), "triplet_l_loss": float(triplet_l.detach()),
                   "triplet_u_loss": float(triplet_u.detach()), "pseudo_label_loss": float(pseudo.detach()),
                   "processed_labeled_views": 2 * size, "processed_unlabeled_views": 2 * u_size,
                   "pseudo_accepted": accepted_count, "pseudo_attempted": attempted,
                   "pseudo_confidence_sum": confidence_sum,
                   **{f"pseudo_class_{i}": count for i, count in enumerate(histogram)}}
        if guided_weight > 0:
            metrics.update(pseudo_triplet_loss=float(guided.detach()), **guided_counts)
            if self.config.get("pseudo_triplet_mining", "random") != "random":
                metrics.setdefault("pseudo_triplet_active", 0)
                metrics.setdefault("pseudo_triplet_semihard", 0)
        metrics.update(soft_metrics)
        metrics.update(teacher_metrics)
        metrics.update(contrastive_metrics)
        if labeled_counts is not None:
            metrics.update(labeled_counts)
        if unlabeled_counts is not None:
            metrics.update(unlabeled_counts)
        return loss, labeled_logits, targets, metrics
