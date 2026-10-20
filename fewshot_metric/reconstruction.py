"""Matched denoising AE/VAE objectives on the existing classification encoder."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .pilot import PilotTraining, WeakStrong, model_fingerprint, pseudo_label_loss


def reconstruction_config(base, variant):
    from .pilot import select_method
    variants = {
        "ce_u_control": ("none", False),
        "dae": ("ae", False), "dae_pseudo": ("ae", True),
        "dvae": ("vae", False), "dvae_pseudo": ("vae", True),
    }
    if variant not in variants:
        raise ValueError("Unknown reconstruction variant.")
    mode, pseudo = variants[variant]
    config = select_method(base, "ce_2v_pseudo" if pseudo else "ce_2v_u_control")
    config.update(reconstruction_pilot=True, reconstruction_variant=variant,
                  reconstruction_mode=mode, reconstruction_weight=1.0,
                  reconstruction_kl_weight=0.01, reconstruction_kl_warmup_epochs=5,
                  reconstruction_input_policy="existing_weak_strong_views",
                  reconstruction_target_policy="same_crop_before_randaugment")
    return config


def validate_reconstruction(config):
    if config["architecture"] != "cnn":
        raise ValueError("The matched reconstruction pilot currently requires the cnn encoder.")
    if config.get("reconstruction_mode") not in {"none", "ae", "vae"}:
        raise ValueError("Unknown reconstruction mode.")
    if config.get("reconstruction_input_policy") != "existing_weak_strong_views" or config.get("reconstruction_target_policy") != "same_crop_before_randaugment":
        raise ValueError("Reconstruction must preserve the shared pilot input views.")
    if any(float(config.get(k, 0)) < 0 for k in ("reconstruction_weight", "reconstruction_kl_weight")):
        raise ValueError("Reconstruction/KL weights must be nonnegative.")
    if int(config.get("reconstruction_kl_warmup_epochs", 5)) < 1:
        raise ValueError("KL warmup must be positive.")
    if any(float(config.get(k, 0)) for k in ("triplet_l_weight", "triplet_u_weight", "pseudo_triplet_weight")):
        raise ValueError("Triplet objectives are excluded from the first reconstruction block.")
    if config.get("pseudo_weighting", "hard") != "hard":
        raise ValueError("First reconstruction block uses the original hard pseudo-label baseline.")


class ReconstructionViews(WeakStrong):
    """Observe a reconstruction target without changing either existing view.

    Keep the original weak->crop->flip->RandAugment random draws, and retain
    the strong branch's crop just before RandAugment. Tensor conversion uses
    no RNG. No extra image is passed through the encoder.
    """
    def __call__(self, image):
        weak = self.weak(image)
        operations = self.strong.transforms
        cropped = operations[1](operations[0](image))
        target = transforms.functional.to_tensor(cropped)
        strong = cropped
        for operation in operations[2:]:
            strong = operation(strong)
        return [weak, strong, target]


def kl_to_standard_normal(mean, log_variance):
    # Average over both samples and latent coordinates; scaling is explicit.
    return 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).mean()


class ReconstructionModel(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.mode = config["reconstruction_mode"]
        self.decoder = self.log_variance = None
        dimension, seed = int(config["embedding_dim"]), int(config["seed"])
        # Auxiliary initializations cannot shift classifier/augmentation RNG.
        if self.mode != "none":
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed + 60_001)
                self.decoder = nn.Sequential(
                    nn.Linear(dimension, 64 * 4 * 4), nn.ReLU(), nn.Unflatten(1, (64, 4, 4)),
                    nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())
            if self.mode == "vae":
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(seed + 60_007)
                    # A restricted posterior: variance predicted from the common embedding mean.
                    self.log_variance = nn.Linear(dimension, dimension)

    def forward(self, images, normalize=False):
        # Classification always uses the deterministic shared embedding mean.
        return self.backbone(images, normalize=normalize)

    def decode(self, mean, generator):
        if self.decoder is None:
            raise ValueError("No decoder in this control arm.")
        if self.log_variance is None:
            return self.decoder(mean), mean.sum() * 0
        logvar = self.log_variance(mean).clamp(-10, 10)
        epsilon = torch.randn(mean.shape, generator=generator, dtype=mean.dtype).to(mean.device)
        sampled = mean + (0.5 * logvar).exp() * epsilon
        return self.decoder(sampled), kl_to_standard_normal(mean, logvar)


class ReconstructionPilot(PilotTraining):
    def __init__(self, config, train, splits):
        super().__init__(config, train, splits)
        if self.loader is None:
            raise ValueError("All reconstruction controls must see the same U batches.")
        self.unlabeled.transform = ReconstructionViews(config)
        self.latent_generator = torch.Generator().manual_seed(int(config["seed"]) + 60_409)
        self.current_epoch = 0

    def start_epoch(self, epoch):
        super().start_epoch(epoch)
        self.current_epoch = int(epoch)

    def state_dict(self):
        return {**super().state_dict(),
                "latent": self.latent_generator.get_state(), "reconstruction_epoch": self.current_epoch}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.latent_generator.set_state(state["latent"])
        self.current_epoch = state["reconstruction_epoch"]

    def step(self, model, images, labels, device):
        try:
            unlabeled = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            unlabeled = next(self.iterator)
        first, second = [x.to(device) for x in images]
        weak, strong, clean_target = [x.to(device) for x in unlabeled]
        labels = labels.to(device)
        l_size, u_size = len(labels), len(weak)
        features, logits = model(torch.cat([first, second, weak, strong]), normalize=False)
        targets = labels.repeat(2)
        ce = F.cross_entropy(logits[:2*l_size], targets)
        pseudo = reconstruction = kl = ce * 0
        accepted_count = attempted = 0
        confidence_sum, histogram = 0.0, [0] * 10
        start = 2*l_size
        if float(self.config.get("pseudo_label_weight", 0)) > 0:
            pseudo, accepted, predictions, confidence = pseudo_label_loss(
                logits[start:start+u_size], logits[start+u_size:], self.config["pseudo_label_threshold"])
            accepted_count, attempted = int(accepted.sum()), u_size
            confidence_sum = float(confidence.sum())
            histogram = torch.bincount(predictions[accepted], minlength=10).cpu().tolist()
        decoded = 0
        if self.config["reconstruction_mode"] != "none":
            prediction, kl = model.decode(features[start+u_size:], self.latent_generator)
            reconstruction = F.mse_loss(prediction, clean_target.detach())
            decoded = u_size
        beta = self.config["reconstruction_kl_weight"] * min(1, (self.current_epoch+1)/self.config["reconstruction_kl_warmup_epochs"])
        loss = ce + self.config["pseudo_label_weight"] * pseudo + self.config["reconstruction_weight"] * reconstruction + beta * kl
        metrics = {"ce_loss": float(ce.detach()), "triplet_l_loss": 0.0, "triplet_u_loss": 0.0,
                   "pseudo_label_loss": float(pseudo.detach()), "reconstruction_loss": float(reconstruction.detach()),
                   "kl_loss": float(kl.detach()), "kl_weight": beta if self.config["reconstruction_mode"] == "vae" else 0.0,
                   "processed_labeled_views": 2*l_size, "processed_unlabeled_views": 2*u_size,
                   "processed_decoder_views": decoded, "pseudo_accepted": accepted_count,
                   "pseudo_attempted": attempted, "pseudo_confidence_sum": confidence_sum,
                   **{f"pseudo_class_{i}": n for i, n in enumerate(histogram)}}
        return loss, logits[:2*l_size], targets, metrics


def auxiliary_metadata(model):
    base = sum(p.numel() for p in model.backbone.parameters())
    return {"trainable_parameter_count": sum(p.numel() for p in model.parameters()),
            "inference_parameter_count": base,
            "decoder_parameter_count": sum(p.numel() for p in model.decoder.parameters()) if model.decoder else 0,
            "initial_decoder_sha256": model_fingerprint(model.decoder) if model.decoder else None,
            "initial_variance_sha256": model_fingerprint(model.log_variance) if model.log_variance else None,
            "reconstruction_target": "same_crop_before_randaugment",
            "reconstruction_reduction": "mean_batch_channels_pixels", "kl_reduction": "mean_batch_latent_dimensions"}


def export_reconstruction_examples(run_dir, output_dir, data_root="data"):
    """Validation-only figure; neither selection nor training uses these images."""
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision import datasets
    from .models import build_model
    from PIL import Image

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    config = json.loads((run_dir / "config.json").read_text())
    if not config.get("reconstruction_pilot") or config["reconstruction_mode"] == "none":
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "reconstruction.png"
    if destination.exists():
        return
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    model = ReconstructionModel(build_model("cnn", embedding_dim=config["embedding_dim"],
                                            normalization=config.get("normalization", "batchnorm")), config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    splits = json.loads((run_dir / "splits.json").read_text())
    base = datasets.CIFAR10(data_root, train=True, download=False)
    indices = splits["validation"][:8]
    transform = ReconstructionViews(config)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        views = [transform(Image.fromarray(base.data[i])) for i in indices]
    strong = torch.stack([v[1] for v in views])
    clean = torch.stack([v[2] for v in views])
    with torch.no_grad():
        mean, _ = model(strong)
        decoded, _ = model.decode(mean, torch.Generator().manual_seed(0))
    average = clean.new_tensor(config["mean"])[None,:,None,None]
    std = clean.new_tensor(config["std"])[None,:,None,None]
    fig, axes = plt.subplots(len(indices), 3, figsize=(5, len(indices)*1.3), squeeze=False)
    for col, (batch, title) in enumerate(((clean, "Target before RandAugment"), (strong*std+average, "Existing strong input"), (decoded, "Reconstruction"))):
        for row in range(len(indices)):
            axes[row, col].imshow(batch[row].clamp(0,1).permute(1,2,0).numpy(), interpolation="nearest")
            axes[row, col].axis("off")
        axes[0,col].set_title(title)
    fig.suptitle(f"{run_dir.name}\nValidation examples; VAE sampling seed=0", fontsize=9)
    fig.tight_layout(rect=(0,0,1,0.95))
    fig.savefig(destination, dpi=180)
    plt.close(fig)
