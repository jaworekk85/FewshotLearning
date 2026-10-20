import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fewshot_metric.models import build_model


@unittest.skipUnless(importlib.util.find_spec("sklearn") and importlib.util.find_spec("matplotlib"),
                     "Plot dependencies are isolated in .venv-analysis")
class RepresentationExportTests(unittest.TestCase):
    def test_audit_loads_groupnorm_and_no_normalization_checkpoints(self):
        torch.set_num_threads(2)
        from fewshot_metric.representations import analyze_run
        rng=np.random.default_rng(11)
        base=SimpleNamespace(data=rng.integers(0,256,(100,32,32,3),dtype=np.uint8),
                             targets=(np.arange(100)%10).tolist(),classes=[f"class{i}" for i in range(10)])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for normalization in ("none","groupnorm"):
                config=json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())
                config.update(embedding_dim=16,normalization=normalization,synthetic_smoke=False)
                run=root/normalization
                run.mkdir()
                (run/"config.json").write_text(json.dumps(config))
                (run/"splits.json").write_text(json.dumps({"train":list(range(20)),"validation":list(range(20,100))}))
                model=build_model("cnn",embedding_dim=16,normalization=normalization)
                torch.save(model.state_dict(),run/"model.pt")
                with patch.dict(os.environ,{"MPLCONFIGDIR":str(root/"mpl")}),patch("fewshot_metric.representations.datasets.CIFAR10",return_value=base):
                    result=analyze_run(run,root/"analysis",root,tsne=False)
                self.assertEqual(result["normalization"],normalization)
                self.assertEqual(result["probe_training_labels"],20)

    def test_vae_checkpoint_exports_validation_reconstruction_examples(self):
        torch.set_num_threads(2)
        from fewshot_metric.reconstruction import ReconstructionModel, reconstruction_config, export_reconstruction_examples
        config = reconstruction_config(json.loads(Path("configs/cifar10_pilot_ssl.json").read_text()), "dvae")
        config["embedding_dim"] = 16
        rng = np.random.default_rng(9)
        base = SimpleNamespace(data=rng.integers(0,256,(32,32,32,3),dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/"vae"
            run.mkdir()
            (run/"config.json").write_text(json.dumps(config))
            (run/"splits.json").write_text(json.dumps({"train":list(range(16)),"validation":list(range(16,32))}))
            model = ReconstructionModel(build_model("cnn",embedding_dim=16),config)
            torch.save({"model":model.state_dict()},run/"latest.pt")
            with patch.dict(os.environ,{"MPLCONFIGDIR":str(root/"mpl")}), patch("torchvision.datasets.CIFAR10",return_value=base):
                export_reconstruction_examples(run,root/"pictures",root)
            self.assertTrue((root/"pictures/reconstruction.png").is_file())

    def test_audit_exports_plots_embeddings_and_reuses_matching_checkpoint(self):
        torch.set_num_threads(2)
        from fewshot_metric.representations import analyze_many
        config = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())
        config.update(embedding_dim=16, synthetic_smoke=False)
        rng = np.random.default_rng(9)
        base = SimpleNamespace(data=rng.integers(0, 256, (100, 32, 32, 3), dtype=np.uint8),
                               targets=(np.arange(100) % 10).tolist(), classes=[f"class{i}" for i in range(10)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/"future_variant"
            run.mkdir()
            (run/"config.json").write_text(json.dumps(config))
            (run/"splits.json").write_text(json.dumps({"train": list(range(20)), "validation": list(range(20, 100))}))
            torch.save(build_model("cnn", embedding_dim=16).state_dict(), run/"model.pt")
            with patch.dict(os.environ, {"MPLCONFIGDIR": str(root/"mpl")}), patch("fewshot_metric.representations.datasets.CIFAR10", return_value=base) as loader:
                result = analyze_many([run], root/"analysis", data_root=root)
                cached = analyze_many([run], root/"analysis", data_root=root)
                self.assertEqual(loader.call_count, 1)
            self.assertEqual(result, cached)
            self.assertEqual(result[0]["probe_training_labels"], 20)
            self.assertEqual(result[0]["validation_size"], 80)
            for filename in ("metrics.json", "embeddings.npz", "pca.png", "pca.svg", "tsne.png", "tsne.svg"):
                self.assertTrue((root/"analysis/future_variant"/filename).is_file())
            self.assertTrue((root/"analysis/comparison.csv").exists())
            with np.load(root/"analysis/future_variant/embeddings.npz") as exported:
                self.assertEqual(exported["validation"].shape, (80, 16))
                self.assertEqual(exported["validation_indices"].tolist(), list(range(20, 100)))


if __name__ == "__main__":
    unittest.main()
