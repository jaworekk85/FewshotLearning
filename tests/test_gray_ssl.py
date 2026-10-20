import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from fewshot_metric.models import build_model
from fewshot_metric.pilot import PilotTraining, select_method, view_transform
from fewshot_metric.supervised import build_datasets, run_supervised, validate_config


class FakeGray:
    """In-memory fixture: no downloads, no persistent research results."""
    def __init__(self, transform=None):
        self.data=torch.from_numpy(np.random.default_rng(7).integers(0,256,(120,28,28),dtype=np.uint8))
        self.targets=torch.arange(120)%10
        self.classes=[str(i) for i in range(10)]
        self.transform=transform

    def __len__(self):return len(self.targets)

    def __getitem__(self,i):
        image=Image.fromarray(self.data[i].numpy())
        return self.transform(image) if self.transform else image,int(self.targets[i])


def fake_source(config,root,train,download=False,transform=None):return FakeGray(transform)


class GraySSLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base=json.loads(Path("configs/fashion_mnist_ssl.json").read_text())["base_config"]

    def test_native_shape_and_parameter_count(self):
        model=build_model("gray_cnn",embedding_dim=64).eval()
        z,logits=model(torch.zeros(2,1,28,28))
        self.assertEqual(tuple(z.shape),(2,64))
        self.assertEqual(tuple(logits.shape),(2,10))
        self.assertEqual(sum(p.numel() for p in model.parameters()),38378)

    def test_nested_budgets_common_validation_and_no_hidden_labels_in_u(self):
        configs=[select_method({**self.base,"budget_per_class":n,"validation_per_class":2},"ce_2v_pseudo_soft") for n in (2,4)]
        with patch("fewshot_metric.supervised.source_dataset",side_effect=fake_source):
            smaller,_,a=build_datasets(configs[0],Path("unused"),False)
            larger,_,b=build_datasets(configs[1],Path("unused"),False)
            _,_,full=build_datasets({**configs[0],"budget_per_class":"all"},Path("unused"),False)
        self.assertEqual(a["validation"],b["validation"])
        self.assertEqual(a["validation"],full["validation"])
        self.assertTrue(set(a["train"]) < set(b["train"]))
        self.assertEqual(len(full["train"]),100)
        self.assertFalse(set(full["train"]) & set(full["validation"]))
        pilot=PilotTraining(configs[0],smaller,a)
        self.assertIsNone(pilot.unlabeled.labels)
        self.assertFalse(set(pilot.unlabeled.indices)&set(a["train"]+a["validation"]))
        self.assertEqual(tuple(smaller[0][0][0].shape),(1,28,28))

    def test_digit_augmentation_does_not_flip(self):
        from torchvision.transforms import RandomHorizontalFlip
        for strong in (False,True):
            digit=view_transform({**self.base,"dataset":"mnist"},strong)
            fashion=view_transform(self.base,strong)
            self.assertFalse(any(isinstance(x,RandomHorizontalFlip) for x in digit.transforms))
            self.assertTrue(any(isinstance(x,RandomHorizontalFlip) for x in fashion.transforms))

    def test_plan_is_small_and_matched(self):
        import sys
        sys.path.insert(0,str(Path("scripts").resolve()))
        from run_gray_ssl import build_jobs
        plan=json.loads(Path("configs/fashion_mnist_ssl.json").read_text())
        jobs=build_jobs(plan)
        self.assertEqual(len(jobs),16)
        self.assertEqual(len({j["name"] for j in jobs}),16)
        for budget in (10,30,50):
            configs=[j["config"] for j in jobs if j["config"]["budget_per_class"]==budget]
            self.assertEqual(len(configs),5)
            self.assertEqual({c["epochs"]*c["steps_per_epoch"] for c in configs},{2000})
            self.assertTrue(all(c["labeled_drop_last"] and c["batch_size"]==64 for c in configs))
        with self.assertRaises(ValueError):validate_config({**self.base,"architecture":"cnn"})

    def test_all_methods_train_and_soft_resume_is_exact(self):
        methods=[("ce_2v",{}),("ce_2v_pseudo",{}),("ce_2v_pseudo_soft",{}),
            ("ce_2v_pseudo_soft_mined_triplet",{}),
            ("ce_2v_pseudo_soft",{"contrastive_weight":.05,"contrastive_pseudo_pairs":True})]
        with tempfile.TemporaryDirectory() as tmp,patch("fewshot_metric.supervised.source_dataset",side_effect=fake_source),patch("builtins.print"):
            root=Path(tmp)
            for i,(method,extra) in enumerate(methods):
                config=select_method({**self.base,"budget_per_class":2,"validation_per_class":2,
                    "batch_size":8,"unlabeled_batch_size":8,"epochs":4,"steps_per_epoch":2},method)
                config.update(extra)
                result=run_supervised(config,root/str(i),Path("unused"),"cpu")
                self.assertEqual(result["dataset"],"fashion_mnist")
                self.assertEqual(result["processed_labeled_views"],128)
                self.assertEqual(result["processed_unlabeled_views"],0 if i==0 else 128)
                self.assertEqual(result["trainable_parameter_count"],38378)
                self.assertTrue(np.isfinite(result["test_loss"]))
                if i==2:
                    paused=root/"resumed"
                    self.assertIsNone(run_supervised(config,paused,Path("unused"),"cpu",stop_after_epoch=2))
                    resumed=run_supervised(config,paused,Path("unused"),"cpu",resume=True)
                    self.assertEqual(result["test_accuracy"],resumed["test_accuracy"])
                    original=torch.load(root/str(i)/"model.pt",weights_only=True)
                    actual=torch.load(paused/"model.pt",weights_only=True)
                    self.assertTrue(all(torch.equal(original[k],actual[k]) for k in original))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"Requires analysis environment")
    def test_representation_audit_is_grayscale_and_omits_pca_figures(self):
        from fewshot_metric.representations import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"gray";run.mkdir()
            config={**self.base,"budget_per_class":2,"validation_per_class":2}
            with patch("fewshot_metric.supervised.source_dataset",side_effect=fake_source):
                _,_,splits=build_datasets(config,root,False)
            (run/"config.json").write_text(json.dumps(config))
            (run/"splits.json").write_text(json.dumps(splits))
            torch.save(build_model("gray_cnn",embedding_dim=64).state_dict(),run/"model.pt")
            with patch("fewshot_metric.representations.source_dataset",side_effect=fake_source):
                result=analyze_run(run,root/"audit",root,tsne=False,pca_plot=False)
            self.assertEqual(result["dataset"],"fashion_mnist")
            self.assertEqual(result["probe_training_labels"],20)
            self.assertFalse(list((root/"audit/gray").glob("pca.*")))
            with np.load(root/"audit/gray/embeddings.npz") as a:
                self.assertEqual(a["validation"].shape,(20,64))


if __name__=="__main__":unittest.main()
