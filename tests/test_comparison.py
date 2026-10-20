import csv
import json
import tempfile
import unittest
from pathlib import Path

from fewshot_metric.comparison import collect_comparison, write_comparison


class ComparisonTests(unittest.TestCase):
    def test_one_table_preserves_equal_encoder_exposure_and_extra_decoder_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,mode in [("cnn_pseudo",None),("cnn_ae","dae_pseudo")]:
                run=root/name
                run.mkdir()
                config={"epochs":20,"seed":0,"budget_per_class":50,"labeled_train_size":500,
                        "method":"ce_2v_pseudo","validation_size":1000,"initial_model_sha256":"same"}
                if mode:
                    config.update(reconstruction_variant=mode,reconstruction_input_policy="existing_weak_strong_views")
                (run/"config.json").write_text(json.dumps(config))
                (run/"splits.json").write_text(json.dumps({"train":list(range(500)),"validation":list(range(500,1500))}))
                (run/"metrics.json").write_text(json.dumps({"processed_labeled_views":1915360,
                    "processed_unlabeled_views":1960960,"processed_decoder_views":980480 if mode else 0,
                    "total_updates":7660,"val_accuracy":0.5,"test_accuracy":0.49,"elapsed_seconds":5000,
                    "trainable_parameter_count":267117 if mode else 111498,"inference_parameter_count":111498}))
            rows=write_comparison([root],root/"report")
            self.assertEqual(len(rows),2)
            self.assertEqual({r["total_encoder_views"] for r in rows},{3876320})
            self.assertEqual({r["initial_model_sha256"] for r in rows},{"same"})
            self.assertEqual({r["decoder_views"] for r in rows},{0,980480})
            self.assertEqual({r["inference_parameters"] for r in rows},{111498})
            with (root/"report.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),2)
            self.assertIn("267117",(root/"report.json").read_text())

    def test_active_learning_keeps_schedule_and_real_incomplete_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root/"cnn_active"
            run.mkdir()
            (run/"config.json").write_text(json.dumps({"epochs":20,"seed":0,"budget_per_class":30,
                "initial_label_budget":300,"active_final_budget":500,"active_strategy":"random"}))
            (run/"splits.json").write_text(json.dumps({"train":list(range(340)),"validation":list(range(500,1500))}))
            with (run/"history.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["epoch","updates","val_accuracy","processed_labeled_views","processed_unlabeled_views"])
                writer.writeheader()
                writer.writerow({"epoch":1,"updates":383,"val_accuracy":0.4,"processed_labeled_views":76400,"processed_unlabeled_views":98048})
            row=collect_comparison([root])[0]
            self.assertEqual(row["budget_schedule"],"gradual")
            self.assertEqual(row["initial_label_budget"],300)
            self.assertEqual(row["final_label_budget"],500)
            self.assertEqual(row["current_label_budget"],340)
            self.assertEqual(row["labeled_encoder_views"],76400)
            self.assertIsNone(row["test_accuracy"])


if __name__=="__main__":
    unittest.main()
