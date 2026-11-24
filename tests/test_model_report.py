"""Evaluation script: table content, figures with recreate commands, no test data touched."""

import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from sepsis_warning import cli, model_report, policy
from sepsis_warning.data import generate_synthetic
from sepsis_warning.evaluation import run_frozen_evaluation
from sepsis_warning.importance import PATTERN, STATIC_FAMILY, VALUES, feature_groups, grouped_permutation_importance


def synthetic_run(folder: Path, stays: int = 40, hours: int = 20, attributes: bool = False) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for stay in range(stays):
        septic = stay % 4 == 0
        for hour in range(hours):
            label = int(septic and hour >= 12)
            base = 0.6 if label else 0.03
            row = {"patient_id": f"s{stay}", "hospital": "A", "hour": hour, "label": label,
                   "p_good": min(1, max(0, base + rng.normal(0, 0.05))),
                   "p_weak": float(rng.uniform(0, 0.3)), "sirs_score": float(label * 2 + rng.integers(0, 2))}
            if attributes:
                # Ages 30-69 and alternating blocks of four: septic stays fall in both sexes and three age bands.
                row.update({"Age": 30 + stay, "Gender": (stay // 4) % 2})
            rows.append(row)
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "validation_scores.csv", index=False)
    pd.DataFrame([
        {"model": "good", "threshold": 0.3, "brier_score": 0.01, "normalized_hourly_utility": 0.8,
         "false_alert_events_per_nonseptic_patient_day": 0.1, "early_detection_recall": 0.9},
        {"model": "weak", "threshold": 0.25, "brier_score": 0.05, "normalized_hourly_utility": 0.1,
         "false_alert_events_per_nonseptic_patient_day": 0.2, "early_detection_recall": 0.1},
    ]).to_csv(folder / "validation_models.csv", index=False)
    (folder / "selection_metrics.json").write_text(json.dumps({
        "selected_model": "good",
        "reference_baselines": {"sirs": {"threshold": 2.0, "normalized_hourly_utility": 0.3,
                                         "false_alert_events_per_nonseptic_patient_day": 0.05,
                                         "early_detection_recall": 0.5}}}))
    return folder


class ModelReportTests(unittest.TestCase):
    def test_table_and_figures(self):
        with tempfile.TemporaryDirectory() as temp:
            run = synthetic_run(Path(temp) / "run")
            out = Path(temp) / "out"
            manifest = model_report.build_report(run, out, bootstrap=30)
            table = pd.read_csv(out / "metrics_table.csv")
            self.assertEqual(set(table["model"]), {"good", "weak", "sirs_rule"})
            row = table.set_index("model").loc["good"]
            self.assertGreater(row["recall"], 0.9)
            self.assertLessEqual(row["f1_ci_low"], row["f1"])
            self.assertGreaterEqual(row["f1_ci_high"], row["f1"])
            self.assertEqual(table.set_index("model").loc["good", "role"], "selected")
            for figure in manifest["figures"]:
                self.assertTrue((out / figure["file"]).stat().st_size > 1000)
                self.assertIn("python -m sepsis_warning model-report", figure["recreate"])
            self.assertIn("F1", (out / "FIGURES.md").read_text())
            self.assertTrue(manifest["skipped"])  # no loss history in this run
            self.assertTrue((out / "environment.json").is_file())

    def test_counts_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            run = model_report._read_run(synthetic_run(Path(temp) / "run"))
            table = model_report.metrics_table(run, draws=0).set_index("model")
            scores = run["scores"]
            predicted = scores["p_good"] >= 0.3
            self.assertEqual(int(table.loc["good", "tp"]), int((predicted & (scores["label"] == 1)).sum()))
            self.assertEqual(int(table.loc["good", "fp"]), int((predicted & (scores["label"] == 0)).sum()))
            self.assertTrue(pd.isna(table.loc["good", "f1_ci_low"]))

    def test_refuses_missing_inputs_and_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty"
            empty.mkdir()
            with self.assertRaises(model_report.ReportInputError):
                model_report.build_report(empty, Path(temp) / "o")
            run = synthetic_run(Path(temp) / "run")
            busy = Path(temp) / "busy"
            busy.mkdir()
            (busy / "x").write_text("x")
            with self.assertRaises(ValueError):
                model_report.build_report(run, busy)

    def test_cli_command(self):
        with tempfile.TemporaryDirectory() as temp:
            run = synthetic_run(Path(temp) / "run")
            out = Path(temp) / "out"
            argv = ["sepsis_warning", "model-report", "--run", str(run), "--output", str(out), "--bootstrap", "10"]
            with mock.patch("sys.argv", argv):
                cli.main()
            self.assertTrue((out / "fig_pr_curves.png").is_file())

    def test_evaluation_folder_needed_for_test_figures(self):
        with tempfile.TemporaryDirectory() as temp:
            run = synthetic_run(Path(temp) / "run")
            with self.assertRaises(model_report.ReportInputError):
                model_report.build_report(run, Path(temp) / "o", evaluation_dir=Path(temp) / "nope", bootstrap=5)

    def test_utility_and_its_interval_agree_with_the_policy_code(self):
        with tempfile.TemporaryDirectory() as temp:
            scores = model_report._read_run(synthetic_run(Path(temp) / "run"))["scores"]
            scores = scores.sample(frac=1, random_state=0).reset_index(drop=True)  # row order must not matter
            decisions = scores["p_weak"] >= 0.25
            fast = float(model_report._utility(model_report._stay_utilities(scores, decisions).sum(axis=0)))
            self.assertAlmostEqual(fast, policy.normalized_utility(scores.assign(hourly_positive=decisions.astype(int))))
            interval = model_report.bootstrap_intervals(scores, "p_weak", 0.25, 60, 1)
            self.assertLess(interval["utility_ci_low"], fast)
            self.assertGreater(interval["utility_ci_high"], fast)
            # A row without a recorded utility gets the computed one, and the table prints its interval.
            row = model_report._row(scores, "p_weak", "weak", "candidate", 0.25, {}, 60, 1)
            self.assertAlmostEqual(row["normalized_hourly_utility"], fast)
            cell = model_report.markdown_table(pd.DataFrame([row]).reindex(columns=model_report.TABLE_COLUMNS))
            self.assertIn(f"{fast:.3f} ({interval['utility_ci_low']:.3f}-{interval['utility_ci_high']:.3f})", cell)

    def test_subgroup_table_counts_scores_and_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            scores = model_report._read_run(synthetic_run(Path(temp) / "run", attributes=True))["scores"]
            table = model_report.subgroup_table(scores, "p_weak", 0.25, 6, model_report.stay_attributes(scores),
                                                scope="validation", draws=40, seed=1, min_septic_stays=3)
            rows = table.set_index(["attribute", "group"])
            everyone = rows.loc[("all", "all stays")]
            reference = policy.summarize(policy.apply_policy(scores.rename(columns={"p_weak": "probability"}), 0.25, 6))
            self.assertAlmostEqual(everyone["utility"], reference["normalized_hourly_utility"])
            self.assertAlmostEqual(everyone["false_alerts_per_day"], reference["false_alert_events_per_nonseptic_patient_day"])
            self.assertGreater(everyone["false_alerts_per_day"], 0)
            sexes = table[(table["attribute"] == "sex") & (table["group"] != model_report.GAP_GROUP)]
            self.assertEqual((int(sexes["stays"].sum()), int(sexes["septic_stays"].sum())), (40, 10))
            self.assertEqual(int(sexes["hours"].sum()), len(scores))
            # One septic stay aged 65-79: counted, not scored, and left out of the gap.
            small = rows.loc[("age_band", "65-79")]
            self.assertEqual((int(small["stays"]), int(small["septic_stays"]), bool(small["sufficient"])), (5, 1, False))
            self.assertTrue(pd.isna(small["recall"]))
            # The whole-cohort row is the reference and is scored even when no subgroup is large enough.
            strict = model_report.subgroup_table(scores, "p_weak", 0.25, 6, model_report.stay_attributes(scores),
                                                 scope="validation", draws=0, min_septic_stays=50)
            self.assertEqual(list(strict.loc[strict["sufficient"], "group"]), ["all stays"])
            self.assertAlmostEqual(strict.loc[0, "utility"], everyone["utility"])
            scored = table[(table["attribute"] == "age_band") & table["sufficient"] & (table["group"] != model_report.GAP_GROUP)]
            gap = rows.loc[("age_band", model_report.GAP_GROUP)]
            self.assertEqual(list(scored["group"]), ["<45", "45-64"])
            self.assertAlmostEqual(gap["utility"], scored["utility"].max() - scored["utility"].min())
            self.assertLessEqual(gap["utility_ci_low"], gap["utility_ci_high"])
            text = model_report.subgroup_markdown(table, "weak", 0.25, 3, "on the validation split")
            self.assertIn("not scored (fewer than 3 septic stays)", text)
            self.assertIn("Not checked", text)


class FakeModel:
    """Scores from the last heart rate only."""

    def predict_proba(self, features):
        risk = 1 / (1 + np.exp(-(features["HR__last"].to_numpy(dtype=float) - 100) / 5))
        return np.column_stack([1 - risk, risk])


class ImportanceTests(unittest.TestCase):
    def test_only_the_columns_the_model_uses_matter(self):
        rng = np.random.default_rng(0)
        table = pd.DataFrame({
            "HR__last": rng.normal(100, 15, 600), "HR__missing_now": rng.integers(0, 2, 600).astype(float),
            "Temp__last": rng.normal(37, 1, 600), "Temp__hours_since_seen": rng.integers(0, 9, 600).astype(float),
            "Age": rng.integers(20, 90, 600).astype(float),
        })
        table["label"] = (table["HR__last"] + rng.normal(0, 5, 600) > 110).astype(int)
        columns = [name for name in table if name != "label"]
        before = table.copy()
        result = grouped_permutation_importance(FakeModel(), table, columns, repeats=2, seed=0)
        pd.testing.assert_frame_equal(table, before)  # shuffling works on a copy
        drop = result.set_index(["level", "group"])["ap_drop_mean"]
        self.assertGreater(drop[("variable", "HR")], 0.2)
        self.assertEqual((drop[("variable", "Temp")], drop[("variable", "Age")]), (0.0, 0.0))
        self.assertGreater(drop[("family", VALUES)], 0.2)
        self.assertEqual((drop[("family", PATTERN)], drop[("family", STATIC_FAMILY)]), (0.0, 0.0))
        self.assertEqual(result["group"].iloc[0], "HR")  # largest first
        groups = feature_groups(columns)
        self.assertEqual(groups["variable"]["HR"], ["HR__last", "HR__missing_now"])
        self.assertEqual(groups["family"][PATTERN], ["HR__missing_now", "Temp__hours_since_seen"])
        for level in groups.values():
            self.assertEqual(sorted(name for names in level.values() for name in names), sorted(columns))


class TrainEvaluateReportTests(unittest.TestCase):
    """The real chain on the synthetic fixture: train, the one frozen evaluation, then the report."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        data = cls.root / "data"
        generate_synthetic(data, source_size=80, external_size=20)
        (data / "SYNTHETIC.json").write_text("{}")
        cls.run_dir, cls.evaluation, cls.out = cls.root / "run", cls.root / "evaluation", cls.root / "report"
        cli.run_experiment(Namespace(
            data=data, splits=None, output=cls.run_dir, seed=42, window=6, budget=0.2, cooldown=6,
            splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False, importance_repeats=1))
        cls.summary = run_frozen_evaluation(data=data, model=cls.run_dir / "model.joblib", output=cls.evaluation)
        cls.manifest = model_report.build_report(cls.run_dir, cls.out, evaluation_dir=cls.evaluation, bootstrap=20,
                                                 min_subgroup_septic_stays=1)
        cls.predictions = pd.read_csv(cls.evaluation / "held_out_predictions.csv")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_test_table_is_labelled_as_test_and_holds_the_baselines(self):
        text = (self.out / "test_metrics_table.md").read_text()
        self.assertIn("on the test splits", text)
        self.assertNotIn("on the validation split, each model", text)
        self.assertIn("on the validation split, each model", (self.out / "metrics_table.md").read_text())
        table = pd.read_csv(self.out / "test_metrics_table.csv")
        selected = self.summary["selected_model"]
        for split in ("internal_test", "external_test"):
            rows = table[table["role"] == split].set_index("model")
            self.assertTrue({selected, "sirs_rule (>= 2 criteria, standard)", "always_no (never alerts)"} <= set(rows.index))
            frame = self.predictions[self.predictions["split"] == split]
            flagged = frame["sirs_score"] >= 0.5
            sirs = rows.loc["sirs_rule (>= 2 criteria, standard)"]
            self.assertEqual((int(sirs["tp"]), int(sirs["fp"])),
                             (int((flagged & (frame["label"] == 1)).sum()), int((flagged & (frame["label"] == 0)).sum())))
            self.assertFalse(pd.isna(sirs["normalized_hourly_utility"]))
            nothing = rows.loc["always_no (never alerts)"]
            self.assertEqual((nothing["recall"], nothing["normalized_hourly_utility"]), (0.0, 0.0))
            self.assertAlmostEqual(nothing["accuracy"], 1 - frame["label"].mean())
            model = rows.loc[selected]
            self.assertAlmostEqual(model["normalized_hourly_utility"],
                                   self.summary["splits"][split]["normalized_hourly_utility"])
            self.assertLessEqual(model["utility_ci_low"], model["normalized_hourly_utility"])
            self.assertGreaterEqual(model["utility_ci_high"], model["normalized_hourly_utility"])

    def test_subgroup_tables_cover_sex_age_and_hospital(self):
        validation = pd.read_csv(self.out / "subgroup_table.csv")
        self.assertEqual(set(validation["scope"]), {"validation"})
        self.assertTrue({"all", "sex", "age_band"} <= set(validation["attribute"]))
        held = pd.read_csv(self.out / "test_subgroup_table.csv")
        hospitals = held[held["attribute"] == "hospital"].set_index("group")
        self.assertTrue({"A (internal_test)", "B (external_test)"} <= set(hospitals.index))
        stays = self.predictions.groupby("split")["patient_id"].nunique()
        self.assertEqual(int(hospitals.loc["A (internal_test)", "stays"]), stays["internal_test"])
        self.assertEqual(int(hospitals.loc["B (external_test)", "stays"]), stays["external_test"])
        self.assertIn("Not checked", (self.out / "test_subgroup_table.md").read_text())

    def test_importance_is_written_by_train_and_drawn_by_the_report(self):
        importance = pd.read_csv(self.run_dir / "feature_importance.csv")
        columns = json.loads((self.run_dir / "config.json").read_text())["feature_columns"]
        by_level = importance.groupby("level")["columns"].agg(["size", "sum"])
        self.assertEqual(tuple(by_level.loc["variable"]), (38, len(columns)))   # 34 measurements + 4 static values
        self.assertEqual(tuple(by_level.loc["family"]), (3, len(columns)))
        models = pd.read_csv(self.run_dir / "validation_models.csv").set_index("model")
        self.assertAlmostEqual(importance["baseline_average_precision"].iloc[0],
                               models.loc[self.summary["selected_model"], "average_precision"])
        self.assertIn("fig_feature_importance.png", {figure["file"] for figure in self.manifest["figures"]})
        self.assertTrue((self.out / "feature_importance.md").is_file())
        self.assertNotIn("F10", {item["id"] for item in self.manifest["skipped"]})

    def test_an_older_evaluate_folder_still_gives_a_report(self):
        older = self.root / "older-evaluation"
        shutil.copytree(self.evaluation, older)
        self.predictions.drop(columns=["sirs_score", "Age", "Gender"]).to_csv(older / "held_out_predictions.csv", index=False)
        out = self.root / "older-report"
        manifest = model_report.build_report(self.run_dir, out, evaluation_dir=older, bootstrap=5)
        self.assertTrue({"T1", "T3"} <= {item["id"] for item in manifest["skipped"]})
        table = pd.read_csv(out / "test_metrics_table.csv")
        self.assertFalse(table["model"].str.startswith("sirs_rule").any())
        self.assertTrue(table["model"].eq("always_no (never alerts)").any())
        self.assertFalse((out / "test_subgroup_table.csv").exists())


if __name__ == "__main__":
    unittest.main()
