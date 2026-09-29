"""Small synthetic study inputs; no real GRAPE scores, weights, or experiments."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from glaboost.data import GrapeDataset, VisitInput
from glaboost.study import (
    create_study_report, describe_cohort, ensure_outside_raw, load_verified_scores,
    score_metadata_path, sha256_file, write_visit_scores, _refresh_project_readme,
)


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_glaboost_study_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.root = self.directory / "raw_grape"
        (self.root / "files").mkdir(parents=True)
        (self.root / "files" / "VF and clinical information.xlsx").write_bytes(b"SYNTHETIC workbook fixture")
        images = self.root / "extracted" / "CFPs"
        images.mkdir(parents=True)
        visits, outcomes = [], {}
        for patient in range(8):
            for side in ("OD", "OS"):
                eye = f"{patient}_{side}"
                outcomes[eye] = {"plr2": patient % 2, "plr3": patient % 2,
                                 "md_slope": int(patient % 3 == 0)}
                for index, elapsed in enumerate((0, 0.6, 2.1), 1):
                    sample = f"{eye}_{index}"
                    image = images / f"{sample}.jpg"
                    image.write_bytes(f"SYNTHETIC {sample}".encode())
                    visits.append(VisitInput(sample, image=image, patient_id=str(patient),
                                             eye_id=eye, time_years=elapsed, structured={"iop": 15.0}))
        # A later visit with no CFP and another eye with too few CFPs.
        visits.append(VisitInput("0_OD_4", patient_id="0", eye_id="0_OD", time_years=3.0))
        outcomes["8_OD"] = {"plr2": 0, "plr3": 0, "md_slope": 0}
        visits.append(VisitInput("8_OD_1", patient_id="8", eye_id="8_OD", time_years=0.0))
        self.dataset = GrapeDataset(visits, outcomes)
        self.visits = self.dataset.image_visits()
        self.scores = [0.35 + 0.04 * int(v.patient_id) + 0.02 * v.time_years for v in self.visits]
        self.model = self.directory / "synthetic_model"
        self.model.mkdir()
        (self.model / "model.json").write_text("SYNTHETIC model checksum fixture")
        self.model_metadata = {
            "implementation": "SYNTHETIC TEST; not a trained diagnosis model",
            "target": {"0": "normal", "1": "glaucoma"},
            "config": {"use_image": True, "use_text": False, "use_structured": False,
                       "use_human_risk": False, "use_human_confidence": False},
            "model_sha256": sha256_file(self.model / "model.json"),
        }
        (self.model / "metadata.json").write_text(json.dumps(self.model_metadata))
        self.scores_path = self.directory / "synthetic_scores.csv"

    def write_scores(self, **kwargs):
        return write_visit_scores(self.scores_path, self.visits, self.scores,
                                  model_directory=self.model, grape_root=self.root, **kwargs)

    def update_csv(self, mutate):
        with self.scores_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = list(reader)
        rows = mutate(rows)
        with self.scores_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        path = score_metadata_path(self.scores_path)
        metadata = json.loads(path.read_text())
        metadata["scores_sha256"] = sha256_file(self.scores_path)
        path.write_text(json.dumps(metadata))

    def test_round_trip_provenance_does_not_assume_external_validation(self):
        self.write_scores()
        rows, metadata = load_verified_scores(self.scores_path, self.dataset, self.root)
        self.assertEqual(len(rows), len(self.visits))
        self.assertEqual(metadata["detector_training_data"]["grape_overlap"], "unknown")
        self.assertEqual(len(metadata["source"]["image_sha256"]), len(self.visits))
        self.assertIn("model_metadata_sha256", metadata)

    def test_no_overlap_requires_description_and_known_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Document"):
            self.write_scores(grape_overlap="none")
        with self.assertRaisesRegex(ValueError, "trained/selected"):
            self.write_scores(grape_overlap="present")
        self.assertFalse(self.scores_path.exists())
        self.write_scores(grape_overlap="none", training_data_description="SYNTHETIC external fixture")
        _, metadata = load_verified_scores(self.scores_path, self.dataset, self.root)
        self.assertEqual(metadata["detector_training_data"]["grape_overlap"], "none")

    def test_checksum_detects_changed_scores(self):
        self.write_scores()
        with self.scores_path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_verified_scores(self.scores_path, self.dataset, self.root)

    def test_missing_visit_cannot_silently_change_the_cohort(self):
        self.write_scores()
        self.update_csv(lambda rows: rows[:-1])
        with self.assertRaisesRegex(ValueError, "every eligible"):
            load_verified_scores(self.scores_path, self.dataset, self.root)

    def test_score_metadata_must_match_raw_patient_eye_and_time(self):
        self.write_scores()
        def mutate(rows):
            rows[0]["time_years"] = "99"
            return rows
        self.update_csv(mutate)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_verified_scores(self.scores_path, self.dataset, self.root)

    def test_source_checksums_detect_changed_raw_input(self):
        self.write_scores()
        self.visits[0].image.write_bytes(b"CHANGED synthetic image")
        with self.assertRaisesRegex(ValueError, "CFP changed"):
            load_verified_scores(self.scores_path, self.dataset, self.root)

    def test_protect_raw_even_through_symlink_and_reject_score_overwrite(self):
        link = self.directory / "raw_alias"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside raw"):
            ensure_outside_raw(link / "report.csv", self.root)
        self.write_scores()
        before = self.scores_path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.write_scores()
        self.assertEqual(self.scores_path.read_bytes(), before)

    def test_followup_span_and_incomplete_image_window_are_distinct(self):
        from glaboost.longitudinal import prepare_eye_records
        self.write_scores()
        rows, metadata = load_verified_scores(self.scores_path, self.dataset, self.root)
        eyes = prepare_eye_records(rows, self.dataset.progression_labels)
        cohort = describe_cohort(self.dataset, eyes, metadata["model_info"])
        self.assertEqual(cohort["n_patients"], 8)
        self.assertEqual(cohort["n_eyes"], 16)
        self.assertAlmostEqual(cohort["followup_months"]["median"], 25.2)
        self.assertEqual(cohort["observation_window"]["eyes_last_cfp_before_last_recorded_visit"], 1)
        self.assertEqual(cohort["exclusions"][0]["eye_id"], "8_OD")

    def test_synthetic_end_to_end_separate_report_no_overwrite(self):
        from glaboost.longitudinal import EvaluationConfig
        from glaboost.reporting import write_report
        self.write_scores()
        # Valid alternative number formatting must survive archival unchanged.
        def mutate(rows):
            rows[0]["time_years"] = "0.0000000000"
            return rows
        self.update_csv(mutate)
        result = self.directory / "result"
        def synthetic_writer(*args, **kwargs):
            return write_report(*args, **kwargs, synthetic=True)
        with patch("glaboost.study.load_grape", return_value=self.dataset), \
                patch("glaboost.reporting.write_report", side_effect=synthetic_writer):
            run = create_study_report(self.scores_path, run_name="synthetic-test", grape_root=self.root,
                                      result_dir=result, config=EvaluationConfig(bootstrap_replicates=20))
        self.assertTrue((run / "report.html").is_file())
        self.assertIn("SYNTHETIC", (run / "report.md").read_text())
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "complete")
        self.assertIn("synthetic-test/report.html", (result / "INDEX.md").read_text())
        self.assertTrue((run / "exclusions.csv").is_file())
        self.assertTrue((run / "cohort.json").is_file())
        self.assertEqual((run / "visit_scores.csv").read_bytes(), self.scores_path.read_bytes())
        self.assertEqual((run / "visit_scores.metadata.json").read_bytes(),
                         score_metadata_path(self.scores_path).read_bytes())
        provenance = json.loads((run / "provenance.json").read_text())
        self.assertEqual(sha256_file(run / "visit_scores.metadata.json"), provenance["score_metadata_sha256"])
        with self.assertRaises(FileExistsError):
            create_study_report(self.scores_path, run_name="synthetic-test", grape_root=self.root,
                                result_dir=result, config=EvaluationConfig(bootstrap_replicates=20))

    def test_failed_render_is_marked_failed_and_never_indexed_as_complete(self):
        from glaboost.longitudinal import EvaluationConfig
        self.write_scores()
        result = self.directory / "result"
        with patch("glaboost.study.load_grape", return_value=self.dataset), \
                patch("glaboost.reporting.write_report", side_effect=RuntimeError("synthetic render failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic render failure"):
                create_study_report(self.scores_path, run_name="failed-test", grape_root=self.root,
                                    result_dir=result, config=EvaluationConfig(bootstrap_replicates=20))
        self.assertEqual(json.loads((result / "failed-test/status.json").read_text())["status"], "failed")
        self.assertFalse((result / "INDEX.md").exists())

    def test_report_index_symlink_is_rejected_before_evaluation(self):
        result = self.directory / "result"
        result.mkdir()
        original = self.root / "files" / "VF and clinical information.xlsx"
        before = original.read_bytes()
        (result / "INDEX.md").symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            create_study_report(self.scores_path, run_name="unsafe", grape_root=self.root, result_dir=result)
        self.assertEqual(original.read_bytes(), before)

    def test_cli_weight_cache_cannot_write_into_raw_data(self):
        from glaboost.cli import main
        with patch("glaboost.cli.load_grape", return_value=self.dataset), \
                patch("glaboost.cli.GlaBoost.load") as load_model, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["score-grape", "--root", str(self.root), "--model", str(self.model),
                      "--output", str(self.scores_path), "--cache-dir", str(self.root / "cache"),
                      "--allow-download"])
        self.assertEqual(caught.exception.code, 2)
        load_model.assert_not_called()

    def test_root_readme_updates_only_designated_block_and_default_result(self):
        readme = self.directory / "README.md"
        initial = "Before\n<!-- glaboost-results:start -->\nNo results\n<!-- glaboost-results:end -->\nAfter\n"
        readme.write_text(initial)
        evaluation = {"endpoints": {endpoint: {"status": "not_estimable"}
                                     for endpoint in ("plr2", "plr3", "md_slope")}}
        cohort = {"n_patients": 8, "n_eyes": 16, "n_visits": 48}
        provenance = {"validation_design": "retrospective_fixed_detector_unverified_external"}
        with patch("glaboost.study.PROJECT_ROOT", self.directory):
            _refresh_project_readme(self.directory / "other", "test", evaluation, cohort, provenance)
            self.assertEqual(readme.read_text(), initial)
            _refresh_project_readme(self.directory / "result", "test", evaluation, cohort, provenance)
        actual = readme.read_text()
        self.assertTrue(actual.startswith("Before\n<!-- glaboost-results:start -->"))
        self.assertTrue(actual.endswith("<!-- glaboost-results:end -->\nAfter\n"))
        self.assertIn("result/test/report.html", actual)
        self.assertIn("cannot establish external validation", actual)
        self.assertIn("Latest completed run: `test`", actual)
        self.assertIn("never selected by performance", actual)
        self.assertIn("| PLR2 | 16 | Not recorded | Not estimable | Not estimable | Not estimable |", actual)
        self.assertIn("| MD slope | B: longitudinal | Not estimable | Not estimable | Not estimable | Not estimable | Not estimable |", actual)
        self.assertNotRegex(actual, r"[\u4e00-\u9fff]")
        self.assertNotIn("No results", actual)

    def test_root_readme_metric_tables_have_explicit_units_counts_and_all_endpoints(self):
        readme = self.directory / "README.md"
        readme.write_text("Before\n<!-- glaboost-results:start -->\nOld\n<!-- glaboost-results:end -->\nAfter\n")
        metric_values = {"auroc": .7, "auprc": .2, "sensitivity": .5, "specificity": .75, "f1": .25}
        evaluation = {"endpoints": {endpoint: {
            "status": "ok", "metrics": {
                method: {"balanced_accuracy": {"estimate": point, "ci_low": low, "ci_high": high},
                         **{key: {"estimate": value} for key, value in metric_values.items()}}
                for method, point, low, high in (("latest", .5, .4, .6), ("longitudinal", .625, .5, .75))},
            "delta_balanced_accuracy": {"estimate": .125, "ci_low": -.05, "ci_high": .25}}
            for endpoint in ("plr2", "plr3", "md_slope")}}
        cohort = {"n_patients": 8, "n_eyes": 16, "n_visits": 48,
                  "visits_per_eye": {"median": 3., "q1": 3., "q3": 4.},
                  "followup_months": {"median": 25.6, "q1": 19.9, "q3": 38.4},
                  "progression": {endpoint: {"positive_eyes": 8, "total_eyes": 16, "prevalence": .5}
                                  for endpoint in evaluation["endpoints"]}}
        with patch("glaboost.study.PROJECT_ROOT", self.directory):
            _refresh_project_readme(self.directory / "result", "latest-run", evaluation, cohort,
                                    {"validation_design": "internal_nested_patient_cv"})
        actual = readme.read_text()
        self.assertIn("Median visits per eye: 3.0 (IQR 3.0–4.0); median CFP follow-up: 25.6 months (IQR 19.9–38.4).", actual)
        self.assertIn("percentages", actual)
        self.assertIn("percentage points (pp)", actual)
        for endpoint in ("PLR2", "PLR3", "MD slope"):
            self.assertIn(f"| {endpoint} | 16 | 8 (50.0%) | 50.0 [40.0, 60.0] | 62.5 [50.0, 75.0] | +12.5 [-5.0, +25.0] |", actual)
            for method in ("A: latest", "B: longitudinal"):
                self.assertIn(f"| {endpoint} | {method} | 0.700 | 0.200 | 0.500 | 0.750 | 0.250 |", actual)
        self.assertIn("point estimates on the 0–1 scale", actual)
        self.assertIn("AUPRC uses average precision", actual)
        self.assertIn("result/latest-run/supplementary_metrics.csv", actual)
        self.assertIn("result/latest-run/figures/primary_comparison.png", actual)
        self.assertIn("retrospective internal validation on GRAPE", actual)
        self.assertIn("conditional on fixed out-of-fold predictions", actual)
        self.assertNotRegex(actual, r"[\u4e00-\u9fff]")

    def test_root_readme_preserves_synthetic_guard_and_legacy_external_scope(self):
        readme = self.directory / "README.md"
        initial = "<!-- glaboost-results:start -->\nExisting result\n<!-- glaboost-results:end -->\n"
        readme.write_text(initial)
        evaluation = {"endpoints": {"plr2": {"status": "ok", "metrics": {
            "latest": {"balanced_accuracy": {"estimate": .6}},
            "longitudinal": {"balanced_accuracy": {"estimate": .6, "ci_low": float('nan'), "ci_high": .8}}},
            "delta_balanced_accuracy": {"estimate": 0.}}}}
        provenance = {"validation_design": "external_fixed_detector"}
        with patch("glaboost.study.PROJECT_ROOT", self.directory):
            for test_evaluation, test_provenance in ((dict(evaluation, synthetic=True), provenance),
                                                     (evaluation, dict(provenance, synthetic=True))):
                _refresh_project_readme(self.directory / "result", "synthetic", test_evaluation, {}, test_provenance)
                self.assertEqual(readme.read_text(), initial)
            _refresh_project_readme(self.directory / "result", "external", evaluation, {}, provenance)
        actual = readme.read_text()
        self.assertIn("60.0 [CI not estimable]", actual)
        self.assertIn("+0.0 [CI not estimable]", actual)
        self.assertIn("fixed visit detector is declared independent of GRAPE", actual)
        self.assertIn("complete progression pipeline has not undergone an independent external validation", actual)
        self.assertNotIn("retrospective internal validation on GRAPE", actual)
        self.assertNotRegex(actual, r"\bnan\b")


if __name__ == "__main__":
    unittest.main()
