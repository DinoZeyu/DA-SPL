"""Synthetic report checks; no GRAPE training or empirical study results."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glaboost.reporting import (ENDPOINTS, METRICS, _clean, _compact_provenance,
                               _report_blocks, _selected_eyes, _summary, _trajectory_figure, write_report)


def synthetic_inputs():
    evaluation = {"config": {"n_splits": 3, "seed": 42, "bootstrap_replicates": 40,
                              "persistence_threshold": .5, "logistic_c": 1, "decision_threshold": .5},
                  "endpoints": {}, "predictions": [], "features": []}
    for i, ep in enumerate(ENDPOINTS):
        metrics = {}
        for method, offset in (("latest", 0), ("longitudinal", .06)):
            metrics[method] = {metric: {"estimate": .6 + offset + i * .02,
                                       "ci_low": .4 + offset, "ci_high": .8 + offset} for metric in METRICS}
        evaluation["endpoints"][ep] = {
            "status": "ok", "n_eyes": 12, "n_patients": 10, "n_positive_eyes": 4,
            "n_positive_patients": 4, "n_splits": 3, "metrics": metrics,
            "delta_balanced_accuracy": {"estimate": .06, "ci_low": -.05, "ci_high": .17},
            "bootstrap": {"requested": 40, "valid": 36, "skipped_single_class": 4}, "folds": []}
        evaluation["predictions"].append({"endpoint": ep, "eye_id": "p1_OD", "patient_id": "p1", "fold": 0,
                                          "y_true": 1, "latest_probability": .6, "longitudinal_probability": .7})
    cohort = {"n_eyes": 12, "n_patients": 10, "n_visits": 36,
              "visits_per_eye": {"median": 3, "q1": 3, "q3": 3},
              "followup_months": {"median": 24, "q1": 18, "q3": 30},
              "source": {"n_patients": 14, "n_eyes": 20, "n_visits": 64, "n_visits_with_cfp": 44},
              "progression": {ep: {"positive_eyes": 4, "total_eyes": 12, "prevalence": 1 / 3} for ep in ENDPOINTS},
              "exclusions": [{"eye_id": "p20_OD", "patient_id": "p20", "available_cfp_visits": 1, "reason": "fewer than 3 CFP visits"}],
              "observation_window": {"eyes_last_cfp_before_last_recorded_visit": 2,
                                     "cfp_to_last_recorded_visit_months": {"median": 0, "q1": 0, "q3": 1}},
              "modalities": {"image": True}}
    provenance = {"run_name": "synthetic_check", "created_at": "2026-01-01T00:00:00Z",
                  "minimum_cfp_visits_per_eye": 3,
                  "model_origin": "paper_reimplementation", "model_info": {"source": "SYNTHETIC"},
                  "validation_design": "retrospective_fixed_detector_unverified_external"}
    records = [{"eye_id": "p{}_OD".format(i), "patient_id": "p{}".format(i),
                "times": [0, .5, 2], "scores": [.2 + i * .05, .3 + i * .05, .4 + i * .05]} for i in range(1, 7)]
    return evaluation, cohort, provenance, records


def synthetic_internal_inputs():
    evaluation, cohort, provenance, records = synthetic_inputs()
    evaluation["config"]["inner_splits"] = 2
    provenance.update(validation_design="internal_nested_patient_cv",
                      model_origin="GlaBoost architecture adapted to GRAPE progression",
                      score_definition="Endpoint-specific visit evidence learned from eye-level retrospective "
                                       "progression labels; not glaucoma diagnosis or prospectively calibrated risk")
    # Deliberately shuffled, with different folds and labels. Selection must use
    # identifiers only and show each endpoint rather than six instances of PLR2.
    records = [dict(record, endpoint=ep, fold=i % 3, labels={ep: i % 2})
               for ep in reversed(ENDPOINTS) for i, record in enumerate(reversed(records))]
    return evaluation, cohort, provenance, records


class SummaryTests(unittest.TestCase):
    def test_gpu_methods_disclose_solver_and_random_generator_changes(self):
        evaluation, cohort, provenance, records = synthetic_internal_inputs()
        evaluation["config"].update(logistic_solver="torch_newton", gpu_device_ids=[0, 1],
                                    base_model_config={"tree_method": "gpu_hist"},
                                    endpoint_compute_devices={"plr2": "cuda:0", "plr3": "cuda:1", "md_slope": "cuda:0"})
        text = str(_report_blocks(evaluation, cohort, provenance, records, synthetic=True))
        for phrase in ("GPU histogram", "GPU prediction", "GPU damped-Newton", "intercept in the L2 penalty",
                       "draws differ", "NumPy PCG64", "one endpoint at a time per device"):
            self.assertIn(phrase, text)
        self.assertNotIn("CPU histogram", text)
        self.assertNotIn("the liblinear solver,", text)

    def test_internal_summary_describes_whole_pipeline_and_weak_labels(self):
        evaluation, cohort, provenance, _ = synthetic_internal_inputs()
        text = _summary(evaluation, cohort, provenance, False)
        self.assertIn("retrospective internal validation", text)
        self.assertIn("nested patient-grouped cross-validation", text)
        self.assertIn("separate visit model", text)
        self.assertIn("not labels of disease state at an individual visit", text)
        self.assertNotIn("fixed detector", text)
        self.assertNotIn("external", text.lower())
        self.assertIn("conditional on the fitted out-of-fold", text)

    def test_inconclusive_positive_point_estimates_are_not_called_improvement(self):
        evaluation, cohort, provenance, _ = synthetic_inputs()
        summary = _summary(evaluation, cohort, provenance, False)
        self.assertIn("changed from 0.600", summary)
        self.assertIn("+0.060", summary)
        self.assertIn("do not establish a consistent gain", summary)
        self.assertNotIn("improved", summary)
        self.assertIn("External validation is not established", summary)

    def test_negative_delta_does_not_claim_consistent_positive_direction(self):
        evaluation, cohort, provenance, _ = synthetic_inputs()
        evaluation["endpoints"]["plr3"]["delta_balanced_accuracy"]["estimate"] = -.1
        self.assertNotIn("Point estimates favored", _summary(evaluation, cohort, provenance, False))

    def test_positive_interval_statement_requires_all_three_endpoints(self):
        evaluation, cohort, provenance, _ = synthetic_inputs()
        for result in evaluation["endpoints"].values():
            result["delta_balanced_accuracy"]["ci_low"] = .01
        self.assertIn("conditional paired intervals were above zero", _summary(evaluation, cohort, provenance, False))
        evaluation["endpoints"]["plr3"] = {"status": "not_estimable", "reason": "too few positive patients"}
        text = _summary(evaluation, cohort, provenance, False)
        self.assertNotIn("conditional paired intervals were above zero", text)
        self.assertIn("PLR3 was not estimable: too few positive patients", text)

    def test_external_detector_is_not_external_complete_pipeline(self):
        evaluation, cohort, provenance, _ = synthetic_inputs()
        provenance["validation_design"] = "external_fixed_detector"
        text = _summary(evaluation, cohort, provenance, False)
        self.assertIn("internally cross-validated progression mappings", text)
        self.assertIn("supplied detector provenance declaration", text)

    def test_nonfinite_results_become_json_null(self):
        self.assertEqual(_clean({"x": float("nan"), "y": [float("inf")]}), {"x": None, "y": [None]})

    def test_per_endpoint_interpretation_tracks_interval_direction(self):
        evaluation, cohort, provenance, _ = synthetic_inputs()
        evaluation["endpoints"]["plr2"]["delta_balanced_accuracy"] = {"estimate": -.2, "ci_low": -.3, "ci_high": -.1}
        text = _summary(evaluation, cohort, provenance, False)
        self.assertIn("favored lower balanced accuracy", text)
        self.assertIn("direction of the difference remains uncertain", text)

    def test_large_provenance_arrays_and_mappings_remain_in_json_only(self):
        detail = {"features": list(range(100)), "images": {str(i): "checksum" for i in range(100)}}
        compact = _compact_provenance(detail)
        self.assertEqual(compact["features"]["item_count"], 100)
        self.assertEqual(compact["images"]["entry_count"], 100)
        self.assertEqual(len(detail["images"]), 100)


class ReportTests(unittest.TestCase):
    def test_internal_report_has_correct_training_target_and_no_missing_weight_claim(self):
        inputs = synthetic_internal_inputs()
        with tempfile.TemporaryDirectory(prefix="glaboost_synthetic_internal_report_") as temp:
            root = Path(temp)
            path = write_report(root, *inputs, synthetic=True)
            report = (root / "report.md").read_text()
            self.assertEqual(path.read_text().count("data:image/png;base64,"), 2)
            for text in ("Retrospective internal validation", "weak supervision", "2,048", "100 trees",
                         "inner out-of-fold summaries", "trained on all eligible training visits",
                         "one positive patient", "each training eye equal total weight", "inner_splits",
                         "not an external validation", "Outer fold", "nor prospectively calibrated risks"):
                self.assertIn(text, report)
            self.assertNotIn("PROVISIONAL", report)
            self.assertNotIn("The fixed diagnosis detector", report)
            self.assertNotIn("Enabled in fixed detector", report)
            self.assertNotIn("independence from GRAPE requires", report)
            saved = json.loads((root / "provenance.json").read_text())
            self.assertEqual(saved["validation_design"], "internal_nested_patient_cv")
            self.assertTrue(saved["synthetic"])
            figure = (root / "figures" / "score_trajectories.svg").read_text()
            self.assertIn("Visit-level progression evidence", figure)
            self.assertNotIn("Visit-level diagnosis score", figure)

    def test_internal_trajectories_select_two_eyes_per_endpoint_without_outcomes(self):
        records = synthetic_internal_inputs()[3]
        chosen = _selected_eyes(records, internal=True)
        self.assertEqual([(row["endpoint"], row["eye_id"]) for row in chosen],
                         [(ep, eye) for ep in ENDPOINTS for eye in ("p1_OD", "p2_OD")])
        changed = [dict(row, labels={ep: 1 for ep in ENDPOINTS}, scores=list(reversed(row["scores"])))
                   for row in reversed(records)]
        self.assertEqual([(row["endpoint"], row["eye_id"]) for row in _selected_eyes(changed, internal=True)],
                         [(row["endpoint"], row["eye_id"]) for row in chosen])

    def test_internal_figure_columns_identify_endpoint_and_use_only_held_out_record_scores(self):
        import matplotlib.pyplot as plt
        records = synthetic_internal_inputs()[3]
        fig = _trajectory_figure(records, synthetic=True, internal=True)
        try:
            for column, ep in enumerate(ENDPOINTS):
                selected = [row for row in _selected_eyes(records, internal=True) if row["endpoint"] == ep]
                for row, record in enumerate(selected):
                    ax = fig.axes[row * 3 + column]
                    self.assertIn("Outer fold {}".format(record["fold"]), ax.get_title(loc="left"))
                    self.assertEqual(list(ax.lines[0].get_xdata()), record["times"])
                    self.assertEqual(list(ax.lines[0].get_ydata()), record["scores"])
                    self.assertEqual(ax.get_ylim(), (0, 1))
        finally:
            plt.close(fig)

    def test_complete_portable_synthetic_report_and_no_overwrite(self):
        inputs = synthetic_inputs()
        with tempfile.TemporaryDirectory(prefix="glaboost_synthetic_report_") as temp:
            root = Path(temp)
            path = write_report(root, *inputs, synthetic=True)
            page = path.read_text()
            report = (root / "report.md").read_text()
            self.assertEqual(page.count("data:image/png;base64,"), 2)
            self.assertNotIn('src="figures/', page)
            self.assertIn("SYNTHETIC TEST DATA", page)
            self.assertIn("SYNTHETIC TEST DATA", report)
            self.assertNotIn("[XX]", report)
            self.assertIn("12 eyes from 10 patients", report)
            self.assertIn("original ascertainment period", report)
            self.assertIn("not clinically calibrated", report)
            self.assertIn("at least 3 visits per eye", report)
            self.assertIn("Original CFP / Corresponding CFP", report)
            self.assertIn("exclusions.csv", report)
            self.assertNotIn("p20_OD", report)
            self.assertIn("36", report)
            for stem in ("primary_comparison", "score_trajectories"):
                for extension in ("png", "pdf", "svg"):
                    self.assertGreater((root / "figures" / (stem + "." + extension)).stat().st_size, 100)
                self.assertIn("SYNTHETIC TEST DATA", (root / "figures" / (stem + ".svg")).read_text())
                from pypdf import PdfReader
                for page in PdfReader(root / "figures" / (stem + ".pdf")).pages:
                    for font in page["/Resources"]["/Font"].values():
                        font = font.get_object()
                        self.assertNotEqual(font["/Subtype"], "/Type3")
                        for descendant in font.get("/DescendantFonts", []):
                            self.assertIn("/FontFile2", descendant.get_object()["/FontDescriptor"])
            with (root / "primary_results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(float(rows[0]["delta_estimate"]), .06)
            self.assertEqual(rows[0]["synthetic"], "True")
            saved = json.loads((root / "evaluation.json").read_text())
            self.assertTrue(saved["synthetic"])
            self.assertEqual(saved["cohort"]["n_patients"], 10)
            with self.assertRaises(FileExistsError):
                write_report(root, *inputs, synthetic=True)

    def test_all_not_estimable_still_generates_factual_summary(self):
        inputs = list(synthetic_inputs())
        for ep in ENDPOINTS:
            inputs[0]["endpoints"][ep] = {"status": "not_estimable", "reason": "single class", "n_eyes": 12, "n_patients": 10}
        # Keep this second content check inexpensive; full rendering is checked above.
        def fake_figures(root, *args, **kwargs):
            (root / "figures").mkdir()
            for stem in ("primary_comparison", "score_trajectories"):
                (root / "figures" / (stem + ".png")).write_bytes(b"synthetic-test")
        with tempfile.TemporaryDirectory() as temp, patch("glaboost.reporting._save_figures", fake_figures):
            write_report(Path(temp), *inputs, synthetic=True)
            report = (Path(temp) / "report.md").read_text()
            self.assertIn("PLR2 was not estimable: single class", report)
            self.assertNotIn("Point estimates favored", report)


if __name__ == "__main__":
    unittest.main()
