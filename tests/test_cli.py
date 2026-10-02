"""External workflow dispatch and shell checks; no resource requests or real fits."""

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from glaboost.cli import _execution_settings, main
from glaboost.config import GlaBoostConfig
from glaboost.data import VisitInput


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_hf_training_then_grape_dispatch_has_prescribed_sources_and_runtime_options(self):
        # A module stub prevents any source training even while its implementation
        # is developed independently of this CLI dispatch test.
        module = types.ModuleType("glaboost.hf_training")
        module.run_hf_grape = Mock(return_value=Path("unused"))
        for options in ([], ["--training-plan", "source-plan.json", "--hf-root", "/tmp/source",
                             "--device", "cuda:1", "--image-batch-size", "7", "--allow-download"]):
            with self.subTest(options=options), \
                    patch.dict(os.sys.modules, {"glaboost.hf_training": module}), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run-hf-grape", "--run-name", "source_then_grape", *options]), 0)
            kwargs = module.run_hf_grape.call_args.kwargs
            self.assertEqual(kwargs["run_name"], "source_then_grape")
            self.assertEqual(kwargs["training_plan_path"], "source-plan.json" if options else "configs/hf_training.json")
            self.assertEqual(kwargs["hf_root"], "/tmp/source" if options else
                             "/scratch/users/zeyuhan/DA-SPL/archive/glaucoma_diagnosis_json_analysis")
            self.assertEqual(kwargs["grape_root"], "data/raw/grape")
            self.assertEqual(kwargs["device"], "cuda:1" if options else "cuda")
            self.assertEqual(kwargs["image_batch_size"], 7 if options else None)
            self.assertEqual(kwargs["allow_download"], bool(options))
            self.assertNotIn("synthetic", kwargs)

    def test_external_defaults_and_runtime_overrides_reach_runner(self):
        for options in ([], ["--plan", "declared.json", "--device", "cuda:1", "--image-batch-size", "7",
                             "--image-weights", "weights.pth", "--allow-download"]):
            with self.subTest(options=options), \
                    patch("glaboost.external.run_external_validation", return_value=Path("unused")) as runner, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate-grape", "--run-name", "external", *options]), 0)
                kwargs = runner.call_args.kwargs
                self.assertEqual(kwargs["run_name"], "external")
                self.assertEqual(kwargs["plan_path"], "declared.json" if options else "configs/external_models.json")
                self.assertEqual(kwargs["device"], "cuda:1" if options else "cuda")
                self.assertEqual(kwargs["image_batch_size"], 7 if options else None)
                self.assertEqual(kwargs["image_weights"], "weights.pth" if options else None)
                self.assertEqual(kwargs["allow_download"], bool(options))
                self.assertNotIn("synthetic", kwargs)

    def test_missing_external_models_fail_before_devices_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "empty.json"
            plan.write_text(json.dumps({"format_version": 1, "primary_model": "pending", "models": [],
                                        "min_visits": 3, "evaluation": {}}))
            with patch("glaboost.external.resolve_image_devices") as devices, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["validate-grape", "--plan", str(plan), "--run-name", "blocked",
                      "--result-dir", str(root / "result"), "--artifact-dir", str(root / "artifacts")])
            self.assertEqual(raised.exception.code, 2)
            devices.assert_not_called()
            self.assertFalse((root / "result").exists())
            self.assertFalse((root / "artifacts").exists())

    def test_removed_training_command_and_tree_overrides_are_rejected(self):
        invalid = (["train-grape", "--run-name", "old"],
                   *(["validate-grape", "--run-name", "bad", *options] for options in (
                       ["--compare-trees"], ["--n-estimators", "100"], ["--max-depth", "3"],
                       ["--tree-counts", "100,500"], ["--tree-depths", "3,6"], ["--synthetic"],
                       ["--device", "auto"])))
        for args in invalid:
            with self.subTest(args=args), patch("glaboost.external.run_external_validation") as runner, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            runner.assert_not_called()

    def test_manual_evaluation_uses_requested_gpu_or_explicit_cpu(self):
        for device, resolved, ids in (("cuda", "cuda:0", (0, 1)), ("cuda:1", "cuda:1", (1,)), ("cpu", "cpu", ())):
            with self.subTest(device=device), \
                    patch("glaboost.encoders.resolve_image_devices", return_value=(resolved, ids)), \
                    patch("glaboost.study.create_study_report", return_value=Path("unused")) as evaluate, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(["evaluate-grape", "--scores", "scores.csv", "--run-name", "analysis",
                                       "--device", device]), 0)
                config = evaluate.call_args.kwargs["config"]
                self.assertEqual(config.compute_device, resolved)
                self.assertEqual((config.n_splits, config.seed, config.bootstrap_replicates), (3, 42, 2000))

    def test_manual_scoring_forwards_batch_and_training_independence_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = Mock(config=GlaBoostConfig())
            detector.predict_score.return_value = [.2, .3, .4]
            visits = [VisitInput(f"visit{i}", image=f"{i}.jpg", patient_id="patient", eye_id="eye", time_years=float(i))
                      for i in range(3)]
            dataset = Mock()
            dataset.image_visits.return_value = visits
            output = root / "scores.csv"
            with patch("glaboost.encoders.resolve_image_devices", return_value=("cuda:0", (0, 1))), \
                    patch("glaboost.cli.GlaBoost.load", return_value=detector) as load, \
                    patch("glaboost.cli.load_grape", return_value=dataset), \
                    patch("glaboost.cli.write_visit_scores", return_value=(output, output.with_suffix(".metadata.json"))) as write, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(["score-grape", "--model", "detector", "--output", str(output),
                                       "--root", str(root / "raw"), "--cache-dir", str(root / "cache"),
                                       "--training-data-description", "Independent diagnostic training cohort",
                                       "--training-data-reference", "training-manifest.json",
                                       "--grape-training-overlap", "none", "--independence-evidence", "Documented cohort provenance"]), 0)
            self.assertEqual(load.call_args.kwargs["image_batch_size"], 128)
            self.assertEqual(load.call_args.kwargs["device"], "cuda")
            self.assertFalse(load.call_args.kwargs["allow_download"])
            self.assertEqual(write.call_args.kwargs["independence_evidence"], "Documented cohort provenance")
            self.assertEqual(write.call_args.kwargs["grape_overlap"], "none")
            detector.fit.assert_not_called()

    def test_batch_size_scales_with_visible_gpus_and_invalid_sizes_fail_first(self):
        for ids in ((0,), (0, 1), (2,)):
            with patch("glaboost.encoders.resolve_image_devices", return_value=(f"cuda:{ids[0]}", ids)):
                self.assertEqual(_execution_settings("cuda")[2], 64 * len(ids))
                self.assertEqual(_execution_settings("cuda", 7)[2], 7)
        for invalid in (0, -1, True, 1.5):
            with patch("glaboost.encoders.resolve_image_devices") as resolve, self.assertRaises(ValueError):
                _execution_settings("cuda", invalid)
            resolve.assert_not_called()


class ShellTests(unittest.TestCase):
    """Exercise a script copy and fake executables entirely under /tmp."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="glaboost-external-shell-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.project = self.directory / "project"
        self.project.mkdir()
        self.script = self.project / "run_grape.sh"
        self.script.write_bytes((ROOT / "run_grape.sh").read_bytes())
        for entry in ("artifacts", ".cache"):
            target = self.directory / ("storage" + entry)
            target.mkdir()
            (self.project / entry).symlink_to(target, target_is_directory=True)
        root = self.project / "data/raw/grape"
        (root / "files").mkdir(parents=True)
        (root / "files/VF and clinical information.xlsx").write_bytes(b"synthetic")
        (root / "extracted/CFPs").mkdir(parents=True)
        (self.project / "configs").mkdir()
        (self.project / "configs/external_models.json").write_text("{}")
        (self.project / "configs/hf_training.json").write_text("{}")
        self.hf_root = self.directory / "hf_archive"
        self.hf_root.mkdir()
        binary = self.directory / "bin"
        binary.mkdir()
        self.log = self.directory / "calls.jsonl"
        fake = (f"#!{os.sys.executable}\nimport json, os, sys\n"
                "name = os.path.basename(sys.argv[0])\n"
                "with open(os.environ['EXTERNAL_CLI_TEST_LOG'], 'a') as f:\n"
                "    f.write(json.dumps([name, *sys.argv[1:]]) + '\\n')\n"
                "if name == 'srun': sys.exit(97)\n"
                "if '-' in sys.argv and os.environ.get('EXTERNAL_CLI_TEST_REAL_PREFLIGHT'):\n"
                "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
                "if '-' in sys.argv: sys.stdin.read()\n"
                "key = 'EXTERNAL_CLI_TEST_PREFLIGHT_EXIT' if '-' in sys.argv else 'EXTERNAL_CLI_TEST_WORKFLOW_EXIT'\n"
                "sys.exit(int(os.environ.get(key, '0')))\n")
        for name in ("python", "conda", "srun"):
            executable = binary / name
            executable.write_text(fake)
            executable.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("SLURM_", "CONDA_"))}
        self.env.update(PATH=f"{binary}:/usr/bin:/bin", CONDA_DEFAULT_ENV="da-spl-repro",
                        CONDA_PREFIX=str(self.directory), EXTERNAL_CLI_TEST_LOG=str(self.log))

    def run_script(self, *args, allocated=False):
        if self.log.exists():
            self.log.unlink()
        environment = dict(self.env)
        if allocated:
            environment["SLURM_JOB_ID"] = "existing-synthetic-job"
        if "--plan" not in args and "--hf-root" not in args:
            args = ("--hf-root", str(self.hf_root), *args)
        result = subprocess.run(["/bin/bash", str(self.script), *args], env=environment,
                                capture_output=True, text=True, timeout=10)
        calls = [json.loads(row) for row in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_one_workflow_uses_existing_environment_without_any_scheduler_call(self):
        for allocated in (False, True):
            with self.subTest(allocated=allocated):
                result, calls = self.run_script(allocated=allocated)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([call[0] for call in calls], ["python", "python"])
                self.assertEqual(calls[0][1], "-")
                self.assertEqual(calls[1][1:4], ["-m", "glaboost", "run-hf-grape"])
                self.assertEqual(calls[1][calls[1].index("--training-plan") + 1], "configs/hf_training.json")
                self.assertEqual(calls[1][calls[1].index("--hf-root") + 1], str(self.hf_root))
                self.assertIn("grape_external_", calls[1][calls[1].index("--run-name") + 1])
                self.assertNotIn("--allow-download", calls[1])
                self.assertNotIn("srun", self.script.read_text())

    def test_explicit_runtime_options_are_forwarded_without_tree_training_settings(self):
        plan = self.project / "configs/declared models.json"
        plan.write_text("{}")
        weights = self.project / ".cache/weights.pth"
        weights.write_bytes(b"mock checkpoint, never loaded")
        result, calls = self.run_script("--plan", str(plan), "--run-name", "external", "--device", "cuda:1",
                                        "--image-batch-size", "7", "--image-weights", str(weights), "--allow-download")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = calls[-1]
        self.assertEqual(command[1:4], ["-m", "glaboost", "validate-grape"])
        self.assertEqual(command[command.index("--plan") + 1], str(plan))
        self.assertEqual(command[command.index("--image-batch-size") + 1], "7")
        self.assertEqual(command[command.index("--image-weights") + 1], str(weights))
        self.assertIn("--allow-download", command)
        self.assertNotIn("--n-estimators", command)
        self.assertNotIn("--training-plan", command)

    def test_custom_hf_training_plan_is_forwarded_and_conflicts_are_rejected(self):
        plan = self.project / "configs/source settings.json"
        plan.write_text("{}")
        result, calls = self.run_script("--training-plan", str(plan), "--run-name", "custom_source")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[-1][calls[-1].index("--training-plan") + 1], str(plan))
        for args in (("--plan", "configs/external_models.json", "--training-plan", str(plan)),
                     ("--plan", "configs/external_models.json", "--hf-root", str(self.hf_root))):
            with self.subTest(args=args):
                result, calls = self.run_script(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, [])

    def test_all_source_and_grape_output_collisions_block_before_python(self):
        for index, (base, suffix) in enumerate((("result", ""), ("artifacts", ""),
                                               ("result", "_source"), ("artifacts", "_source"))):
            with self.subTest(base=base, suffix=suffix):
                name = f"collision{index}"
                collision = self.project / base / (name + suffix)
                collision.mkdir(parents=True)
                result, calls = self.run_script("--run-name", name)
                self.assertEqual(result.returncode, 2)
                self.assertIn(str(collision), result.stderr)
                self.assertEqual(calls, [])

    def test_hf_archive_cannot_be_used_as_an_output_cache(self):
        # Only execute the read-only storage preflight with real project code.
        # The workflow command remains a stub and must never be reached here.
        cache = self.project / ".cache"
        cache.unlink()
        protected = self.hf_root / "unsafe_output"
        protected.mkdir()
        cache.symlink_to(protected, target_is_directory=True)
        self.env["EXTERNAL_CLI_TEST_REAL_PREFLIGHT"] = "1"
        self.env["PYTHONPATH"] = str(ROOT / "src")
        result, calls = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside raw data", result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(protected.iterdir()), [])

    def test_conda_run_is_used_if_environment_is_not_already_active(self):
        self.env.pop("CONDA_DEFAULT_ENV")
        self.env.pop("CONDA_PREFIX")
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[:6] == ["conda", "run", "--no-capture-output", "-n", "da-spl-repro", "python"]
                            for call in calls))

    def test_obsolete_flags_invalid_batch_and_overwrite_are_rejected_before_python(self):
        for args in (("--compare-trees",), ("--method", "ch"), ("--n-estimators", "500"),
                     ("--image-batch-size", "0"), ("--run-name", "../escape")):
            with self.subTest(args=args):
                result, calls = self.run_script(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, [])
        (self.project / "result/existing").mkdir(parents=True)
        result, calls = self.run_script("--run-name", "existing")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_missing_scratch_link_fails_before_python(self):
        (self.project / "artifacts").unlink()
        (self.project / "artifacts").mkdir()
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("scratch", result.stderr)
        self.assertEqual(calls, [])

    def test_preflight_or_workflow_failure_does_not_report_success(self):
        for key, count in (("EXTERNAL_CLI_TEST_PREFLIGHT_EXIT", 1), ("EXTERNAL_CLI_TEST_WORKFLOW_EXIT", 2)):
            with self.subTest(stage=key):
                self.env[key] = "19"
                result, calls = self.run_script()
                self.assertEqual(result.returncode, 19)
                self.assertEqual(len(calls), count)
                self.assertNotIn("完成。报告", result.stdout)
                self.assertIn("运行失败", result.stderr)
                self.env.pop(key)


if __name__ == "__main__":
    unittest.main()
