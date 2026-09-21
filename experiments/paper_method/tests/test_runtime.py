"""Temporary-data and mocked workflow checks; never launch a real experiment."""

import ast
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import data, runner
from paper_method.evaluation import FIELDS, structural_checks


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = data.read_json(runner.PACKAGE / 'config.json')

    def prepared(self, path):
        path.mkdir()
        words = {word: i for i, word in enumerate(('<pad>', '<unk>', '<start>', '<end>', 'a', 'b'))}
        tags = {word: i for i, word in enumerate(('finding', '<pad>', '<unk>', '<start>', '<end>'))}
        data.write_json(path / 'WORDMAP_glaucoma.json', words)
        data.write_json(path / 'TAGMAP_glaucoma.json', tags)
        for number, split in enumerate(('TRAIN', 'VAL', 'TEST')):
            data.write_json(path / f'{split}_CAPTIONS_glaucoma.json', [[2, 4, 3, 0, 0], [2, 5, 4, 3, 0]])
            data.write_json(path / f'{split}_CAPLENS_glaucoma.json', [3, 4])
            data.write_json(path / f'{split}_TAGSYN_glaucoma.json', [[1], [0]])
            data.write_json(path / f'{split}_records.json', [{'id': f'{split}:{i}', 'pixel_sha256': f'{split}-pixel-{i}'} for i in range(2)])
            with h5py.File(path / f'{split}_IMAGES_glaucoma.hdf5', 'w') as handle:
                handle.attrs['captions_per_image'] = 1
                handle.create_dataset('images', data=np.full((2, 3, 224, 224), number, dtype=np.uint8))
        return words, tags

    def close(self, splits):
        for split in splits.values():
            split.close()

    def config_file(self, root):
        self.prepared(root / 'data')
        (root / 'weights.pth').write_bytes(b'preflight fixture, never loaded')
        config = {**self.config, 'amp': False, 'data_directory': str(root / 'data'),
                  'pretrained_weights': str(root / 'weights.pth')}
        path = root / 'config.json'
        data.write_json(path, config)
        return path, config

    def predictions(self):
        report = ' '.join(f'{field} : value .' for field in FIELDS)
        return [{'id': 'TEST:0', 'reference': report, 'prediction': report,
                 'terminated_with_end': True, 'token_ids': [2, 4, 3], 'log_probability': -1.}]

    def test_reader_preserves_padding_and_normalizes_rgb_without_changing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'
            self.prepared(path)
            words, tags, splits, hashes = data.load_dataset(path)
            try:
                row = splits['TRAIN'][0]
                self.assertEqual(row['id'], 'TRAIN:0')
                self.assertEqual(row['caption'].tolist(), [2, 4, 3, 0, 0])
                self.assertEqual(row['length'].item(), 3)
                self.assertEqual(tuple(row['image'].shape), (3, 224, 224))
                self.assertAlmostEqual(row['image'][0, 0, 0].item(), -.485 / .229, places=5)
                self.assertEqual(row['tags'].tolist(), [1.])
                self.assertEqual(hashes, {name: data.fingerprint(path / name) for name in hashes})
            finally:
                self.close(splits)
            self.assertTrue(all(split.handle is None for split in splits.values()))

    def test_reader_rejects_bad_eos_padding_and_label_width(self):
        for kind in ('duplicate_end', 'nonpad_tail', 'label_width'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'data'
                self.prepared(path)
                if kind == 'label_width':
                    data.write_json(path / 'TRAIN_TAGSYN_glaucoma.json', [[1, 0], [0, 1]])
                else:
                    caps = data.read_json(path / 'TRAIN_CAPTIONS_glaucoma.json')
                    caps[0][1 if kind == 'duplicate_end' else 4] = 3 if kind == 'duplicate_end' else 5
                    data.write_json(path / 'TRAIN_CAPTIONS_glaucoma.json', caps)
                with self.assertRaises(ValueError):
                    data.load_dataset(path)

    def test_reader_rejects_duplicate_ids_and_pixels_across_splits(self):
        for key in ('id', 'pixel_sha256'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'data'
                self.prepared(path)
                records = data.read_json(path / 'VAL_records.json')
                records[0][key] = data.read_json(path / 'TRAIN_records.json')[0][key]
                data.write_json(path / 'VAL_records.json', records)
                with self.assertRaisesRegex(ValueError, 'crosses a split'):
                    data.load_dataset(path)

    def test_vocab_and_configuration_validation(self):
        runner.validate_config(self.config)
        for change in ({'epochs': 0}, {'seed': True}, {'amp': 'true'}, {'lambda_parallel': 2},
                       {'weight_decay': float('nan')}, {'attention_heads': 3}, {'old_eos_weight': 5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_config({**self.config, **change})
        with self.assertRaises(ValueError):
            data.validate_vocabulary({'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 2})

    def test_checkpoint_is_strict_about_format_data_vocab_and_archived_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            runner.archive_sources(source)
            checkpoint = {'format': runner.FORMAT, 'config': self.config, 'words': {'a': 0},
                          'tags': {'x': 0}, 'data_fingerprints': {'data': 'hash'}, 'model': {}, 'epoch': 1}
            torch.save(checkpoint, source / 'best.pt')
            loaded = runner.load_checkpoint(source, self.config, {'a': 0}, {'x': 0}, {'data': 'hash'})
            self.assertEqual(loaded['epoch'], 1)
            for key, value in (('format', 'old'), ('words', {}), ('data_fingerprints', {})):
                torch.save({**checkpoint, key: value}, source / 'best.pt')
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(source, self.config, {'a': 0}, {'x': 0}, {'data': 'hash'})
            torch.save(checkpoint, source / 'best.pt')
            with tarfile.open(source / 'source.tar.gz', 'w:gz') as archive:
                for name in ('model.py', 'losses.py', 'decoding.py', 'data.py', 'evaluation.py'):
                    path = runner.PACKAGE / name
                    blob = b'changed model' if name == 'model.py' else path.read_bytes()
                    member = tarfile.TarInfo(str(path.relative_to(runner.ROOT)))
                    member.size = len(blob)
                    archive.addfile(member, io.BytesIO(blob))
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                runner.load_checkpoint(source, self.config, {'a': 0}, {'x': 0}, {'data': 'hash'})

    def test_preflight_never_builds_model_or_runs_metrics_and_closes_readers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.config_file(root)
            output = root / 'output'
            with patch('sys.argv', ['run.py', '--config', str(path), '--run-dir', str(output), '--check']), \
                    patch('sys.stdout', new_callable=io.StringIO) as stdout, \
                    patch.object(runner, 'build_model') as build, patch.object(runner, 'train') as train, \
                    patch.object(runner, 'generate_reports') as generate, patch.object(runner, 'text_metrics') as metrics:
                runner.main()
            for function in (build, train, generate, metrics):
                function.assert_not_called()
            self.assertFalse(output.exists())
            report = json.loads(stdout.getvalue())
            self.assertEqual(report['samples'], {'TRAIN': 2, 'VAL': 2, 'TEST': 2})
            self.assertFalse(report['paper_comparable'])

    def test_training_dispatch_is_mocked_and_archives_only_new_framework(self):
        for train_only in (True, False):
            with self.subTest(train_only=train_only), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, _ = self.config_file(root)
                output = root / 'output'
                args = ['run.py', '--config', str(path), '--run-dir', str(output), '--device', 'cpu', '--accept-reconstruction']
                if train_only:
                    args.append('--train-only')
                before = {p.name: data.fingerprint(p) for p in (root / 'data').iterdir()}
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(runner, 'build_model', return_value=torch.nn.Linear(1, 1)), \
                        patch.object(runner, 'train', return_value=2) as train, \
                        patch.object(runner, 'generate_reports', return_value=self.predictions()) as generate, \
                        patch.object(runner, 'text_metrics', return_value={'Bleu_4': .2}) as metrics, \
                        patch('torch.optim.Adam', side_effect=AssertionError('No optimizer execution in workflow tests')):
                    runner.main()
                train.assert_called_once()
                self.assertEqual(before, {p.name: data.fingerprint(p) for p in (root / 'data').iterdir()})
                status = data.read_json(output / 'status.json')
                self.assertEqual(status['stage'], 'training_complete' if train_only else 'complete')
                with tarfile.open(output / 'source.tar.gz') as archive:
                    self.assertFalse(any('github_fixed' in name or 'github_original' in name for name in archive.getnames()))
                    self.assertIn('experiments/paper_method/model.py', archive.getnames())
                if train_only:
                    generate.assert_not_called()
                    metrics.assert_not_called()
                    self.assertFalse((output / 'predictions.json').exists())
                else:
                    self.assertEqual(data.read_json(output / 'metrics.json')['metrics_x100']['Bleu_4'], 20.)
                    self.assertTrue((output / 'report_checks.json').is_file())

    def test_acknowledgement_and_no_overwrite_guards_stop_before_model_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.config_file(root)
            for output in (root, root / 'new'):
                with patch('sys.argv', ['run.py', '--config', str(path), '--run-dir', str(output), '--device', 'cpu']), \
                        patch('sys.stderr', new_callable=io.StringIO), patch.object(runner, 'build_model') as build:
                    with self.assertRaises(SystemExit):
                        runner.main()
                build.assert_not_called()
            self.assertFalse((root / 'new').exists())

    def test_mocked_epochs_reload_best_validation_checkpoint_not_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model = torch.nn.Linear(1, 1, bias=False)
            config = {**self.config, 'epochs': 3, 'amp': False}
            validation_losses = iter((4., 2., 3.))
            epoch = 0

            def fake_epoch(model, loader, config, device, optimizer=None, scaler=None):
                nonlocal epoch
                if optimizer is not None:
                    epoch += 1
                    with torch.no_grad():
                        model.weight.fill_(epoch)
                    return {'total': 1.}
                return {'total': next(validation_losses)}

            with patch.object(runner, 'run_epoch', side_effect=fake_epoch) as run_epoch, \
                    patch('torch.optim.Adam') as optimizer, patch('torch.cuda.amp.GradScaler') as scaler, \
                    patch('sys.stdout', new_callable=io.StringIO):
                optimizer.return_value.state_dict.return_value = {}
                scaler.return_value.state_dict.return_value = {}
                selected = runner.train(model, {'TRAIN': [0, 1], 'VAL': [2]}, config,
                                        {'a': 0}, {'tag': 0}, {}, output, torch.device('cpu'))
            self.assertEqual(selected, 2)
            self.assertEqual(model.weight.item(), 2.)
            self.assertEqual(run_epoch.call_count, 6)
            optimizer.return_value.step.assert_not_called()
            scaler.return_value.step.assert_not_called()
            last = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(last['epoch'], 3)
            self.assertEqual(last['model']['weight'].item(), 3.)
            self.assertEqual(len(data.read_json(output / 'history.json')), 3)

    def test_evaluation_dispatch_loads_new_checkpoint_without_training_even_if_amp_was_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config = self.config_file(root)
            config['amp'] = True
            source, output = root / 'source', root / 'evaluation'
            source.mkdir()
            data.write_json(source / 'config.json', config)
            words, tags, splits, hashes = data.load_dataset(root / 'data')
            self.close(splits)
            runner.archive_sources(source)
            model = torch.nn.Linear(1, 1, bias=False)
            checkpoint = {'format': runner.FORMAT, 'config': config, 'words': words, 'tags': tags,
                          'data_fingerprints': hashes, 'epoch': 2, 'model': {'weight': torch.full((1, 1), 3.)}}
            torch.save(checkpoint, source / 'best.pt')
            args = ['run.py', '--evaluate-run', str(source), '--run-dir', str(output),
                    '--device', 'cpu', '--accept-reconstruction']
            with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                    patch.object(runner, 'build_model', return_value=model), \
                    patch.object(runner, 'train') as train, \
                    patch.object(runner, 'generate_reports', return_value=self.predictions()) as generate, \
                    patch.object(runner, 'text_metrics', return_value={'Bleu_4': .2}), \
                    patch('torch.optim.Adam', side_effect=AssertionError('Evaluation must not train')):
                runner.main()
            train.assert_not_called()
            generate.assert_called_once()
            self.assertEqual(model.weight.item(), 3.)
            self.assertEqual(data.read_json(output / 'status.json'), {'stage': 'complete', 'selected_epoch': 2})
            self.assertEqual(data.read_json(output / 'manifest.json')['mode'], 'evaluation_only')
            self.assertFalse((output / 'best.pt').exists())

    def test_structural_checks_detect_loops_conflicts_and_early_incomplete_reports(self):
        first = self.predictions()[0]
        rows = [first,
                {**first, 'id': 'loop', 'prediction': first['prediction'] + ' confidence level : other .', 'terminated_with_end': False},
                {**first, 'id': 'early', 'prediction': 'optic disc size : value .', 'terminated_with_end': True},
                {**first, 'id': 'partial', 'prediction': first['prediction'][:-1]}]
        result = structural_checks(rows)
        summary = result['summary']
        self.assertEqual(summary['all_fields_once_and_complete'], 1)
        self.assertEqual(summary['repeated_confidence'], 1)
        self.assertEqual(summary['conflicting_fields'], 1)
        self.assertEqual(summary['unfinished_predictions'], 1)
        self.assertEqual(summary['ended_with_incomplete_structure'], 2)
        self.assertEqual(summary['reference_field_exact_matches']['glaucoma risk assessment'], 3)

    def test_no_live_retired_code_or_import_dependency_remains(self):
        for name in ('github_fixed', 'github_original', 'common'):
            self.assertFalse((runner.ROOT / 'experiments' / name).exists())
        for path in runner.PACKAGE.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else \
                        [item.name for item in node.names] if isinstance(node, ast.Import) else []
                self.assertFalse(any(name.startswith(('github_fixed', 'github_original', 'common')) for name in names))


if __name__ == '__main__':
    unittest.main()
