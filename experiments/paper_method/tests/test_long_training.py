"""Long-run controls with synthetic tensors and mocked training; no actual experiments."""

import copy
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import train_lem_off as training
from paper_method.data import fingerprint, read_json, write_json
from paper_method.evaluation import FIELDS
from paper_method.runner import archive_sources
import test_pipeline_audit


class LongTrainingTests(unittest.TestCase):
    def setUp(self):
        helper = test_pipeline_audit.PipelineAuditTests()
        helper.setUp()
        self.helper = helper
        self.words, self.splits = helper.words, helper.splits
        self.records, self.config = helper.records, {**helper.config, 'epochs': 1, 'max_decode_steps': 161}

    def source_fixture(self, root, model):
        source = root / 'paired' / 'lem_off'
        source.mkdir(parents=True)
        weights = root / 'weights.pt'
        weights.write_bytes(b'synthetic pretrained fixture; builder mocked')
        config = {**self.config, 'data_directory': str(root / 'data'), 'pretrained_weights': str(weights)}
        baseline = {**config, 'tag_loss_weight': 5.}
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        final = {key: value.clone() + 1 for key, value in initial.items()}
        digest = training.state_digest(initial)
        write_json(source.parent / 'config.json', baseline)
        write_json(source / 'config.json', config)
        write_json(source.parent / 'manifest.json', {'experiment': training.SOURCE_FORMAT})
        write_json(source / 'manifest.json', {'format': training.SOURCE_FORMAT, 'arm': 'lem_off',
                   'tag_loss_weight': 0., 'initial_state_sha256': digest})
        write_json(source.parent / 'status.json', {'stage': 'complete', 'epochs_per_arm': 1})
        write_json(source / 'status.json', {'stage': 'complete', 'completed_epochs': 1})
        for folder in (source, source.parent):
            write_json(folder / 'data_fingerprints.json', {})
        torch.save({'format': training.SOURCE_FORMAT, 'model': initial, 'words': self.words, 'tags': {},
                    'config': baseline, 'data_fingerprints': {}}, source.parent / 'initial.pt')
        write_json(source.parent / 'initialization.json', {'state_sha256': digest,
                   'file_sha256': fingerprint(source.parent / 'initial.pt')})
        torch.save({'format': training.SOURCE_FORMAT, 'arm': 'lem_off', 'model': final, 'words': self.words, 'tags': {},
                    'config': config, 'data_fingerprints': {}, 'epoch': 1, 'initial_state_sha256': digest}, source / 'last.pt')
        order = training.training_orders(len(self.splits['TRAIN']), 1, config['seed'])[0]
        ids = [self.splits['TRAIN'].records[i]['id'] for i in order]
        write_json(source.parent / 'training_order.json', [{'epoch': 1, 'ids': ids}])
        (source / 'epoch_001').mkdir()
        write_json(source / 'epoch_001' / 'train_ids.json', ids)
        write_json(source / 'history.json', [{'epoch': 1}])
        archive_sources(source.parent)
        return source, initial

    def test_setup_changes_only_epochs_and_reuses_initial_not_final_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, initial = self.source_fixture(Path(tmp), torch.nn.Linear(2, 2))
            config, state, orders, plan, reference = training.load_training_setup(
                source, 30, self.words, {}, self.splits, {})
            baseline = read_json(source / 'config.json')
            self.assertEqual([k for k in config if config[k] != baseline[k]], ['epochs'])
            self.assertEqual(config['tag_loss_weight'], 0)
            self.assertEqual(len(orders), 30)
            self.assertEqual(plan[:1], read_json(source.parent / 'training_order.json'))
            self.assertEqual(training.state_digest(state), training.state_digest(initial))
            self.assertNotEqual(training.state_digest(state), reference['model_state_sha256'])
            for total in (0, 1, True):
                with self.assertRaisesRegex(ValueError, 'Total epochs'):
                    training.load_training_setup(source, total, self.words, {}, self.splits, {})

    def test_setup_rejects_changed_initialization_and_training_ids(self):
        for what in ('initial', 'plan', 'consumed'):
            with self.subTest(what=what), tempfile.TemporaryDirectory() as tmp:
                source, _ = self.source_fixture(Path(tmp), torch.nn.Linear(2, 2))
                if what == 'initial':
                    (source.parent / 'initial.pt').write_bytes(b'corrupted checkpoint')
                    message = 'fingerprint mismatch'
                elif what == 'plan':
                    write_json(source.parent / 'training_order.json', [])
                    message = 'order prefix'
                else:
                    write_json(source / 'epoch_001' / 'train_ids.json', [])
                    message = 'recorded training order'
                with self.assertRaisesRegex(ValueError, message):
                    training.load_training_setup(source, 30, self.words, {}, self.splits, {})

    def test_preflight_acknowledgement_and_output_guards_never_build_or_train(self):
        for mode in ('check', 'missing_ack', 'nested'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, _ = self.source_fixture(root, torch.nn.Linear(2, 2))
                output = source / 'nested' if mode == 'nested' else root / 'result'
                args = ['train_lem_off.py', '--source-run', str(source), '--output', str(output)]
                if mode == 'check':
                    args.append('--check')
                for split in self.splits.values():
                    split.calls, split.closed = 0, False
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch('sys.stderr', new_callable=io.StringIO), \
                        patch.object(training, 'load_dataset', return_value=(self.words, {}, self.splits, {})), \
                        patch.object(training, 'build_model') as build, patch.object(training, 'run_training') as run:
                    if mode == 'check':
                        training.main()
                    else:
                        with self.assertRaises(SystemExit):
                            training.main()
                build.assert_not_called()
                run.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(sum(s.calls for s in self.splits.values()), 0)
                if mode != 'nested':
                    self.assertTrue(all(s.closed for s in self.splits.values()))

    def test_monitor_uses_one_encoding_for_both_policies_and_no_gold_generation_inputs(self):
        for include_beam in (True, False):
            with self.subTest(include_beam=include_beam), tempfile.TemporaryDirectory() as tmp:
                model = self.helper.model
                before = {key: value.clone() for key, value in model.state_dict().items()}
                generated = []
                for record in self.records['VAL']:
                    row = {'prediction': record['reference'], 'token_ids': record['tokens'],
                           'terminated_with_end': True, 'log_probability': -1.}
                    generated.extend([row] * (2 if include_beam else 1))
                self.splits['VAL'].calls = 0
                with patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(training, 'beam_search', side_effect=generated) as beam, \
                        patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as teacher:
                    result = training.monitor_split(model, self.splits['VAL'], self.records['VAL'], self.words,
                                                     self.config, torch.device('cpu'), Path(tmp), include_beam)
                self.assertEqual(self.splits['VAL'].calls, 2)
                self.assertEqual(set(result['generation']), {'beam5', 'greedy'} if include_beam else {'greedy'})
                widths = [5, 1, 5, 1] if include_beam else [1, 1]
                for i, call in enumerate(beam.call_args_list):
                    self.assertEqual(len(call.args), 3)
                    self.assertIs(call.args[0], model.decoder)
                    self.assertEqual(call.kwargs, {'coefficient': .5, 'width': widths[i], 'max_steps': 161})
                    teacher_index = i // 2 if include_beam else i
                    self.assertIs(call.args[1], teacher.call_args_list[teacher_index].args[0])
                self.assertEqual(result['generation']['greedy']['field_match_rate'], 1.)
                self.assertEqual(result['generation']['greedy']['content_field_match_rate'], 1.)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_content_selection_excludes_only_two_nonvisual_fields(self):
        matches = {field: 0 for field in FIELDS}
        matches['confidence level'] = matches['additional observations'] = 111
        summary = {'generation': {'samples': 111}, 'structure': {'reference_field_exact_matches': matches}}
        self.assertEqual(training.content_match_rate(summary), 0)
        for field in training.CONTENT_FIELDS:
            matches[field] = 111
        self.assertEqual(training.content_match_rate(summary), 1)
        self.assertEqual(len(training.CONTENT_FIELDS), 12)
        self.assertEqual(training.train_monitor_epochs(30, 5), [5, 10, 15, 20, 25, 30])
        self.assertEqual(training.train_monitor_epochs(7, 5), [5, 7])

    def test_mocked_training_keeps_single_optimizer_resets_rng_and_selects_separate_bests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(2, 2)
            source, initial = self.source_fixture(root, model)
            output = root / 'result'
            before = {str(p): p.read_bytes() for p in source.parent.rglob('*') if p.is_file()}
            seen, optimizers, monitored = [], [], []

            def epoch(model, loader, config, device, optimizer=None, scaler=None):
                batches = list(loader)
                self.assertEqual(config['tag_loss_weight'], 0)
                if optimizer is not None:
                    number = len(seen) + 1
                    if number == 1:
                        self.assertEqual(training.state_digest(model.state_dict()), training.state_digest(initial))
                        self.assertFalse(optimizer.state)
                    optimizers.append(optimizer)
                    seen.append([item for batch in batches for item in batch['id']])
                    seed = config['seed'] + number
                    self.assertEqual(torch.rand(1).item(), torch.rand(1, generator=torch.Generator().manual_seed(seed)).item())
                    self.assertEqual(random.random(), random.Random(seed).random())
                    self.assertEqual(float(np.random.rand()), float(np.random.RandomState(seed).rand()))
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(1)
                else:
                    self.assertFalse(config['amp'])
                return {'primary_ce': .5 + len(seen) / 10, 'secondary_ce': .2, 'total': 1., 'tag_bce': .5}

            def monitor(model, split, records, words, config, device, destination, include_beam):
                name, number = destination.name, len(seen)
                monitored.append((number, name, include_beam))
                self.assertIs(split, self.splits[name.upper()])
                torch.rand(17)
                random.random()
                np.random.rand()
                greedy = {'content_field_match_rate': number / 10, 'field_match_rate': .6,
                          'field_agreement': {'glaucoma risk assessment': {'rate': .5, 'macro_recall': .25,
                              'class_recall': {'healthy': 0., 'high risk': 1., 'moderate risk': 0., 'very healthy': 0.}}},
                          'generation': {'unique_reports': 1, 'largest_identical_group': len(records)}}
                return {'generation': {'greedy': greedy}, 'teacher_forcing': {}}

            args = ['train_lem_off.py', '--source-run', str(source), '--output', str(output), '--epochs', '3',
                    '--device', 'cpu', '--accept-reconstruction']
            with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                    patch.object(training, 'load_dataset', return_value=(self.words, {}, self.splits, {})), \
                    patch.object(training, 'build_model', return_value=model), \
                    patch.object(training, 'run_epoch', side_effect=epoch), \
                    patch.object(training, 'monitor_split', side_effect=monitor):
                training.main()
            self.assertEqual(read_json(output / 'status.json'), {'stage': 'complete', 'completed_epochs': 3})
            self.assertTrue(all(s.closed for s in self.splits.values()))
            self.assertEqual(self.splits['TEST'].calls, 0)
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.parent.rglob('*') if p.is_file()})
            self.assertEqual(monitored, [(1, 'val', True), (1, 'train', False), (2, 'val', True),
                                         (3, 'val', True), (3, 'train', False)])
            self.assertEqual(len({id(o) for o in optimizers}), 1)
            plan = read_json(output / 'training_order.json')
            self.assertEqual(seen, [row['ids'] for row in plan])
            best = torch.load(output / 'best.pt', weights_only=True)
            fields = torch.load(output / 'best_fields.pt', weights_only=True)
            last = torch.load(output / 'last.pt', weights_only=True)
            self.assertEqual((best['epoch'], fields['epoch'], last['epoch']), (1, 3, 3))
            self.assertEqual(best['selection_metric'], 'validation_report_ce')
            self.assertEqual(fields['selection_metric'], 'validation_greedy_content_field_match_rate')
            self.assertEqual(last['format'], training.FORMAT)
            self.assertIn('optimizer', last)
            self.assertIn('scaler', last)
            self.assertTrue(read_json(output / 'reference_epoch_comparison.json')['matches_source_final_state'])
            history = read_json(output / 'history.json')
            self.assertIsNone(history[1]['train_monitor'])
            self.assertEqual(read_json(output / 'manifest.json')['config_changes'], {'epochs': {'before': 1, 'after': 3}})
            with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                    patch.object(training, 'load_dataset') as load:
                with self.assertRaises(SystemExit):
                    training.main()
                load.assert_not_called()

    def test_failure_writes_status_and_closes_readers_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(2, 2)
            source, _ = self.source_fixture(root, model)
            output = root / 'result'
            args = ['train_lem_off.py', '--source-run', str(source), '--output', str(output), '--epochs', '3',
                    '--device', 'cpu', '--accept-reconstruction']
            with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                    patch.object(training, 'load_dataset', return_value=(self.words, {}, self.splits, {})), \
                    patch.object(training, 'build_model', return_value=model), \
                    patch.object(training, 'run_epoch', side_effect=ValueError('synthetic failure')):
                with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                    training.main()
            self.assertEqual(read_json(output / 'status.json')['stage'], 'failed')
            self.assertEqual(read_json(source / 'status.json')['stage'], 'complete')
            self.assertTrue(all(s.closed for s in self.splits.values()))


if __name__ == '__main__':
    unittest.main()
