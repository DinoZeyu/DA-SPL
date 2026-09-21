"""Paired-training invariants with synthetic gradients and mocked training/generation."""

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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import ablate_lem as audit
from paper_method.data import fingerprint, read_json, write_json
from paper_method.diagnose_endings import ending_record
from paper_method.evaluation import FIELDS
from paper_method.losses import objective
from paper_method.model import DASPL
from paper_method.runner import PACKAGE, validate_config
import test_risk_diagnostic


class LemAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.config = {**read_json(PACKAGE / 'config.json'), 'amp': False, 'epochs': 2, 'batch_size': 2}
        helper = test_risk_diagnostic.RiskDiagnosticTests()
        helper.setUp()
        self.words, self.tokens = helper.words, helper.tokens

    def readers(self):
        tokens = self.tokens

        class Reader:
            def __init__(self, name):
                self.name, self.calls, self.closed = name, 0, False
                size = 3 if name == 'TRAIN' else 1
                self.records = [{'id': f'{name.lower()}:{i}'} for i in range(size)]
                self.captions, self.lengths = [tokens] * size, [len(tokens)] * size

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                if self.name == 'TEST':
                    raise AssertionError('No TEST image access')
                self.calls += 1
                return {'id': self.records[index]['id'], 'image': torch.zeros(3, 8),
                        'caption': torch.tensor(tokens), 'length': torch.tensor(len(tokens)),
                        'tags': torch.tensor([1., 0., 1.])}

            def close(self):
                self.closed = True

        return {name: Reader(name) for name in ('TRAIN', 'VAL', 'TEST')}

    def test_only_loss_weight_changes_and_normal_runner_remains_strict(self):
        before = copy.deepcopy(self.config)
        configs = audit.arm_configs(self.config)
        self.assertEqual(self.config, before)
        self.assertEqual(configs['lem_on'], before)
        self.assertEqual([key for key in before if configs['lem_on'][key] != configs['lem_off'][key]], ['tag_loss_weight'])
        self.assertEqual(configs['lem_off']['tag_loss_weight'], 0)
        with self.assertRaises(ValueError):
            validate_config(configs['lem_off'])

    def test_zero_lem_full_model_gradients_equal_ce_and_ignore_tags(self):
        model = DASPL(torch.nn.Identity(), len(self.words), 3, dim=8, heads=2, pad_id=self.words['<pad>'])
        memory = torch.randn(1, 3, 8)
        captions, lengths = torch.tensor([self.tokens]), torch.tensor([len(self.tokens)])
        results = []
        for mode in ('off_zeros', 'off_ones', 'ce_only'):
            model.zero_grad(set_to_none=True)
            outputs = model.from_memory(memory, captions, lengths)
            if mode == 'ce_only':
                loss = F.cross_entropy(outputs['primary_logits'][0], captions[0, 1:])
                loss += .5 * F.cross_entropy(outputs['secondary_logits'][0], captions[0, 1:])
            else:
                tags = torch.ones(1, 3) if mode == 'off_ones' else torch.zeros(1, 3)
                loss, _, _ = objective(outputs, captions, lengths, tags, coefficient=.5, tag_weight=0.)
            loss.backward()
            results.append({key: parameter.grad.clone() for key, parameter in model.decoder.named_parameters()})
            if mode != 'ce_only':
                self.assertTrue(all(parameter.grad.abs().sum().item() == 0
                                    for parameter in model.label_enhancement.parameters()))
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
            torch.testing.assert_close(results[0][key], results[2][key], rtol=0, atol=0)

    def test_shared_orders_cover_all_samples_and_check_delivered_ids(self):
        split = self.readers()['TRAIN']
        orders = audit.training_orders(len(split), 3, 123)
        self.assertEqual(orders, audit.training_orders(len(split), 3, 123))
        torch.manual_seed(22)
        expected_rng = torch.rand(2)
        torch.manual_seed(22)
        for order in orders:
            loader = audit.OrderedBatches(split, order, 2, 123)
            batches = list(loader)
            self.assertEqual([len(batch['id']) for batch in batches], [2, 1])
            self.assertEqual(loader.seen, [split.records[i]['id'] for i in order])
        torch.testing.assert_close(torch.rand(2), expected_rng, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            audit.OrderedBatches(split, [0, 0, 1], 2, 123)
        bad = audit.OrderedBatches(split, orders[0], 2, 123)
        bad.loader = [{'id': ['wrong', 'ids']}]
        with self.assertRaisesRegex(ValueError, 'IDs differ'):
            list(bad)

    def test_generation_metrics_count_invalid_risk_without_rewriting(self):
        healthy = 'glaucoma risk assessment : very healthy . confidence level : 0.95 .'
        risk = 'glaucoma risk assessment : high risk . confidence level : 0.9 .'
        malformed = 'glaucoma risk assessment : very healthy : very healthy . confidence level : 0.95 .'
        records = [{'id': f'val:{i}', 'risk': 'very healthy'} for i in range(4)]
        predictions = [{'id': row['id'], 'reference': healthy, 'prediction': text, 'terminated_with_end': True}
                       for row, text in zip(records, (healthy, risk, malformed, risk))]
        before = copy.deepcopy(predictions)
        result = audit.generation_summary(predictions, records)
        self.assertEqual(result['risk_matches'], 1)
        self.assertEqual(result['invalid_risk'], 1)
        self.assertEqual(result['healthy_to_high_risk'], 2)
        self.assertEqual(result['largest_identical_group'], 2)
        self.assertEqual(predictions, before)
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            audit.generation_summary(predictions[::-1], records)

    def test_val_audit_uses_same_memory_and_keeps_generation_separate_from_teacher_forcing(self):
        report = ' '.join(f'{field} : {"very healthy" if field == "glaucoma risk assessment" else "0.95" if field == "confidence level" else "value"} .'
                          for field in FIELDS)
        words = {word: i for i, word in enumerate(dict.fromkeys(['<pad>', '<unk>', '<start>', '<end>'] + report.split()))}
        tokens = [words['<start>']] + [words[word] for word in report.split()] + [words['<end>']]
        record = ending_record('val:0', tokens, len(tokens), words)
        model = DASPL(torch.nn.Identity(), len(words), 3, dim=8, heads=2, pad_id=words['<pad>']).eval()
        split = self.readers()['VAL']
        generated = {'prediction': report, 'token_ids': tokens, 'terminated_with_end': True, 'log_probability': -2.}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(audit, 'beam_search', return_value=generated) as beam, \
                patch.object(audit, 'score_batch', wraps=audit.score_batch) as score, \
                patch('sys.stdout', new_callable=io.StringIO):
            result = audit.audit_validation(model, split, [record], words, self.config, torch.device('cpu'), Path(temporary))
            self.assertEqual(result['generation']['risk_matches'], 1)
            self.assertEqual(result['endings']['all']['samples'], 1)
            self.assertEqual(read_json(Path(temporary) / 'predictions.json')[0]['prediction'], report)
            self.assertIs(beam.call_args.args[1], score.call_args.args[1][0])
            self.assertEqual(beam.call_args.args[3:], (.5, 5, 161))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_compare_uses_matched_epochs_and_off_minus_on_direction(self):
        record = [{'id': 'val:0', 'risk': 'very healthy'}]
        prediction = [{'id': 'val:0', 'reference': 'x', 'prediction': 'x', 'terminated_with_end': True}]
        metrics = audit.generation_summary(prediction, record)
        on = [{'epoch': i, 'validation': {'generation': metrics}} for i in (1, 2)]
        off = copy.deepcopy(on)
        off[1]['validation']['generation']['risk_matches'] = 1
        result = audit.compare_histories({'lem_on': on, 'lem_off': off})
        self.assertEqual(result['primary_comparison_epoch'], 2)
        self.assertEqual(result['epochs'][1]['off_minus_on']['risk_matches'], 1)
        with self.assertRaisesRegex(ValueError, 'matching epochs'):
            audit.compare_histories({'lem_on': on, 'lem_off': off[:1]})

    def test_mocked_pair_checks_initialization_rng_orders_fresh_optimizers_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            weights = root / 'weights.pth'
            weights.write_bytes(b'fixture; build_model is mocked')
            config_path = root / 'config.json'
            write_json(config_path, {**self.config, 'data_directory': str(root / 'data'), 'pretrained_weights': str(weights)})
            before = {path.name: fingerprint(path) for path in root.iterdir() if path.is_file()}
            splits = self.readers()
            model = torch.nn.Linear(2, 2)
            initial_hash = audit.state_digest(model.state_dict())
            seen, rngs, optimizers, initial_hashes = {}, {}, {}, {}

            def epoch(model, loader, config, device, optimizer=None, scaler=None):
                batches = list(loader)
                weight = config['tag_loss_weight']
                if optimizer is not None:
                    count = len(seen.setdefault(weight, [])) + 1
                    if count == 1:
                        initial_hashes[weight] = audit.state_digest(model.state_dict())
                        optimizers[weight] = optimizer
                        self.assertFalse(optimizer.state)
                    seen[weight].append([sample_id for batch in batches for sample_id in batch['id']])
                    rngs.setdefault(weight, []).append((torch.rand(1).item(), random.random(), float(np.random.rand())))
                    with torch.no_grad():
                        model.weight.add_(1)
                    return {'primary_ce': .5, 'secondary_ce': .4, 'tag_bce': .1, 'total': .7 + weight * .1}
                self.assertFalse(config['amp'])
                count = len(seen[weight])
                # Epoch two has lower TOTAL loss but worse report CE: best must remain epoch one.
                return {'primary_ce': .5 if count == 1 else .6, 'secondary_ce': .4,
                        'tag_bce': .8 if count == 1 else .01, 'total': 4.7 if count == 1 else .85}

            def validate(model, split, records, words, config, device, destination):
                self.assertIs(split, splits['VAL'])
                torch.rand(7 if config['tag_loss_weight'] else 13)
                random.random()
                np.random.rand()
                prediction = 'glaucoma risk assessment : very healthy . confidence level : 0.95 .'
                rows = [{'id': record['id'], 'prediction': prediction, 'reference': prediction,
                         'terminated_with_end': True} for record in records]
                return {'generation': audit.generation_summary(rows, records), 'endings': {}, 'structure': {}}

            args = ['ablate_lem.py', '--config', str(config_path), '--output', str(output), '--device', 'cpu',
                    '--accept-reconstruction']
            with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                    patch.object(audit, 'load_dataset', return_value=(self.words, {}, splits, {})), \
                    patch.object(audit, 'build_model', return_value=model) as build, \
                    patch.object(audit, 'run_epoch', side_effect=epoch), \
                    patch.object(audit, 'audit_validation', side_effect=validate):
                audit.main()
            build.assert_called_once()
            self.assertEqual(initial_hashes, {5.: initial_hash, 0.: initial_hash})
            self.assertEqual(seen[5.], seen[0.])
            self.assertEqual(rngs[5.], rngs[0.])
            self.assertIsNot(optimizers[5.], optimizers[0.])
            self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
            for arm in audit.ARMS:
                self.assertEqual(read_json(output / arm / 'status.json')['stage'], 'complete')
                best = torch.load(output / arm / 'best.pt', weights_only=True)
                last = torch.load(output / arm / 'last.pt', weights_only=True)
                self.assertEqual(best['format'], audit.FORMAT)
                self.assertEqual(best['epoch'], 1)
                self.assertEqual(last['epoch'], 2)
                self.assertEqual(read_json(output / arm / 'epoch_001/train_ids.json'), seen[5.][0])
            self.assertTrue(all(split.closed for split in splits.values()))
            self.assertEqual(splits['TEST'].calls, 0)
            self.assertEqual(before, {name: fingerprint(root / name) for name in before})
            with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                    patch.object(audit, 'load_dataset') as load:
                with self.assertRaises(SystemExit):
                    audit.main()
                load.assert_not_called()

    def test_preflight_acknowledgement_and_saved_run_guards_do_not_train_or_write(self):
        for mode in ('check', 'missing_ack', 'nested_saved_run'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                weights, config_path = root / 'weights.pth', root / 'config.json'
                weights.write_bytes(b'fixture')
                write_json(config_path, {**self.config, 'data_directory': str(root / 'data'), 'pretrained_weights': str(weights)})
                output = root / 'output'
                splits = self.readers()
                if mode == 'nested_saved_run':
                    write_json(root / 'status.json', {'stage': 'complete'})
                args = ['ablate_lem.py', '--config', str(config_path), '--output', str(output)]
                if mode == 'check':
                    args.append('--check')
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch('sys.stderr', new_callable=io.StringIO), \
                        patch.object(audit, 'load_dataset', return_value=(self.words, {}, splits, {})), \
                        patch.object(audit, 'build_model') as build, patch.object(audit, 'run_pair') as run:
                    if mode == 'check':
                        audit.main()
                    else:
                        with self.assertRaises(SystemExit):
                            audit.main()
                build.assert_not_called()
                run.assert_not_called()
                self.assertFalse(output.exists())
                self.assertTrue(all(split.calls == 0 for split in splits.values()))
                if mode != 'nested_saved_run':
                    self.assertTrue(all(split.closed for split in splits.values()))


if __name__ == '__main__':
    unittest.main()
