"""Synthetic field probes and mocked data loading; no actual retinal inference or training."""

import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import diagnose_fields as audit
from paper_method.data import read_json, write_json
from paper_method.model import DASPL, mixture_log_probs
from paper_method.runner import archive_sources, PACKAGE


class FieldDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        vocabulary = ('<pad> <unk> <start> <end> optic disc size : normal large . cup to ratio '
                      '0.3 0.4 0.8 rim color pink pale orange with no abnormal discoloration '
                      'glaucoma risk assessment high very healthy confidence level 0.9 0.95').split()
        self.words = {word: i for i, word in enumerate(vocabulary)}
        self.texts = [
            self.report('normal', '0.4', 'pink', 'very healthy'),
            self.report('large', '0.8', 'pale', 'high risk'),
            self.report('normal', '0.8', 'pink', 'high risk'),
            self.report('large', '0.4', 'orange', 'very healthy'),
        ]
        words, texts = self.words, self.texts

        class Reader:
            def __init__(self, split):
                self.name, self.calls, self.closed = split, 0, False
                self.records = [{'id': f'{split.lower()}:{i}'} for i in range(len(texts))]
                self.captions = [[words[w] for w in ('<start> ' + text + ' <end>').split()] for text in texts]
                self.lengths = [len(tokens) for tokens in self.captions]

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                if self.name != 'VAL':
                    raise AssertionError('Only VAL image access is allowed')
                self.calls += 1
                return {'id': self.records[index]['id'], 'image': torch.full((3, 8), float(index))}

            def close(self):
                self.closed = True

        self.splits = {name: Reader(name) for name in ('TRAIN', 'VAL', 'TEST')}
        self.records = audit.report_records(self.splits['VAL'], self.words)
        self.train = audit.report_records(self.splits['TRAIN'], self.words)
        self.candidates, _ = audit.candidate_audit(self.train, self.records, self.words)
        self.model = DASPL(torch.nn.Identity(), len(words), 3, dim=8, heads=2, pad_id=words['<pad>']).eval()
        self.config = {**read_json(PACKAGE / 'config.json'), 'tag_loss_weight': 0., 'amp': False}

    def report(self, size, ratio, color, risk):
        return (f'optic disc size : {size} . cup to disc ratio : {ratio} . rim color : {color} . '
                f'glaucoma risk assessment : {risk} . confidence level : 0.95 .')

    def encode(self, text):
        return [self.words[word] for word in text.split()]

    def predictions(self):
        rows = []
        for i, record in enumerate(self.records):
            tokens = self.splits['VAL'].captions[i]
            row = {'id': record['id'], 'reference': record['reference'], 'prediction': record['reference'],
                   'token_ids': tokens, 'terminated_with_end': True}
            self.set_score(row, i)
            rows.append(row)
        return rows

    def set_score(self, row, index):
        memory = torch.full((1, 3, 8), float(index))
        tokens = torch.tensor([row['token_ids']])
        with torch.no_grad():
            first, second = self.model.decoder(memory, tokens, torch.tensor([tokens.size(1)]))
            distribution = mixture_log_probs(first, second, .5)
            row['log_probability'] = distribution[0].gather(1, tokens[0, 1:, None]).double().sum().item()

    def test_prefix_stops_before_value_and_uses_first_marker_without_later_risk(self):
        tokens = self.splits['VAL'].captions[0]
        expected = self.encode('<start> optic disc size : normal . cup to disc ratio :')
        self.assertEqual(audit.field_prefix(tokens, 'cup to disc ratio', self.words), expected)
        repeated = tokens[:-1] + self.encode('cup to disc ratio : 0.8 . <end>')
        self.assertEqual(audit.field_prefix(repeated, 'cup to disc ratio', self.words), expected)
        self.assertIsNone(audit.field_prefix(self.encode('<start> optic disc size : normal . <end>'), 'rim color', self.words))
        with self.assertRaises(ValueError):
            audit.field_prefix(self.encode('<start> <end> rim color :'), 'rim color', self.words)

    def test_candidates_use_train_only_and_do_not_normalize_uncovered_values(self):
        train, val = copy.deepcopy(self.train), copy.deepcopy(self.records)
        train[0]['fields']['rim color']['value'] = 'orange with no abnormal discoloration'
        val[0]['fields']['cup to disc ratio']['value'] = '0.3'
        candidates, coverage = audit.candidate_audit(train, val, self.words)
        self.assertNotIn('0.3', candidates['cup to disc ratio'])
        self.assertEqual(coverage['cup to disc ratio']['uncovered_val_counts'], {'0.3': 1})
        self.assertEqual(coverage['rim color']['excluded_train_values'], {'orange with no abnormal discoloration': 1})
        self.assertEqual(train[0]['fields']['rim color']['value'], 'orange with no abnormal discoloration')

    def test_donors_are_seeded_different_value_and_prefer_same_risk(self):
        donors = audit.pair_donors(self.records, self.candidates, 123)
        self.assertEqual(donors, audit.pair_donors(self.records, self.candidates, 123))
        for field, indices in donors.items():
            for i, j in enumerate(indices):
                a, b = self.records[i], self.records[j]
                self.assertNotEqual(i, j)
                self.assertNotEqual(a['fields'][field]['value'], b['fields'][field]['value'])
                same_risk_exists = any(row['risk'] == a['risk'] and row['fields'][field]['value'] != a['fields'][field]['value']
                                       for row in self.records)
                if same_risk_exists:
                    self.assertEqual(a['risk'], b['risk'])
                else:
                    self.assertNotEqual(a['risk'], b['risk'])
        changed = copy.deepcopy(self.records)
        changed[0]['fields']['cup to disc ratio']['value'] = '0.3'
        self.assertIsNone(audit.pair_donors(changed, self.candidates, 123)['cup to disc ratio'][0])

    def test_field_scores_match_teacher_forced_value_and_period_probabilities(self):
        decoder, memory = self.model.decoder, torch.randn(1, 3, 8)
        before = {key: value.clone() for key, value in decoder.state_dict().items()}
        for field in audit.FIELDS:
            prefix = self.records[0]['fields'][field]['prefix_ids']
            scored = audit.score_field(decoder, memory, prefix, field, self.candidates[field], self.words, .5)
            for value in self.candidates[field]:
                tokens = prefix + [self.words[value], self.words['.'], self.words['<end>']]
                with torch.no_grad():
                    first, second = decoder(memory, torch.tensor([tokens]), torch.tensor([len(tokens)]))
                    heads = {'primary': first.log_softmax(-1), 'secondary': second.log_softmax(-1),
                             'mixture': mixture_log_probs(first, second, .5)}
                for head, distribution in heads.items():
                    offset = len(prefix) - 1
                    prob = distribution[0, offset, self.words[value]]
                    self.assertAlmostEqual(scored[head]['value_probabilities'][value], prob.exp().item(), places=6)
                    expected = prob + distribution[0, offset + 1, self.words['.']]
                    self.assertAlmostEqual(scored[head]['value_period_log_probabilities'][value], expected.item(), places=5)
                    self.assertLess(scored[head]['candidate_mass'], 1.)
        for key, value in decoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        with self.assertRaises(ValueError):
            audit.score_field(decoder, memory.expand(2, -1, -1), prefix, field, self.candidates[field], self.words, .5)

    def test_saved_predictions_and_score_replay_reject_mismatches(self):
        predictions = self.predictions()
        audit.validate_predictions(predictions, self.records, self.words, 161)
        memory = torch.zeros(1, 3, 8)
        audit.replay_prediction(self.model.decoder, memory, predictions[0], .5)
        changed = copy.deepcopy(predictions)
        changed[0]['prediction'] += ' .'
        with self.assertRaises(ValueError):
            audit.validate_predictions(changed, self.records, self.words, 161)
        changed = copy.deepcopy(predictions)
        changed[0]['terminated_with_end'] = False
        with self.assertRaises(ValueError):
            audit.validate_predictions(changed, self.records, self.words, 161)
        changed = copy.deepcopy(predictions[0])
        changed['log_probability'] += 1
        with self.assertRaisesRegex(ValueError, 'replay mismatch'):
            audit.replay_prediction(self.model.decoder, memory, changed, .5)

    def test_probes_hold_prefix_fixed_and_do_not_impute_missing_generated_markers(self):
        predictions = self.predictions()
        predictions[0]['prediction'] = 'optic disc size : normal .'
        predictions[0]['token_ids'] = self.encode('<start> ' + predictions[0]['prediction'] + ' <end>')
        self.set_score(predictions[0], 0)
        donors = audit.pair_donors(self.records, self.candidates, 123)
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        with patch('sys.stdout', new_callable=io.StringIO), patch.object(audit, 'score_field', wraps=audit.score_field) as scorer:
            rows, replays = audit.diagnose(self.model, self.splits['VAL'], self.records, predictions, self.candidates,
                                          donors, self.words, self.config, torch.device('cpu'))
        for row in rows:
            i = next(i for i, record in enumerate(self.records) if record['id'] == row['id'])
            for name, profile in row['conditions'].items():
                image_index = donors[row['field']][i] if name.startswith('donor') else i
                prefix = row['generated_prefix_ids'] if name.endswith('generated_prefix') else row['reference_prefix_ids']
                matching = [call for call in scorer.call_args_list if call.args[3] == row['field'] and call.args[2] == prefix
                            and torch.equal(call.args[1], torch.full((1, 3, 8), float(image_index)))]
                self.assertTrue(matching)
            if row['id'] == 'val:0' and row['field'] != 'optic disc size':
                self.assertIsNone(row['generated_prefix_ids'])
                self.assertNotIn('own_image_generated_prefix', row['conditions'])
                self.assertNotIn('donor_image_generated_prefix', row['conditions'])
        summary = audit.summarize(rows)['fields']
        self.assertEqual(len(replays), 4)
        cup = summary['cup to disc ratio']
        self.assertEqual(cup['missing_generated_prefix'], 1)
        self.assertEqual(cup['contrasts']['image_swap_generated_prefix']['all']['mixture']['paired_samples'], 3)
        self.assertEqual(cup['contrasts']['image_swap_generated_prefix']['same_risk']['mixture']['paired_samples'], 0)
        self.assertIsNone(cup['contrasts']['image_swap_generated_prefix']['same_risk']['mixture']['mean_tv_candidates_plus_other'])
        size = summary['optic disc size']['contrasts']['reference_to_generated_prefix']['all']['mixture']
        self.assertEqual(size['mean_tv_candidates_plus_other'], 0.)
        self.assertEqual(self.splits['TRAIN'].calls + self.splits['TEST'].calls, 0)
        self.assertEqual(self.splits['VAL'].calls, 4)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def source_fixture(self, root):
        parent = root / 'paired'
        source = parent / 'lem_off'
        source.mkdir(parents=True)
        weights = root / 'weights.pth'
        weights.write_bytes(b'fixture, model builder mocked')
        config = {**self.config, 'pretrained_weights': str(weights), 'data_directory': str(root / 'data')}
        write_json(source / 'config.json', config)
        write_json(parent / 'config.json', {**config, 'tag_loss_weight': 5.})
        write_json(source / 'manifest.json', {'format': audit.FORMAT, 'arm': 'lem_off',
                   'tag_loss_weight': 0., 'initial_state_sha256': 'initial'})
        write_json(parent / 'manifest.json', {'experiment': audit.FORMAT})
        write_json(source / 'status.json', {'stage': 'complete', 'completed_epochs': 5})
        write_json(parent / 'status.json', {'stage': 'complete', 'epochs_per_arm': 5})
        for folder in (source, parent):
            write_json(folder / 'data_fingerprints.json', {})
        archive_sources(parent)
        checkpoint = {'format': audit.FORMAT, 'arm': 'lem_off', 'config': config, 'words': self.words,
                      'tags': {}, 'data_fingerprints': {}, 'epoch': 5, 'initial_state_sha256': 'initial',
                      'model': self.model.state_dict()}
        torch.save(checkpoint, source / 'last.pt')
        epoch = source / 'epoch_005'
        epoch.mkdir()
        write_json(epoch / 'predictions.json', self.predictions())
        return source, checkpoint

    def test_ablation_loader_is_strict_about_arm_final_epoch_data_and_archived_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, original = self.source_fixture(Path(temporary))
            config, checkpoint = audit.load_source(source, self.words, {}, {})
            self.assertEqual(checkpoint['epoch'], 5)
            self.assertEqual(config['tag_loss_weight'], 0.)
            for change in ({'format': 'wrong'}, {'arm': 'lem_on'}, {'epoch': 4}, {'data_fingerprints': {'changed': 'hash'}},
                           {'words': {}}, {'initial_state_sha256': 'other'}):
                with self.subTest(change=change):
                    torch.save({**original, **change}, source / 'last.pt')
                    with self.assertRaisesRegex(ValueError, 'checkpoint .* mismatch'):
                        audit.load_source(source, self.words, {}, {})
            torch.save(original, source / 'last.pt')
            with patch.object(audit.tarfile, 'open') as opened:
                opened.return_value.__enter__.return_value.extractfile.return_value.read.return_value = b'changed'
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    audit.load_source(source, self.words, {}, {})

    def test_cli_preflight_and_synthetic_pipeline_preserve_source_and_reject_overwrite(self):
        for check in (True, False):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, _ = self.source_fixture(root)
                output = root / 'output'
                before = {str(path): path.read_bytes() for path in source.parent.rglob('*') if path.is_file()}
                args = ['diagnose_fields.py', '--run', str(source), '--output', str(output), '--device', 'cpu']
                if check:
                    args.append('--check')
                for split in self.splits.values():
                    split.calls, split.closed = 0, False
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(audit, 'load_dataset', return_value=(self.words, {}, self.splits, {})), \
                        patch.object(audit, 'build_model', return_value=self.model) as build, \
                        patch('torch.optim.Adam', side_effect=AssertionError('No optimization')):
                    audit.main()
                self.assertEqual(before, {str(path): path.read_bytes() for path in source.parent.rglob('*') if path.is_file()})
                self.assertTrue(all(split.closed for split in self.splits.values()))
                self.assertEqual(self.splits['TRAIN'].calls + self.splits['TEST'].calls, 0)
                if check:
                    build.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertEqual(self.splits['VAL'].calls, 0)
                else:
                    self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
                    self.assertEqual(read_json(output / 'status.json')['field_cases'], 12)
                    with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                            patch.object(audit, 'load_dataset') as load:
                        with self.assertRaises(SystemExit):
                            audit.main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
