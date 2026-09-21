"""Synthetic and mocked audit checks; never infer on actual retinal images."""

import copy
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import diagnose_risk as audit
from paper_method.data import read_json, write_json
from paper_method.model import ParallelDecoder, mixture_log_probs
import test_runtime


class RiskDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        tokens = ('<pad>', '<unk>', '<start>', '<end>', 'rim', 'color', ':', 'pink', 'pale', '.',
                  'glaucoma', 'risk', 'assessment', 'healthy', 'very', 'moderate', 'high',
                  'confidence', 'level', '0.95', '0.9')
        self.words = {word: i for i, word in enumerate(tokens)}
        self.text = 'rim color : pink . glaucoma risk assessment : very healthy . confidence level : 0.95 .'
        self.tokens = self.encode('<start> ' + self.text + ' <end>')
        self.prefix = audit.extract_prefix(self.tokens, self.words)

    def encode(self, text):
        return [self.words[word] for word in text.split()]

    def profile(self, choice):
        probabilities = {label: .7 if label == choice else .05 for label in audit.RISKS}
        profile = {'risk_start_probabilities': probabilities, 'risk_start_total_mass': .85,
                   'restricted_first_token_choice': choice, 'restricted_sequence_choice': choice,
                   'sequence_log_probabilities': {label: math.log(p) for label, p in probabilities.items()}}
        return {head: copy.deepcopy(profile) for head in audit.HEADS}

    def test_prefix_excludes_risk_value_confidence_and_uses_first_marker(self):
        expected = self.encode('<start> rim color : pink . glaucoma risk assessment :')
        self.assertEqual(self.prefix, expected)
        repeated = self.tokens[:-1] + self.encode('glaucoma risk assessment : high risk . <end>')
        self.assertEqual(audit.extract_prefix(repeated, self.words), expected)
        self.assertIsNone(audit.extract_prefix(self.encode('<start> rim color : pink . <end>'), self.words))
        with self.assertRaises(ValueError):
            audit.extract_prefix(self.encode('<start> <end> glaucoma risk assessment :'), self.words)

    def test_reference_cases_validate_labels_without_feeding_them_into_prefix(self):
        split = SimpleNamespace(records=[{'id': 'val:0'}], captions=[self.tokens], lengths=[len(self.tokens)])
        rows = audit.reference_cases(split, self.words)
        self.assertEqual(rows[0]['reference_risk'], 'very healthy')
        self.assertEqual(rows[0]['reference_prefix_ids'], self.prefix)
        split.captions = [self.encode('<start> glaucoma risk assessment : pink . <end>')]
        split.lengths = [len(split.captions[0])]
        with self.assertRaises(ValueError):
            audit.reference_cases(split, self.words)

    def test_pairing_is_seeded_opposite_group_and_reuses_donors_evenly(self):
        labels = ['very healthy'] * 3 + ['healthy'] + ['high risk'] * 5 + ['moderate risk'] * 2
        cases = [{'reference_risk': label} for label in labels]
        donors = audit.pair_donors(cases, 123)
        self.assertEqual(donors, audit.pair_donors(cases, 123))
        for i, j in enumerate(donors):
            self.assertNotEqual(i, j)
            self.assertNotEqual(audit.risk_group(labels[i]), audit.risk_group(labels[j]))
        for group in ('healthy', 'at_risk'):
            counts = [donors.count(i) for i, label in enumerate(labels) if audit.risk_group(label) == group]
            self.assertLessEqual(max(counts) - min(counts), 1)
        with self.assertRaises(ValueError):
            audit.pair_donors([{'reference_risk': 'healthy'}], 123)

    def test_candidate_scores_match_teacher_forcing_for_each_head_without_parameter_changes(self):
        decoder = ParallelDecoder(len(self.words), 8, 2, self.words['<pad>']).eval()
        memory = torch.randn(1, 3, 8)
        before = {key: value.clone() for key, value in decoder.state_dict().items()}
        result = audit.score_prefix(decoder, memory, self.prefix, self.words, .5)
        for label, candidate in audit.candidate_tokens(self.words).items():
            tokens = self.prefix + candidate + [self.words['<end>']]
            with torch.no_grad():
                first, second = decoder(memory, torch.tensor([tokens]), torch.tensor([len(tokens)]))
            heads = {'primary': first.log_softmax(-1), 'secondary': second.log_softmax(-1),
                     'mixture': mixture_log_probs(first, second, .5)}
            for head, distribution in heads.items():
                offset = len(self.prefix) - 1
                expected = sum(distribution[0, offset + i, token].item() for i, token in enumerate(candidate))
                self.assertAlmostEqual(result[head]['sequence_log_probabilities'][label], expected, places=5)
                self.assertAlmostEqual(result[head]['risk_start_probabilities'][label],
                                       distribution[0, offset, candidate[0]].exp().item(), places=6)
                self.assertEqual(result[head]['sequence_token_counts'][label], len(candidate))
        for key, value in decoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        again = audit.score_prefix(decoder, memory, self.prefix, self.words, .5)
        self.assertEqual(result, again)
        with self.assertRaises(ValueError):
            audit.score_prefix(decoder, memory, self.tokens, self.words, .5)
        with self.assertRaises(ValueError):
            audit.score_prefix(decoder, memory.expand(2, -1, -1), self.prefix, self.words, .5)
        with self.assertRaises(ValueError):
            audit.score_prefix(decoder.train(), memory, self.prefix, self.words, .5)

    def test_strict_risk_parser_rejects_internal_repeats_and_missing_values(self):
        self.assertEqual(audit.strict_generated_risk(self.text)['label'], 'very healthy')
        for text in ('glaucoma risk assessment : very healthy : very healthy .',
                     'glaucoma risk assessment : high risk . glaucoma risk assessment : high risk .',
                     'glaucoma risk assessment : high risk', 'rim color : pink .'):
            with self.subTest(text=text):
                self.assertFalse(audit.strict_generated_risk(text)['valid'])

    def test_summaries_use_paired_denominators_and_do_not_impute_missing_prefixes(self):
        base = {'id': 'val:0', 'donor_id': 'val:1', 'reference_risk': 'very healthy',
                'donor_reference_risk': 'high risk', 'generation': {'prediction': self.text, 'terminated_with_end': True},
                'generated_risk': audit.strict_generated_risk(self.text), 'generated_prefix_ids': self.prefix,
                'conditions': {'own_image_reference_prefix': self.profile('very healthy'),
                               'donor_image_reference_prefix': self.profile('high risk'),
                               'own_image_donor_prefix': self.profile('high risk'),
                               'own_image_generated_prefix': self.profile('high risk')}}
        missing = copy.deepcopy(base)
        missing.update(id='val:2', generated_prefix_ids=None,
                       generation={'prediction': 'rim color : pink .', 'terminated_with_end': True},
                       generated_risk={'label': None, 'valid': False, 'entries': []})
        del missing['conditions']['own_image_generated_prefix']
        summary = audit.summarize([base, missing])
        contrast = summary['contrasts']['reference_to_generated_prefix']['mixture']
        self.assertEqual(contrast['paired_samples'], 1)
        self.assertEqual(contrast['first_token_reference_matches_before'], 1)
        self.assertEqual(contrast['first_token_reference_matches_after'], 0)
        self.assertAlmostEqual(contrast['mean_risk_start_tv_with_other_bucket'], .65)
        self.assertAlmostEqual(contrast['mean_donor_group_risk_start_mass_change'], .65)
        self.assertEqual(summary['generation']['invalid_or_missing_risk_values'], 1)
        self.assertEqual(summary['generation']['missing_risk_prefix'], 1)
        empty = summary['contrasts']['image_swap_generated_prefix']['mixture']
        self.assertEqual(empty['paired_samples'], 0)
        self.assertIsNone(empty['mean_risk_start_tv_with_other_bucket'])

    def test_diagnose_uses_correct_images_prefixes_and_never_supplies_risk_labels_to_decoder(self):
        class Split:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {'id': f'val:{index}', 'image': torch.full((2, 4), float(index))}

        model = torch.nn.Module()
        model.encoder = torch.nn.Identity()
        model.decoder = torch.nn.Identity()
        other_prefix = self.encode('<start> rim color : pale . glaucoma risk assessment :')
        cases = [{'id': 'val:0', 'reference_risk': 'very healthy', 'reference_prefix_ids': self.prefix},
                 {'id': 'val:1', 'reference_risk': 'high risk', 'reference_prefix_ids': other_prefix}]
        config = {'lambda_parallel': .5, 'beam_size': 5, 'max_decode_steps': 161}
        generated = {'prediction': self.text, 'token_ids': self.tokens, 'terminated_with_end': True}
        missing = {'prediction': 'rim color : pink .',
                   'token_ids': self.encode('<start> rim color : pink . <end>'), 'terminated_with_end': True}
        with patch.object(audit, 'score_prefix', return_value=self.profile('very healthy')) as score, \
                patch.object(audit, 'beam_search', side_effect=[generated, missing]), \
                patch('sys.stdout', new_callable=io.StringIO):
            rows = audit.diagnose(model, Split(), self.words, config, torch.device('cpu'), cases, [1, 0])
        self.assertEqual(score.call_count, 8)
        for call, memory_value, prefix in zip(score.call_args_list[:5], (0, 1, 0, 0, 1),
                                               (self.prefix, self.prefix, other_prefix, self.prefix, self.prefix)):
            self.assertEqual(call.args[1].mean().item(), memory_value)
            self.assertEqual(call.args[2], prefix)
        self.assertNotIn('own_image_generated_prefix', rows[1]['conditions'])
        self.assertEqual(rows[0]['memory_difference_rms'], 1.)
        self.assertFalse(model.training)

    def test_cli_preflight_and_mocked_dispatch_preserve_source_and_only_pass_val_to_audit(self):
        helper = test_runtime.RuntimeTests()
        helper.setUp()
        for check in (True, False):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, config = helper.config_file(root)
                source, output = root / 'source', root / 'output'
                source.mkdir()
                write_json(source / 'config.json', config)
                (source / 'best.pt').write_bytes(b'checkpoint fixture, loader mocked')
                before = {p.name: p.read_bytes() for p in source.iterdir()}
                cases = [{'reference_risk': 'very healthy'}, {'reference_risk': 'high risk'}]
                sentinel_val = object()
                class Reader:
                    def close(self):
                        pass
                splits = {'TRAIN': Reader(), 'VAL': Reader(), 'TEST': Reader()}
                sentinel_val = splits['VAL']
                checkpoint = {'epoch': 5, 'model': torch.nn.Linear(1, 1).state_dict()}
                args = ['diagnose_risk.py', '--run', str(source), '--output', str(output), '--device', 'cpu']
                if check:
                    args.append('--check')
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(audit, 'load_dataset', return_value=(self.words, {}, splits, {})), \
                        patch.object(audit, 'load_checkpoint', return_value=checkpoint), \
                        patch.object(audit, 'reference_cases', return_value=cases), \
                        patch.object(audit, 'build_model', return_value=torch.nn.Linear(1, 1)) as build, \
                        patch.object(audit, 'diagnose', return_value=[]) as diagnose, \
                        patch('torch.optim.Adam', side_effect=AssertionError('No training')):
                    audit.main()
                self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
                if check:
                    build.assert_not_called()
                    diagnose.assert_not_called()
                    self.assertFalse(output.exists())
                else:
                    build.assert_called_once()
                    self.assertIs(diagnose.call_args.args[1], sentinel_val)
                    self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
                    self.assertEqual(read_json(output / 'summary.json')['split'], 'VAL')
                    with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                            patch.object(audit, 'load_dataset') as load:
                        with self.assertRaises(SystemExit):
                            audit.main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
