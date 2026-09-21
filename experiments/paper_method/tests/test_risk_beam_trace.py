"""Controlled token trees and mocked runtime only; no real-image inference."""

import copy
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import trace_risk_beam as trace
from paper_method.data import fingerprint, read_json, write_json
from paper_method.decoding import beam_search
from paper_method.diagnose_risk import score_prefix
from paper_method.model import ParallelDecoder
from paper_method.runner import PACKAGE


class ScriptedDecoder(torch.nn.Module):
    def __init__(self, words, transitions):
        super().__init__()
        self.words, self.transitions = words, transitions

    def initial_state(self, memory):
        return ()

    def step(self, memory, word, state):
        path = state + (word.item(),)
        logits = memory.new_full((1, len(self.words)), -40.)
        for token, probability in self.transitions.get(path, {'<end>': 1.}).items():
            logits[0, self.words[token]] = math.log(probability)
        return logits, logits.clone(), path


class BeamTraceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        vocabulary = ('<pad>', '<unk>', '<start>', '<end>', 'glaucoma', 'risk', 'assessment', ':',
                      'very', 'healthy', 'high', '.', 'moderate', 'confidence', 'level', '0.95', '0.9', 'x', 'y')
        self.words = {word: index for index, word in enumerate(vocabulary)}
        self.prefix = self.encode('<start> glaucoma risk assessment :')
        self.memory = torch.zeros(1, 3, 8)

    def encode(self, text):
        return [self.words[word] for word in text.split()]

    def fixture(self, pruning=None):
        table = {}
        for position in range(1, len(self.prefix)):
            reverse = {value: key for key, value in self.words.items()}
            table[tuple(self.prefix[:position])] = {reverse[self.prefix[position]]: 1.}
        start = {'very': .6, 'high': .4} if pruning != 'global' else {'very': .55, 'high': .45}
        table[tuple(self.prefix)] = start
        healthy_next = {'healthy': 1.} if pruning is None else \
                       {'healthy': .2, 'x': .45, 'y': .35} if pruning == 'local' else {'healthy': .4, 'x': .6}
        table[tuple(self.prefix + self.encode('very'))] = healthy_next
        table[tuple(self.prefix + self.encode('very healthy'))] = {'.': 1.}
        table[tuple(self.prefix + self.encode('very x'))] = {'x': .2, 'y': .2, '0.9': .2, '0.95': .2, 'moderate': .2}
        table[tuple(self.prefix + self.encode('high'))] = {'risk': 1.}
        table[tuple(self.prefix + self.encode('high risk'))] = {'.': 1.} if pruning is None else \
                                                              {'.': .21, 'x': .1975, 'y': .1975, '0.95': .1975, 'moderate': .1975}
        for label in ('very healthy', 'high risk'):
            path = self.prefix + self.encode(label + ' .')
            for word in ('confidence', 'level', ':'):
                table[tuple(path)] = {word: 1.}
                path.append(self.words[word])
            number = '0.95' if label == 'very healthy' else '0.9'
            table[tuple(path)] = {number: .4, 'x': .3, 'y': .3} if label == 'very healthy' and pruning is None else {number: 1.}
            path.append(self.words[number])
            table[tuple(path)] = {'.': 1.}
            path.append(self.words['.'])
            table[tuple(path)] = {'<end>': 1.}
        model = torch.nn.Module()
        model.encoder = torch.nn.Identity()
        model.decoder = ScriptedDecoder(self.words, table)
        model.eval()
        config = {'lambda_parallel': .5, 'beam_size': 5 if pruning is None else 2, 'max_decode_steps': 25}
        generated = beam_search(model.decoder, self.memory, self.words, .5, config['beam_size'], 25)
        self.assertIn('assessment : high risk .', generated['prediction'])
        profile = score_prefix(model.decoder, self.memory, self.prefix, self.words, .5)
        self.assertEqual(profile['mixture']['restricted_sequence_choice'], 'very healthy')
        case = {'id': 'val:0', 'reference_risk': 'very healthy', 'generated_prefix_ids': self.prefix,
                'generation': generated, 'conditions': {'own_image_generated_prefix': profile}}
        return model, config, case

    def test_observer_is_identical_to_original_search_including_ties_caps_and_eos(self):
        decoders = [ParallelDecoder(len(self.words), 8, 2, self.words['<pad>']).eval(),
                    ScriptedDecoder(self.words, {}).eval()]
        for decoder in decoders:
            for width in (1, 2, 5):
                for cap in (1, 8):
                    with self.subTest(decoder=type(decoder).__name__, width=width, cap=cap):
                        original = beam_search(decoder, self.memory, self.words, .5, width, cap)
                        prefix = [self.words['<start>']]
                        healthy = prefix + self.encode('very healthy .')
                        observer = trace.ObservedDecoder(decoder, self.words, .5, width, healthy)
                        result = beam_search(observer, self.memory, self.words, .5, width, cap)
                        self.assertEqual(original, result)
                        trace.reconstruct_search(observer.events, result, self.words, width, cap, prefix, healthy)
                        self.assertEqual(result, beam_search(observer, self.memory, self.words, .5, width, cap))
                        broken = copy.deepcopy(observer.events)
                        broken[0]['parent_token_ids'] = []
                        with self.assertRaises(ValueError):
                            trace.reconstruct_search(broken, result, self.words, width, cap, prefix, healthy)

    def test_continuation_bos_is_virtual_and_does_not_add_a_second_start_token(self):
        decoder = ParallelDecoder(len(self.words), 8, 2, self.words['<pad>']).eval()
        adapter = trace.PrefixContinuation(decoder, [self.words['<start>']], self.words['<start>'])
        self.assertEqual(beam_search(decoder, self.memory, self.words, width=3, max_steps=8),
                         beam_search(adapter, self.memory, self.words, width=3, max_steps=8))
        fixed = self.prefix + self.encode('very healthy .')
        adapter = trace.PrefixContinuation(decoder, fixed, self.words['<start>'])
        with torch.no_grad():
            initial = adapter.initial_state(self.memory)
            a, b, _ = adapter.step(self.memory, torch.tensor([self.words['<start>']]), initial)
            state = decoder.initial_state(self.memory)
            for token in fixed:
                expected_a, expected_b, state = decoder.step(self.memory, torch.tensor([token]), state)
        torch.testing.assert_close(a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(b, expected_b, rtol=0, atol=0)

    def test_healthy_path_can_survive_but_lose_at_confidence_value(self):
        model, config, case = self.fixture()
        result = trace.trace_case(model, self.memory, case, self.words, config)
        self.assertEqual(result['diagnostic_outcome'], 'explored_completed_healthy_path_scores_lower_or_equal')
        self.assertTrue(result['original_search']['healthy_path_survived_to_final_beam'])
        self.assertIsNone(result['original_search']['first_healthy_path_elimination'])
        reversal = result['comparison']['first_score_reversal_on_explored_continuation']
        self.assertEqual(reversal['healthy_token'], '0.95')
        self.assertEqual(reversal['original_token'], '0.9')
        self.assertEqual(reversal['healthy_stage'], 'after_confidence_marker')
        self.assertAlmostEqual(result['comparison']['final_healthy_minus_original_log_probability'], math.log(.6), places=5)

    def test_pruned_healthy_path_can_have_higher_complete_score_local_and_global(self):
        for pruning, reason in (('local', 'outside_parent_topk'), ('global', 'global_beam_pruning')):
            with self.subTest(pruning=pruning):
                model, config, case = self.fixture(pruning)
                result = trace.trace_case(model, self.memory, case, self.words, config)
                removal = result['original_search']['first_healthy_path_elimination']
                self.assertEqual(removal['required_token'], 'healthy')
                self.assertEqual(removal['reason'], reason)
                self.assertEqual(removal['stage'], 'risk_phrase')
                self.assertEqual(result['diagnostic_outcome'], 'completed_valid_healthy_path_scores_higher')
                self.assertGreater(result['comparison']['final_healthy_minus_original_log_probability'], 0)
                self.assertIsNone(result['comparison']['first_score_reversal_on_explored_continuation'])
                self.assertTrue(result['constrained_healthy_continuation']['terminated_with_end'])

    def test_trace_refuses_changed_generation_or_changed_recorded_risk_score(self):
        model, config, case = self.fixture()
        changed = copy.deepcopy(case)
        changed['generation']['prediction'] = 'wrong saved text'
        with self.assertRaisesRegex(ValueError, 'no longer reproduces'):
            trace.trace_case(model, self.memory, changed, self.words, config)
        changed = copy.deepcopy(case)
        changed['conditions']['own_image_generated_prefix']['mixture']['sequence_log_probabilities']['very healthy'] += 1
        with self.assertRaisesRegex(ValueError, 'Score replay mismatch'):
            trace.trace_case(model, self.memory, changed, self.words, config)

    def test_selection_and_total_decode_budget(self):
        model, config, case = self.fixture()
        self.assertEqual(trace.select_cases([case], self.words), [case])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            trace.select_cases([case, case], self.words)
        changed = copy.deepcopy(case)
        changed['conditions']['own_image_generated_prefix']['mixture']['restricted_sequence_choice'] = 'high risk'
        with self.assertRaisesRegex(ValueError, 'No local-healthy'):
            trace.select_cases([changed], self.words)
        result = trace.trace_case(model, self.memory, case, self.words, config)
        alternative = result['constrained_healthy_continuation']
        self.assertEqual(alternative['remaining_decode_steps'], config['max_decode_steps'] - (len(self.prefix) + 3 - 1))
        self.assertLessEqual(len(alternative['token_ids']) - 1, config['max_decode_steps'])
        self.assertEqual(trace.summarize([result])['samples'], 1)

    def test_mocked_cli_and_preflight_use_only_val_and_leave_source_artifacts_unchanged(self):
        model, settings, case = self.fixture()
        for check in (True, False):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, audit, output = root / 'source', root / 'audit', root / 'output'
                source.mkdir()
                audit.mkdir()
                config = {**read_json(PACKAGE / 'config.json'), **settings, 'amp': False, 'data_directory': str(root / 'data')}
                write_json(source / 'config.json', config)
                (source / 'best.pt').write_bytes(b'fixture checkpoint, loader mocked')
                write_json(audit / 'manifest.json', {'source_run': str(source), 'split': 'VAL',
                           'diagnostic': 'paper_core_risk_conditioning_v1', 'checkpoint_sha256': fingerprint(source / 'best.pt')})
                write_json(audit / 'status.json', {'stage': 'complete'})
                write_json(audit / 'config.json', config)
                write_json(audit / 'data_fingerprints.json', {})
                write_json(audit / 'cases.json', [case])
                before = {str(path): path.read_bytes() for directory in (source, audit) for path in directory.iterdir()}
                class Reader:
                    def __init__(self, split):
                        self.split, self.calls, self.closed = split, 0, False
                        self.records = [{'id': 'val:0'}] if split == 'VAL' else []

                    def __getitem__(self, index):
                        if self.split != 'VAL':
                            raise AssertionError('Only VAL images may be read')
                        self.calls += 1
                        return {'id': 'val:0', 'image': torch.zeros(3, 8)}

                    def close(self):
                        self.closed = True
                splits = {key: Reader(key) for key in ('TRAIN', 'VAL', 'TEST')}
                args = ['trace_risk_beam.py', '--audit', str(audit), '--output', str(output), '--device', 'cpu']
                if check:
                    args.append('--check')
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(trace, 'load_dataset', return_value=(self.words, {}, splits, {})), \
                        patch.object(trace, 'load_checkpoint', return_value={'model': {}}), \
                        patch.object(trace, 'build_model', return_value=model) as build, \
                        patch('torch.optim.Adam', side_effect=AssertionError('No optimizer')):
                    trace.main()
                self.assertTrue(all(reader.closed for reader in splits.values()))
                self.assertEqual(before, {str(path): path.read_bytes() for directory in (source, audit) for path in directory.iterdir()})
                if check:
                    build.assert_not_called()
                    self.assertEqual(splits['VAL'].calls, 0)
                    self.assertFalse(output.exists())
                else:
                    self.assertEqual(splits['VAL'].calls, 1)
                    self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
                    self.assertEqual(read_json(output / 'summary.json')['samples'], 1)
                    with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                            patch.object(trace, 'load_dataset') as load:
                        with self.assertRaises(SystemExit):
                            trace.main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
