"""Synthetic decoding-policy comparison tests; no real-image inference or training."""

import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import compare_decoding as audit
from paper_method.ablate_lem import FORMAT
from paper_method.data import read_json, write_json
from paper_method.diagnose_fields import report_records
from paper_method.evaluation import FIELDS
from paper_method.model import DASPL, mixture_log_probs
from paper_method.runner import archive_sources, PACKAGE


class DecodingComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.texts = [self.report(), self.report(size='large', ratio='0.8', color='pale', risk='high risk')]
        vocabulary = ['<pad>', '<unk>', '<start>', '<end>']
        vocabulary += ' '.join(self.texts).split() + ['0.7']
        self.words = {word: i for i, word in enumerate(dict.fromkeys(vocabulary))}
        self.config = {**read_json(PACKAGE / 'config.json'), 'amp': False, 'tag_loss_weight': 0.}
        texts, words = self.texts, self.words

        class Reader:
            def __init__(self, name):
                self.name, self.calls, self.closed = name, 0, False
                self.records = [{'id': f'{name.lower()}:{i}'} for i in range(len(texts))]
                self.captions = [[words[w] for w in ('<start> ' + text + ' <end>').split()] for text in texts]
                self.lengths = [len(tokens) for tokens in self.captions]

            def __getitem__(self, index):
                if self.name != 'VAL':
                    raise AssertionError('No TRAIN or TEST image access')
                self.calls += 1
                return {'id': self.records[index]['id'], 'image': torch.full((3, 8), float(index))}

            def close(self):
                self.closed = True

        self.splits = {name: Reader(name) for name in ('TRAIN', 'VAL', 'TEST')}
        self.records = report_records(self.splits['VAL'], self.words)
        self.model = DASPL(torch.nn.Identity(), len(words), 3, dim=8, heads=2, pad_id=words['<pad>']).eval()
        self.baseline = [self.prediction(i, text) for i, text in enumerate(self.texts)]
        for i, row in enumerate(self.baseline):
            tokens = torch.tensor([row['token_ids']])
            with torch.no_grad():
                a, b = self.model.decoder(torch.full((1, 3, 8), float(i)), tokens, torch.tensor([tokens.size(1)]))
                scores = mixture_log_probs(a, b, .5)
                row['log_probability'] = scores[0].gather(1, tokens[0, 1:, None]).double().sum().item()

    def report(self, size='normal', ratio='0.4', color='pink', risk='very healthy'):
        values = {'optic disc size': size, 'cup to disc ratio': ratio, 'rim color': color,
                  'glaucoma risk assessment': risk, 'confidence level': '0.95'}
        return ' '.join(f'{field} : {values.get(field, "false")} .' for field in FIELDS)

    def prediction(self, index, text):
        return {'id': f'val:{index}', 'reference': self.texts[index], 'prediction': text,
                'token_ids': [self.words[w] for w in ('<start> ' + text + ' <end>').split()],
                'terminated_with_end': True, 'log_probability': -2.}

    def test_comparison_reports_fixes_regressions_invalid_risk_and_reference_agreement(self):
        baseline = [self.prediction(0, self.report(size='large')), copy.deepcopy(self.baseline[1])]
        greedy = [self.prediction(0, self.report(ratio='0.8')),
                  self.prediction(1, self.report(size='large', ratio='0.8', color='pale', risk='high risk high risk'))]
        before = copy.deepcopy((baseline, greedy))
        summary, pairs, _ = audit.compare_predictions(baseline, greedy, self.records)
        self.assertEqual(summary['field_transitions']['optic disc size']['greedy_fixes'], 1)
        self.assertEqual(summary['field_transitions']['cup to disc ratio']['greedy_regresses'], 1)
        self.assertEqual(summary['greedy_minus_beam5']['invalid_risk'], 1)
        self.assertEqual(summary['greedy_minus_beam5']['risk_matches'], -1)
        self.assertEqual(summary['greedy_minus_beam5']['field_matches'], -1)
        self.assertEqual(pairs[0]['fields']['optic disc size']['greedy'], 'normal')
        self.assertEqual((baseline, greedy), before)

    def test_comparison_rejects_id_and_reference_mismatches(self):
        with self.assertRaisesRegex(ValueError, 'aligned VAL IDs'):
            audit.compare_predictions(self.baseline, self.baseline[::-1], self.records)
        changed = copy.deepcopy(self.baseline)
        changed[0]['reference'] += ' .'
        with self.assertRaisesRegex(ValueError, 'reference text mismatch'):
            audit.compare_predictions(self.baseline, changed, self.records)

    def test_generation_changes_only_width_and_never_supplies_reference_text_to_decoder(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        generated = [{key: value for key, value in row.items() if key not in ('id', 'reference')}
                     for row in self.baseline]
        with patch.object(audit, 'beam_search', side_effect=generated) as search, \
                patch.object(audit, 'replay_prediction', wraps=audit.replay_prediction) as replay, \
                patch('sys.stdout', new_callable=io.StringIO):
            rows, replays = audit.generate_greedy(self.model, self.splits['VAL'], self.records, self.baseline,
                                                 self.words, self.config, torch.device('cpu'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(replays), 2)
        for i, call in enumerate(search.call_args_list):
            self.assertEqual(len(call.args), 3)
            self.assertIs(call.args[0], self.model.decoder)
            self.assertIs(call.args[1], replay.call_args_list[i].args[1])
            self.assertEqual(call.args[2], self.words)
            self.assertEqual(call.kwargs, {'coefficient': .5, 'width': 1, 'max_steps': 161})
            torch.testing.assert_close(call.args[1], torch.full((1, 3, 8), float(i)))
        self.assertEqual(self.splits['TRAIN'].calls + self.splits['TEST'].calls, 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_wrong_baseline_score_stops_before_new_generation(self):
        baseline = copy.deepcopy(self.baseline)
        baseline[0]['log_probability'] += 1
        with patch.object(audit, 'beam_search') as search:
            with self.assertRaisesRegex(ValueError, 'replay mismatch'):
                audit.generate_greedy(self.model, self.splits['VAL'], self.records, baseline,
                                      self.words, self.config, torch.device('cpu'))
            search.assert_not_called()

    def test_existing_width_one_decoder_is_greedy_without_forced_eos_or_repair(self):
        words = {'<pad>': 0, '<start>': 1, '<end>': 2, 'a': 3, 'b': 4}

        class Decoder(torch.nn.Module):
            def initial_state(self, memory):
                return ()

            def step(self, memory, previous, state):
                # A and B have the highest legal probabilities, with A always winning.
                scores = torch.tensor([[-20., -20., -5., 3., 2.]])
                return scores, scores, ()

        result = audit.beam_search(Decoder().eval(), torch.zeros(1, 3, 8), words, width=1, max_steps=3)
        self.assertEqual(result['token_ids'], [1, 3, 3, 3])
        self.assertEqual(result['prediction'], 'a a a')
        self.assertFalse(result['terminated_with_end'])

    def source_fixture(self, root):
        parent = root / 'paired'
        source = parent / 'lem_off'
        source.mkdir(parents=True)
        weights = root / 'weights.pth'
        weights.write_bytes(b'fixture; model builder mocked')
        config = {**self.config, 'pretrained_weights': str(weights), 'data_directory': str(root / 'data')}
        write_json(source / 'config.json', config)
        write_json(parent / 'config.json', {**config, 'tag_loss_weight': 5.})
        write_json(source / 'manifest.json', {'format': FORMAT, 'arm': 'lem_off', 'tag_loss_weight': 0.,
                   'initial_state_sha256': 'initial'})
        write_json(parent / 'manifest.json', {'experiment': FORMAT})
        write_json(source / 'status.json', {'stage': 'complete', 'completed_epochs': 5})
        write_json(parent / 'status.json', {'stage': 'complete', 'epochs_per_arm': 5})
        for folder in (source, parent):
            write_json(folder / 'data_fingerprints.json', {})
        archive_sources(parent)
        checkpoint = {'format': FORMAT, 'arm': 'lem_off', 'epoch': 5, 'config': config, 'words': self.words,
                      'tags': {}, 'data_fingerprints': {}, 'initial_state_sha256': 'initial', 'model': self.model.state_dict()}
        torch.save(checkpoint, source / 'last.pt')
        epoch = source / 'epoch_005'
        epoch.mkdir()
        write_json(epoch / 'predictions.json', self.baseline)
        return source

    def test_cli_preflight_and_mocked_generation_preserve_inputs_and_protect_output(self):
        for check in (True, False):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.source_fixture(root)
                output = root / 'output'
                before = {str(path): path.read_bytes() for path in source.parent.rglob('*') if path.is_file()}
                args = ['compare_decoding.py', '--run', str(source), '--output', str(output), '--device', 'cpu']
                if check:
                    args.append('--check')
                for split in self.splits.values():
                    split.calls, split.closed = 0, False
                generated = [{key: value for key, value in row.items() if key not in ('id', 'reference')}
                             for row in self.baseline]
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(audit, 'load_dataset', return_value=(self.words, {}, self.splits, {})), \
                        patch.object(audit, 'build_model', return_value=self.model) as build, \
                        patch.object(audit, 'beam_search', side_effect=generated) as search, \
                        patch('torch.optim.Adam', side_effect=AssertionError('No training')):
                    audit.main()
                self.assertEqual(before, {str(path): path.read_bytes() for path in source.parent.rglob('*') if path.is_file()})
                self.assertTrue(all(split.closed for split in self.splits.values()))
                self.assertEqual(self.splits['TRAIN'].calls + self.splits['TEST'].calls, 0)
                if check:
                    build.assert_not_called()
                    search.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertEqual(self.splits['VAL'].calls, 0)
                else:
                    self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
                    self.assertEqual(read_json(output / 'summary.json')['changed_reports'], 0)
                    self.assertEqual(read_json(output / 'beam5_predictions.json'), self.baseline)
                    with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                            patch.object(audit, 'load_dataset') as load:
                        with self.assertRaises(SystemExit):
                            audit.main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
