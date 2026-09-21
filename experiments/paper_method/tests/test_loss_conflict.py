"""Synthetic local-gradient probes; no real images, checkpoint inference or training."""

import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import diagnose_loss_conflict as audit
from paper_method.data import read_json, write_json
from paper_method.diagnose_endings import ending_record
from paper_method.losses import objective
from paper_method.model import DASPL, mixture_log_probs
from paper_method.runner import PACKAGE
import test_risk_diagnostic


class LossConflictTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        helper = test_risk_diagnostic.RiskDiagnosticTests()
        helper.setUp()
        self.words, self.tokens = helper.words, helper.tokens
        short = [self.words[word] for word in
                 '<start> glaucoma risk assessment : high risk . confidence level : 0.9 . <end>'.split()]
        self.lengths = torch.tensor([len(self.tokens), len(short)])
        self.captions = torch.tensor([self.tokens, short + [self.words['<pad>']] * (len(self.tokens) - len(short))])
        self.tags = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        self.model = DASPL(torch.nn.Identity(), len(self.words), 3, dim=8, heads=2,
                           pad_id=self.words['<pad>']).eval()
        self.first = torch.randn(2, len(self.tokens) - 1, len(self.words))
        self.second = torch.randn_like(self.first)

    def losses(self, first, second):
        embeddings = mixture_log_probs(first, second, .5).exp() @ self.model.decoder.embedding.weight
        tag_logits = self.model.label_enhancement(embeddings, self.lengths - 1)
        total, _, _ = objective({'primary_logits': first, 'secondary_logits': second, 'tag_logits': tag_logits},
                                self.captions, self.lengths, self.tags)
        return {'total': total, 'lem': 5 * F.binary_cross_entropy_with_logits(tag_logits, self.tags)}

    def test_ce_gradients_match_weighted_token_mean_and_exclude_padding(self):
        gradients, values = audit.loss_gradients(self.model, self.first, self.second,
                                                 self.captions, self.lengths, self.tags, .5, 5.)
        mask = torch.arange(self.first.size(1))[None] < (self.lengths - 1)[:, None]
        targets = self.captions[:, 1:]
        self.assertEqual(values['supervised_tokens'], mask.sum().item())
        for head, scores, weight in zip(audit.HEADS, (self.first, self.second), (1., .5)):
            expected = (scores.softmax(-1) - F.one_hot(targets, len(self.words))) * weight / mask.sum()
            expected[~mask] = 0
            torch.testing.assert_close(gradients[head]['ce'], expected, atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(gradients[head]['ce'] + gradients[head]['lem'], gradients[head]['total'])
            for name in ('ce', 'lem', 'total'):
                torch.testing.assert_close(gradients[head][name][~mask], torch.zeros_like(expected[~mask]))
            for i, length in enumerate(self.lengths.tolist()):
                self.assertLess(gradients[head]['ce'][i, length - 2, self.words['<end>']].item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertAlmostEqual(values['total'], self.losses(self.first, self.second)['total'].item(), places=6)

    def test_margin_gradients_match_finite_differences_with_other_logits_fixed(self):
        gradients, _ = audit.loss_gradients(self.model, self.first, self.second,
                                            self.captions, self.lengths, self.tags, .5, 5.)
        position, target, colon = len(self.tokens) - 2, self.words['<end>'], self.words[':']
        epsilon = .02
        for branch, head in enumerate(audit.HEADS):
            plus, minus = [self.first.clone(), self.second.clone()], [self.first.clone(), self.second.clone()]
            plus[branch][0, position, target] += epsilon
            plus[branch][0, position, colon] -= epsilon
            minus[branch][0, position, target] -= epsilon
            minus[branch][0, position, colon] += epsilon
            a, b = self.losses(*plus), self.losses(*minus)
            for name in ('lem', 'total'):
                numeric = (a[name].item() - b[name].item()) / (2 * epsilon)
                exact = (gradients[head][name][0, position, target] - gradients[head][name][0, position, colon]).item()
                self.assertAlmostEqual(numeric, exact, delta=5e-5)

    def test_zero_lem_weight_leaves_only_ce(self):
        gradients, values = audit.loss_gradients(self.model, self.first, self.second,
                                                 self.captions, self.lengths, self.tags, .5, 0.)
        self.assertEqual(values['weighted_lem'], 0.)
        for head in audit.HEADS:
            self.assertEqual(gradients[head]['lem'].abs().sum().item(), 0.)
            torch.testing.assert_close(gradients[head]['ce'], gradients[head]['total'])

    def test_opposition_means_descent_reduces_target_over_colon_margin(self):
        ce, lem = torch.tensor([-.4, .4]), torch.tensor([.8, -.8])
        stats = audit.site_statistics(torch.zeros(2), {'ce': ce, 'lem': lem, 'total': ce + lem}, 0, 1)
        self.assertTrue(stats['lem_opposes_ce_margin'])
        self.assertTrue(stats['total_opposes_ce_margin'])
        self.assertAlmostEqual(stats['ce_lem_gradient_cosine'], -1, places=6)
        weak = audit.site_statistics(torch.zeros(2), {'ce': ce, 'lem': lem / 4, 'total': ce + lem / 4}, 0, 1)
        self.assertTrue(weak['lem_opposes_ce_margin'])
        self.assertFalse(weak['total_opposes_ce_margin'])
        rows = [{'reference_risk': 'very healthy', 'reference_confidence': '0.95',
                 'sites': {site: {head: stats for head in audit.HEADS} for site in audit.SITES}}]
        self.assertEqual(audit.summarize(rows)['all']['sites']['eos']['primary']['total_opposes_ce_margin_count'], 1)

    def test_token_frequency_excludes_start_pad_and_decimal_numbers_from_punctuation(self):
        record = ending_record('train:0', self.tokens, len(self.tokens), self.words)
        counts = audit.token_audit([record], self.words)
        self.assertEqual(counts['supervised_tokens'], len(self.tokens) - 1)
        self.assertEqual(counts['token_counts']['<end>'], 1)
        self.assertNotIn('<start>', counts['token_counts'])
        self.assertNotIn('0.95', counts['punctuation_counts'])
        self.assertNotIn('<end>', counts['punctuation_counts'])

    def test_cli_preflight_and_synthetic_pipeline_preserve_inputs_and_never_read_val_test(self):
        for check in (True, False):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / 'source', root / 'output'
                source.mkdir()
                config = {**read_json(PACKAGE / 'config.json'), 'amp': False, 'batch_size': 2,
                          'data_directory': str(root / 'data')}
                write_json(source / 'config.json', config)
                (source / 'best.pt').write_bytes(b'fixture; checkpoint loader is mocked')
                before = {path.name: path.read_bytes() for path in source.iterdir()}
                state = {key: value.clone() for key, value in self.model.state_dict().items()}
                tokens, words = self.tokens, self.words

                class Reader:
                    def __init__(self, name):
                        self.name, self.calls, self.closed = name, 0, False
                        self.records = [{'id': f'{name.lower()}:0'}]
                        self.captions, self.lengths = [tokens], [len(tokens)]

                    def __getitem__(self, index):
                        if self.name != 'TRAIN':
                            raise AssertionError('Only TRAIN images are allowed')
                        self.calls += 1
                        return {'id': self.records[index]['id'], 'image': torch.zeros(3, 8),
                                'caption': torch.tensor(tokens), 'length': torch.tensor(len(tokens)),
                                'tags': torch.tensor([1., 0., 1.])}

                    def close(self):
                        self.closed = True

                splits = {key: Reader(key) for key in ('TRAIN', 'VAL', 'TEST')}
                args = ['diagnose_loss_conflict.py', '--run', str(source), '--output', str(output), '--device', 'cpu']
                if check:
                    args.append('--check')
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(audit, 'load_dataset', return_value=(words, {}, splits, {})), \
                        patch.object(audit, 'load_checkpoint', return_value={'epoch': 5, 'model': state}), \
                        patch.object(audit, 'build_model', return_value=self.model) as build, \
                        patch('torch.optim.Adam', side_effect=AssertionError('No optimizer')):
                    audit.main()
                self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
                for key, value in self.model.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                self.assertTrue(all(split.closed for split in splits.values()))
                self.assertEqual(splits['VAL'].calls + splits['TEST'].calls, 0)
                if check:
                    build.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertEqual(splits['TRAIN'].calls, 0)
                else:
                    self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
                    self.assertEqual(read_json(output / 'summary.json')['groups']['all']['samples'], 1)
                    self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
                    with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                            patch.object(audit, 'load_dataset') as load:
                        with self.assertRaises(SystemExit):
                            audit.main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
