"""Synthetic ending probes and mocked CLI; no real-image inference or optimizer."""

import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import diagnose_endings as audit
from paper_method.data import fingerprint, read_json, write_json
from paper_method.losses import objective
from paper_method.model import ParallelDecoder, mixture_log_probs
from paper_method.runner import PACKAGE
from paper_method.trace_risk_beam import score_path
import test_risk_diagnostic


class EndingDiagnosticTests(unittest.TestCase):
    def setUp(self):
        helper = test_risk_diagnostic.RiskDiagnosticTests()
        helper.setUp()
        self.words, self.tokens = helper.words, helper.tokens
        self.record = audit.ending_record('val:0', self.tokens, len(self.tokens), self.words)

    def encode(self, text):
        return [self.words[word] for word in text.split()]

    def model(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Identity()
        model.decoder = ParallelDecoder(len(self.words), 8, 2, self.words['<pad>'])
        return model.eval()

    def focus_case(self, model, memory):
        tokens = self.encode('<start> rim color : pale . glaucoma risk assessment : very healthy . confidence level : 0.95 . <end>')
        record = audit.ending_record('val:0', tokens, len(tokens), self.words)
        scores = score_path(model.decoder, memory, tokens, self.words, .5, len(record['risk_prefix_ids']), 3)
        return {'id': 'val:0', 'generated_prefix_ids': record['risk_prefix_ids'],
                'constrained_healthy_continuation': {'token_ids': tokens, 'terminated_with_end': True, 'scores': scores}}

    def test_data_audit_counts_only_confidence_suffix_and_rejects_bad_targets(self):
        second = self.encode('<start> rim color : 0.9 . glaucoma risk assessment : high risk . confidence level : 0.9 . <end>')
        records = [self.record, audit.ending_record('val:1', second, len(second), self.words)]
        stats = audit.data_audit(records, self.words)
        self.assertEqual(stats['risk_confidence_counts']['very healthy'], {'0.95': 1})
        self.assertEqual(stats['terminal_suffix_counts'], {'0.95 . <end>': 1, '0.9 . <end>': 1})
        self.assertEqual(stats['eos_target_count'], 2)
        self.assertEqual(stats['supervised_token_count'], sum(len(row['tokens']) - 1 for row in records))
        for text in (
                '<start> glaucoma risk assessment : very healthy . confidence level : 0.95 <end>',
                '<start> glaucoma risk assessment : very healthy . confidence level : 0.95 : 0.95 . <end>',
                '<start> glaucoma risk assessment : very healthy . confidence level : 0.95 . <end> <end>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokens = self.encode(text)
                audit.ending_record('bad', tokens, len(tokens), self.words)

    def test_site_logits_use_previous_position_and_active_batch_members(self):
        model = self.model()
        short = self.encode('<start> glaucoma risk assessment : high risk . confidence level : 0.9 . <end>')
        other = audit.ending_record('val:1', short, len(short), self.words)
        records = [self.record, other]
        memories = [torch.randn(1, 3, 8), torch.randn(1, 3, 8)]
        scored = audit.score_batch(model.decoder, memories, records, self.words, .5, torch.device('cpu'))
        captions = torch.tensor([self.tokens, short + [self.words['<pad>']] * (len(self.tokens) - len(short))])
        with torch.no_grad():
            first, second = model.decoder(torch.cat(memories), captions, torch.tensor([len(self.tokens), len(short)]))
        distributions = {'primary': first.log_softmax(-1), 'secondary': second.log_softmax(-1),
                         'mixture': mixture_log_probs(first, second, .5)}
        for i, row in enumerate(records):
            for site, position in row['sites'].items():
                for head, distribution in distributions.items():
                    self.assertAlmostEqual(scored[i]['sites'][site][head]['target_log_probability'],
                                           distribution[i, position - 1, row['tokens'][position]].item(), places=6)
        self.assertEqual(scored[0]['active_ids_at_site']['eos'], ['val:0'])
        self.assertEqual(scored[1]['active_ids_at_site']['eos'], ['val:0', 'val:1'])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with self.assertRaises(ValueError):
            audit.score_batch(model.decoder.train(), memories, records, self.words, .5, torch.device('cpu'))

    def test_real_objective_supervises_eos_in_both_heads_and_excludes_padding(self):
        short = self.encode('<start> glaucoma risk assessment : high risk . confidence level : 0.9 . <end>')
        captions = torch.tensor([self.tokens, short + [self.words['<pad>']] * (len(self.tokens) - len(short))])
        lengths = torch.tensor([len(self.tokens), len(short)])
        first = torch.zeros(2, len(self.tokens) - 1, len(self.words), requires_grad=True)
        second = torch.zeros_like(first, requires_grad=True)
        loss, _, count = objective({'primary_logits': first, 'secondary_logits': second, 'tag_logits': torch.zeros(2, 1)},
                                   captions, lengths, torch.zeros(2, 1))
        loss.backward()
        self.assertEqual(count, sum(lengths).item() - 2)
        for i, length in enumerate(lengths.tolist()):
            expected = (1 / len(self.words) - 1) / count
            self.assertAlmostEqual(first.grad[i, length - 2, self.words['<end>']].item(), expected, places=7)
            self.assertAlmostEqual(second.grad[i, length - 2, self.words['<end>']].item(), .5 * expected, places=7)
            self.assertEqual(first.grad[i, length - 1:].abs().sum().item(), 0.)
            self.assertEqual(second.grad[i, length - 1:].abs().sum().item(), 0.)

    def test_controlled_prefix_swap_holds_entire_suffix_fixed_and_checks_old_scores(self):
        model, memory = self.model(), torch.randn(1, 3, 8)
        trace = self.focus_case(model, memory)
        reference_tokens = self.encode('<start> rim color : pink . glaucoma risk assessment : healthy . confidence level : 0.9 . <end>')
        reference = audit.ending_record('val:0', reference_tokens, len(reference_tokens), self.words)
        generated, controlled = audit.controlled_pair(reference, trace, self.words)
        self.assertEqual(generated['tokens'][len(generated['risk_prefix_ids']):],
                         controlled['tokens'][len(controlled['risk_prefix_ids']):])
        self.assertEqual(controlled['confidence'], '0.95')
        self.assertEqual(controlled['risk'], 'very healthy')
        self.assertEqual(controlled['risk_prefix_ids'], reference['risk_prefix_ids'])
        scored = audit.score_batch(model.decoder, [memory], [generated], self.words, .5, torch.device('cpu'))[0]
        audit.verify_trace_replay(scored, generated, trace)
        changed = copy.deepcopy(scored)
        changed['sites']['eos']['primary']['target_log_probability'] += .1
        with self.assertRaisesRegex(ValueError, 'replay mismatch'):
            audit.verify_trace_replay(changed, generated, trace)

    def test_orders_summaries_and_equal_target_guard(self):
        orders = audit.batch_orders(7, 123)
        self.assertEqual(orders, audit.batch_orders(7, 123))
        self.assertEqual(sorted(orders['reference_shuffled_batch']), orders['reference_native_batch'])
        scored = audit.score_batch(self.model().decoder, [torch.zeros(1, 3, 8)], [self.record],
                                   self.words, .5, torch.device('cpu'))[0]
        row = {'id': 'val:0', 'reference_risk': 'very healthy', 'reference_confidence': '0.95',
               'conditions': {'reference_single': scored, 'reference_native_batch': copy.deepcopy(scored),
                              'reference_shuffled_batch': copy.deepcopy(scored)}}
        result = audit.summarize([row])
        self.assertEqual(result['contrasts']['batch_to_single']['all']['paired_samples'], 1)
        self.assertEqual(result['contrasts']['batch_to_single']['all']['sites']['eos']['mixture']['mean_target_probability_change'], 0.)
        self.assertEqual(result['contrasts']['reference_to_generated_body_same_suffix']['all']['paired_samples'], 0)
        row['conditions']['reference_single']['sites']['eos']['primary']['target_token'] = '.'
        with self.assertRaisesRegex(ValueError, 'target token unchanged'):
            audit.contrasts([row], 'reference_native_batch', 'reference_single')

    def test_synthetic_pipeline_and_preflight_never_read_test_or_modify_inputs(self):
        model = self.model()
        trace = self.focus_case(model, torch.zeros(1, 3, 8))
        for check in (True, False):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, traced, output = root / 'source', root / 'trace', root / 'output'
                source.mkdir()
                traced.mkdir()
                config = {**read_json(PACKAGE / 'config.json'), 'amp': False, 'batch_size': 2,
                          'data_directory': str(root / 'data')}
                write_json(source / 'config.json', config)
                (source / 'best.pt').write_bytes(b'fixture, checkpoint loader mocked')
                write_json(traced / 'manifest.json', {'diagnostic': 'paper_core_risk_beam_trace_v1', 'split': 'VAL',
                           'checkpoint_sha256': fingerprint(source / 'best.pt'), 'source_run': str(source), 'selected_ids': ['val:0']})
                write_json(traced / 'status.json', {'stage': 'complete'})
                write_json(traced / 'cases.json', [trace])
                before = {str(path): path.read_bytes() for folder in (source, traced) for path in folder.iterdir()}
                tokens = self.tokens
                class Reader:
                    def __init__(self, name):
                        self.name, self.calls, self.closed = name, 0, False
                        self.records = [{'id': f'{name.lower()}:0'}]
                        self.captions, self.lengths = [tokens], [len(tokens)]

                    def __getitem__(self, index):
                        if self.name == 'TEST':
                            raise AssertionError('No TEST image access')
                        self.calls += 1
                        return {'id': self.records[index]['id'], 'image': torch.zeros(3, 8)}

                    def close(self):
                        self.closed = True
                splits = {key: Reader(key) for key in ('TRAIN', 'VAL', 'TEST')}
                args = ['diagnose_endings.py', '--run', str(source), '--trace', str(traced), '--output', str(output), '--device', 'cpu']
                if check:
                    args.append('--check')
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(audit, 'load_dataset', return_value=(self.words, {}, splits, {})), \
                        patch.object(audit, 'load_checkpoint', return_value={'epoch': 5, 'model': model.state_dict()}), \
                        patch.object(audit, 'build_model', return_value=model) as build, \
                        patch('torch.optim.Adam', side_effect=AssertionError('No optimization')):
                    audit.main()
                self.assertTrue(all(split.closed for split in splits.values()))
                self.assertEqual(before, {str(path): path.read_bytes() for folder in (source, traced) for path in folder.iterdir()})
                if check:
                    build.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertTrue(all(split.calls == 0 for split in splits.values()))
                else:
                    self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
                    self.assertEqual(splits['TEST'].calls, 0)
                    val = read_json(output / 'val_cases.json')
                    self.assertEqual(len(val[0]['conditions']), 5)
                    self.assertEqual(val[0]['conditions']['reference_single'], val[0]['conditions']['reference_native_batch'])
                    with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO), \
                            patch.object(audit, 'load_dataset') as load:
                        with self.assertRaises(SystemExit):
                            audit.main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
