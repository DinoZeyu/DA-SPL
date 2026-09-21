"""Synthetic tests only: audit coverage, causal inputs, provenance and read-only behavior."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method import audit_pipeline as audit
from paper_method.data import fingerprint, read_json, write_json
from paper_method.evaluation import FIELDS
from paper_method.model import DASPL, VisualEncoder
from paper_method.runner import PACKAGE


class PipelineAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        values = {'optic disc size': 'normal', 'cup to disc ratio': '0.4', 'neuroretinal rim': 'thick , pink',
                  'rim color': 'pink', 'glaucoma risk assessment': 'very healthy', 'confidence level': '0.95'}
        text = ' '.join(f'{f} : {values.get(f, "false")} .' for f in FIELDS)
        self.texts = [text, text.replace('normal', 'large').replace('very healthy', 'high risk')]
        vocabulary = ['<pad>', '<unk>', '<start>', '<end>'] + ' '.join(self.texts).split()
        self.words = {word: i for i, word in enumerate(dict.fromkeys(vocabulary))}
        words, texts = self.words, self.texts

        class Reader:
            def __init__(self, name):
                self.records = [{'id': f'{name}:{i}'} for i in range(2)]
                self.captions = [[words[w] for w in ('<start> ' + text + ' <end>').split()] for text in texts]
                self.lengths = list(map(len, self.captions))
                self.name, self.calls, self.closed = name, 0, False

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                if self.name == 'TEST':
                    raise AssertionError('No TEST image access')
                self.calls += 1
                return {'id': self.records[index]['id'], 'image': torch.full((3, 8), float(index + 1))}

            def close(self):
                self.closed = True

        class Backbone(torch.nn.Module):
            def forward_features(self, images):
                return images

        self.splits = {name: Reader(name) for name in ('TRAIN', 'VAL', 'TEST')}
        self.records = {name: audit.records_for(self.splits[name], words) for name in ('TRAIN', 'VAL')}
        encoder = VisualEncoder(Backbone(), 8, 8)
        self.model = DASPL(encoder, len(words), 2, dim=8, heads=2, pad_id=words['<pad>']).eval()
        self.config = {**read_json(PACKAGE / 'config.json'), 'amp': False, 'tag_loss_weight': 0.,
                       'batch_size': 2, 'max_decode_steps': 6}
        self.device = torch.device('cpu')

    def test_score_spans_match_next_tokens_and_exclude_template_and_eos(self):
        record = self.records['TRAIN'][0]
        tokens = record['tokens']
        scores = torch.full((len(tokens) - 1, len(self.words)), -12.)
        scores[torch.arange(len(tokens) - 1), torch.tensor(tokens[1:])] = 0
        values = audit.describe_scores({head: scores for head in audit.HEADS}, record, self.words)
        for result in values.values():
            self.assertEqual(sum(s['tokens'] for s in result['token_buckets'].values()), len(tokens) - 1)
            self.assertEqual(result['token_buckets']['eos']['tokens'], 1)
            self.assertTrue(all(f['value_tokens_correct'] for f in result['fields'].values()))
        start, end = record['spans']['neuroretinal rim']
        self.assertEqual([tokens[i + 1] for i in range(start, end)], [self.words[w] for w in 'thick , pink'.split()])
        scores[start + 2] = -12.
        scores[start + 2, self.words['false']] = 0
        result = audit.describe_scores({head: scores for head in audit.HEADS}, record, self.words)['mixture']['fields']['neuroretinal rim']
        self.assertTrue(result['first_correct'])
        self.assertFalse(result['value_tokens_correct'])

    def test_class_recall_and_majority_never_fit_val(self):
        records = copy.deepcopy(self.records)
        records['TRAIN'][1]['values']['optic disc size'] = 'normal'
        for row in records['VAL']:
            row['values']['optic disc size'] = 'large'
        result = audit.majority_baselines(records)['optic disc size']
        self.assertEqual(result['majority'], 'normal')
        self.assertEqual(result['splits']['VAL']['rate'], 0)
        score = audit.agreement(['a', 'a', 'a', 'b'], ['a'] * 4)
        self.assertEqual(score['rate'], .75)
        self.assertEqual(score['macro_recall'], .5)

    def test_knn_excludes_self_and_only_votes_from_train(self):
        features = {'TRAIN': {'raw_cls': torch.eye(2)}, 'VAL': {'raw_cls': torch.eye(2)}}
        before = copy.deepcopy(features)
        result = audit.neighbor_probe(features, self.records, k=1)['views']['raw_cls']
        self.assertEqual(result['TRAIN']['neighbors'][0]['train_ids'], ['TRAIN:1'])
        self.assertEqual(result['VAL']['neighbors'][0]['train_ids'], ['TRAIN:0'])
        modified = copy.deepcopy(self.records)
        modified['VAL'][0]['values']['optic disc size'] = 'unseen'
        again = audit.neighbor_probe(features, modified, k=1)['views']['raw_cls']
        self.assertEqual(again['VAL']['neighbors'], result['VAL']['neighbors'])
        self.assertEqual(again['VAL']['fields']['optic disc size']['prediction_counts'],
                         result['VAL']['fields']['optic disc size']['prediction_counts'])
        for name in features:
            torch.testing.assert_close(features[name]['raw_cls'], before[name]['raw_cls'])

    def test_teacher_conditions_use_same_prefix_and_never_mutate_parameters(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        memories = [torch.randn(1, 3, 8) for _ in range(2)]
        mean = torch.stack(memories).mean(0)
        with patch('sys.stdout', new_callable=io.StringIO), \
                patch.object(self.model.decoder, 'forward', wraps=self.model.decoder.forward) as forward:
            cases, attention = audit.teacher_audit(self.model.decoder, memories, self.records['VAL'], mean,
                                                  self.words, self.config, self.device, batch_probes=True)
        for i in range(2):
            own, constant = forward.call_args_list[2 * i:2 * i + 2]
            torch.testing.assert_close(own.args[1], constant.args[1])
            torch.testing.assert_close(constant.args[0], mean)
        self.assertEqual(len(cases), 2)
        self.assertEqual(set(cases[0]['conditions']), {'own_single', 'train_mean_memory', 'saved_order_batch', 'shuffled_batch'})
        summary = audit.summarize_scores(cases, self.records['VAL'], self.words)
        self.assertEqual(summary['conditions']['own_single']['mixture']['fields']['rim color']['first_token']['samples'], 2)
        self.assertGreater(attention['calls'], 0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_single_head_routes_same_step_logits_and_encoding_runs_once(self):
        with patch('sys.stdout', new_callable=io.StringIO):
            memories, features = audit.encode(self.model, self.splits['VAL'], self.device)
        self.assertEqual(self.splits['VAL'].calls, 2)
        self.assertEqual(features['raw_cls'].shape, (2, 8))
        memory = memories[0]
        state = self.model.decoder.initial_state(memory)
        token = torch.tensor([self.words['<start>']])
        with torch.no_grad():
            a, b, expected_state = self.model.decoder.step(memory, token, state)
            for head, expected in (('primary', a), ('secondary', b)):
                first, second, actual_state = audit.SingleHead(self.model.decoder, head).step(memory, token, state)
                torch.testing.assert_close(first, expected)
                torch.testing.assert_close(second, expected)
                for actual, wanted in zip(actual_state, expected_state):
                    torch.testing.assert_close(actual, wanted)

    def test_parameter_audit_and_raw_normalization(self):
        before = {'encoder.backbone.w': torch.ones(2), 'decoder.primary_head.w': torch.zeros(2)}
        after = {k: v.clone() for k, v in before.items()}
        after['decoder.primary_head.w'][0] = 1
        result = audit.parameter_updates(before, after)
        self.assertEqual(result['encoder.backbone']['changed_tensors'], 0)
        self.assertEqual(result['decoder.primary_head']['changed_tensors'], 1)
        after['encoder.backbone.w'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
            audit.parameter_updates(before, after)
        self.assertEqual(audit.normalized_value(.95), '0.95')
        self.assertEqual(audit.normalized_value(False), 'false')
        self.assertEqual(audit.normalized_value(None), 'not reported')
        self.assertEqual(audit.normalized_value('Thick, pink.'), 'thick , pink')

    def test_raw_alignment_checks_actual_pixels_and_encoding_not_just_metadata(self):
        import h5py
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            records = copy.deepcopy(self.records['TRAIN'])
            raw, saved, arrays = [], [], []
            tags = {'optic_disc_size=normal': 0, 'optic_disc_size=large': 1,
                    '<pad>': 2, '<unk>': 3, '<start>': 4, '<end>': 5}
            reverse = {i: word for word, i in self.words.items()}
            for i, record in enumerate(records):
                record['id'] = f'train:{i}'
                image = Image.fromarray(np.full((9, 9, 3), 30 + i, dtype=np.uint8))
                stream = io.BytesIO()
                image.save(stream, format='PNG')
                arrays.append(np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR)).transpose(2, 0, 1))
                description = {'fundus_features': {f.replace(' ', '_'): v for f, v in record['values'].items()
                                                  if f not in ('glaucoma risk assessment', 'confidence level')},
                               'glaucoma_risk_assessment': record['risk'], 'confidence_level': .95}
                raw.append({'image': {'bytes': stream.getvalue(), 'path': None}, 'label': i,
                            'annotation': str(i), 'filename': f'{i}.png', 'description': json.dumps(description)})
                saved.append({'id': record['id'], 'hf_split': 'train', 'hf_row': i,
                              'label': i, 'annotation': str(i), 'filename': f'{i}.png',
                              'tokens': [reverse[t] for t in record['tokens'][1:-1]],
                              'tags': ['optic_disc_size=' + record['values']['optic disc size']]})
            raw_path = directory / 'train.parquet'
            pq.write_table(pa.Table.from_pylist(raw), raw_path)
            write_json(directory / 'config.json', {'dataset': {'directory': str(directory),
                       'splits': {'train': {'file': 'train.parquet'}}}})
            write_json(directory / 'audit.json', {'provenance': {'files': {'train.parquet': fingerprint(raw_path)}}})
            image_path = directory / 'TRAIN.hdf5'
            with h5py.File(image_path, 'w') as handle:
                handle.create_dataset('images', data=np.stack(arrays))
            split = SimpleNamespace(records=saved, image_path=image_path, labels=[[1, 0], [0, 1]])
            result = audit.data_alignment(directory, {'TRAIN': split}, {'TRAIN': records}, self.words, tags)
            self.assertEqual(result['issues'], [])
            self.assertEqual(result['splits']['TRAIN']['raw_field_matches'], 28)
            with h5py.File(image_path, 'r+') as handle:
                handle['images'][0] = arrays[1]
            split.labels[0] = [0, 1]
            result = audit.data_alignment(directory, {'TRAIN': split}, {'TRAIN': records}, self.words, tags)
            self.assertIn('raw_to_hdf5_pixels', result['issues'][0]['failures'])
            self.assertIn('tag_encoding', result['issues'][0]['failures'])
            self.assertEqual(len(result['exact_processed_pixel_duplicates']), 1)

    def test_comparison_requires_matching_checkpoint_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, comparison = root / 'source', root / 'comparison'
            (source / 'epoch_005').mkdir(parents=True)
            comparison.mkdir()
            (source / 'last.pt').write_bytes(b'checkpoint fixture')
            predictions = [{'id': r['id'], 'reference': r['reference'], 'prediction': r['reference'],
                            'token_ids': r['tokens'], 'terminated_with_end': True, 'log_probability': -1.}
                           for r in self.records['VAL']]
            config = {**self.config, 'max_decode_steps': 161}
            baseline = source / 'epoch_005' / 'predictions.json'
            write_json(baseline, predictions)
            for name in ('beam5', 'greedy'):
                write_json(comparison / f'{name}_predictions.json', predictions)
            write_json(comparison / 'config.json', config)
            write_json(comparison / 'data_fingerprints.json', {})
            write_json(comparison / 'status.json', {'stage': 'complete'})
            manifest = {'experiment': 'paper_core_greedy_vs_beam5_v1', 'checkpoint_sha256': fingerprint(source / 'last.pt'),
                        'checkpoint_epoch': 5, 'max_decode_steps': 161, 'prediction_sha256': fingerprint(baseline)}
            write_json(comparison / 'manifest.json', manifest)
            audit.validate_comparison(comparison, source, config, {'epoch': 5}, {}, self.records['VAL'], self.words)
            manifest['checkpoint_epoch'] = 4
            write_json(comparison / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                audit.validate_comparison(comparison, source, config, {'epoch': 5}, {}, self.records['VAL'], self.words)

    def test_cli_check_and_mocked_full_workflow_never_train_or_touch_test(self):
        for check in (True, False):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'paired' / 'lem_off'
                source.mkdir(parents=True)
                output = root / 'result'
                weights = root / 'weights.pt'
                weights.write_bytes(b'synthetic builder is mocked')
                config = {**self.config, 'data_directory': str(root / 'data'), 'pretrained_weights': str(weights)}
                write_json(source / 'config.json', config)
                write_json(source / 'history.json', [])
                state = {k: v.clone() for k, v in self.model.state_dict().items()}
                torch.save({'model': state}, source.parent / 'initial.pt')
                write_json(source.parent / 'initialization.json', {'file_sha256': fingerprint(source.parent / 'initial.pt')})
                checkpoint = {'model': state, 'epoch': 5, 'initial_state_sha256': audit.state_digest(state)}
                torch.save(checkpoint, source / 'last.pt')
                comparison = root / 'comparison'
                comparison.mkdir()
                saved = {name: [] for name in ('beam5', 'greedy')}
                for name in saved:
                    for record in self.records['VAL']:
                        saved[name].append({'id': record['id'], 'reference': record['reference'], 'prediction': record['reference'],
                                            'token_ids': record['tokens'], 'terminated_with_end': True, 'log_probability': -1.})
                    write_json(comparison / f'{name}_predictions.json', saved[name])
                args = ['audit_pipeline.py', '--run', str(source), '--output', str(output),
                        '--comparison', str(comparison), '--device', 'cpu'] + (['--check'] if check else [])
                for split in self.splits.values():
                    split.calls, split.closed = 0, False
                before = {str(p): p.read_bytes() for p in source.parent.rglob('*') if p.is_file()}
                with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                        patch.object(audit, 'load_dataset', return_value=(self.words, {}, self.splits, {})), \
                        patch.object(audit, 'load_source', return_value=(config, checkpoint)), \
                        patch.object(audit, 'validate_comparison', return_value=saved), \
                        patch.object(audit, 'data_alignment', return_value={'issues': [], 'exact_processed_pixel_duplicates': []}), \
                        patch.object(audit, 'build_model', return_value=self.model) as build, \
                        patch.object(audit, 'neighbor_probe', return_value={}), \
                        patch.object(audit, 'replay_prediction', return_value=-1.), \
                        patch('torch.optim.Adam', side_effect=AssertionError('No training')):
                    audit.main()
                self.assertTrue(all(s.closed for s in self.splits.values()))
                self.assertEqual(self.splits['TEST'].calls, 0)
                self.assertEqual(before, {str(p): p.read_bytes() for p in source.parent.rglob('*') if p.is_file()})
                for key, value in self.model.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                if check:
                    build.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertEqual(sum(s.calls for s in self.splits.values()), 0)
                else:
                    self.assertEqual(read_json(output / 'status.json')['stage'], 'complete')
                    self.assertEqual(sum(s.calls for s in self.splits.values()), 4)
                    self.assertEqual(set(read_json(output / 'summary.json')['generation']),
                                     {'train_mixture', 'val_primary', 'val_secondary', 'val_greedy', 'val_beam5'})
                    with patch('sys.argv', args), patch('sys.stderr', new_callable=io.StringIO):
                        with self.assertRaises(SystemExit):
                            audit.main()


if __name__ == '__main__':
    unittest.main()
