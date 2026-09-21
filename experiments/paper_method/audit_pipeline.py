"""One read-only TRAIN/VAL audit of data, fitting, vision, branches and batch effects."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch.nn import functional as F

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.ablate_lem import seed_all, state_digest
from paper_method.compare_decoding import mode_summary
from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.decoding import beam_search
from paper_method.diagnose_fields import field_prefix, load_source, replay_prediction, validate_predictions
from paper_method.evaluation import FIELDS, parse_fields
from paper_method.model import build_model, mixture_log_probs
from paper_method.runner import PACKAGE, ROOT, archive_sources, resolve

HEADS = ('primary', 'secondary', 'mixture')
LIMITS = [
    'Final lem_off checkpoint only; no optimization, parameter changes or TEST inference.',
    'Teacher forcing uses earlier reference tokens: its value-token matches are NOT freely generated field accuracy.',
    'Mean TRAIN memory is an image-independent, potentially out-of-distribution intervention, not a trained text-only model.',
    'Fixed cosine 5-NN probes use TRAIN labels only, exclude self for TRAIN, and never tune on VAL.',
    'Pooled-feature kNN failure does not prove absence of visual information; success does not establish clinical validity.',
    'Batch probes change only cached-memory grouping in eval FP32, not historical AMP training conditions.',
    'Parameter deltas include weight decay; a changed tensor does not by itself prove useful task gradients.',
    'Single-head decoding is a diagnostic ablation, not a change to the paper-method default.',
    'Raw alignment checks verify stored-source consistency, not correctness of the source clinical annotations.',
    'TRAIN majority baselines, per-class recalls and token categories expose imbalance; no target reweighting is applied.',
    'Repeatedly inspected VAL and a single five-epoch seed cannot establish generalization or a unique failure cause.',
]


def records_for(split, words):
    reverse = {index: word for word, index in words.items()}
    result = []
    for row, caption, length in zip(split.records, split.captions, split.lengths):
        tokens = caption[:length]
        text = ' '.join(reverse[t] for t in tokens[1:-1])
        fields = parse_fields(text)
        if any(len(v) != 1 or not v[0]['complete'] for v in fields.values()):
            raise ValueError(f'{row["id"]}: malformed reference')
        spans = {}
        for field in FIELDS:
            prefix = field_prefix(tokens, field, words)
            if prefix is None:
                raise ValueError(f'{row["id"]}: missing field prefix')
            start = len(prefix) - 1  # Logit t predicts caption token t+1.
            spans[field] = (start, start + len(fields[field][0]['value'].split()))
        result.append({'id': row['id'], 'tokens': tokens, 'reference': text, 'spans': spans,
                       'values': {f: fields[f][0]['value'] for f in FIELDS},
                       'risk': fields['glaucoma risk assessment'][0]['value']})
    return result


def agreement(targets, predictions):
    counts, correct = Counter(targets), Counter()
    for target, prediction in zip(targets, predictions):
        correct[target] += target == prediction
    if len(targets) != len(predictions) or not targets:
        raise ValueError('Agreement requires nonempty aligned labels')
    recalls = {key: correct[key] / count for key, count in sorted(counts.items())}
    return {'samples': len(targets), 'matches': sum(correct.values()),
            'rate': sum(correct.values()) / len(targets),
            'macro_recall': sum(recalls.values()) / len(recalls), 'class_recall': recalls,
            'reference_counts': dict(counts), 'prediction_counts': dict(Counter(predictions))}


def majority_baselines(records):
    result = {}
    for field in FIELDS:
        counts = Counter(r['values'][field] for r in records['TRAIN'])
        majority = min(counts, key=lambda key: (-counts[key], key))
        result[field] = {'train_counts': dict(counts), 'majority': majority,
                         'splits': {name: agreement([r['values'][field] for r in rows], [majority] * len(rows))
                                    for name, rows in records.items()}}
    return result


def normalized_value(value):
    if value is None:
        value = 'not reported'
    elif isinstance(value, bool):
        value = str(value).lower()
    elif not isinstance(value, (str, int, float)):
        raise ValueError('Unsupported structured source field; inspect rather than invent a conversion')
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?|[^\w\s]", str(value).lower())
    while tokens and tokens[-1] == '.':
        tokens.pop()
    return ' '.join(tokens)


def data_alignment(directory, splits, records, words, tags):
    import h5py
    import pyarrow.parquet as pq
    from PIL import Image

    prepared = read_json(directory / 'config.json')['dataset']
    raw_root = resolve(prepared['directory'])
    provenance = read_json(directory / 'audit.json')['provenance']['files']
    sources, source_hashes = {}, {}
    needed = {r['hf_split'] for name in records for r in splits[name].records}
    for name in sorted(needed):
        filename = prepared['splits'][name]['file']
        path = raw_root / filename
        source_hashes[filename] = fingerprint(path)
        if source_hashes[filename] != provenance[filename]:
            raise ValueError(f'Raw source changed: {filename}')
        sources[name] = pq.read_table(path).to_pylist()
    issues, pixels, stats = [], defaultdict(list), {}
    for name, rows in records.items():
        matched_pixels, matched_fields = 0, 0
        with h5py.File(splits[name].image_path, 'r') as handle:
            for i, (saved, record) in enumerate(zip(splits[name].records, rows)):
                raw = sources[saved['hf_split']][saved['hf_row']]
                failures = []
                if saved['id'] != f'{saved["hf_split"]}:{saved["hf_row"]}' or any(
                        raw[key] != saved[key] for key in ('filename', 'annotation', 'label')):
                    failures.append('source_identity')
                raw_image = Image.open(io.BytesIO(raw['image']['bytes'])).convert('RGB')
                expected = np.asarray(raw_image.resize((224, 224), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
                actual = handle['images'][i]
                matched_pixels += int(np.array_equal(expected, actual))
                if not np.array_equal(expected, actual):
                    failures.append('raw_to_hdf5_pixels')
                pixels[hashlib.sha256(actual.tobytes()).hexdigest()].append({'split': name, 'id': saved['id']})
                expected_tokens = [words['<start>']] + [words.get(t, words['<unk>']) for t in saved['tokens']] + [words['<end>']]
                if expected_tokens != record['tokens']:
                    failures.append('caption_encoding')
                expected_tags = [0] * (len(tags) - 4)
                for tag in saved['tags']:
                    if tag not in tags or tags[tag] >= len(expected_tags):
                        failures.append('uncovered_tag:' + tag)
                    else:
                        expected_tags[tags[tag]] = 1
                if expected_tags != splits[name].labels[i]:
                    failures.append('tag_encoding')
                description = json.loads(raw['description'])
                tag_fields = {t.partition('=')[0] for t in tags if not t.startswith('<')}
                raw_tags = {key + '=' + ' '.join(str(description['fundus_features'][key]).lower().split())
                            for key in tag_fields}
                if raw_tags != set(saved['tags']):
                    failures.append('raw_to_saved_tags')
                for field in FIELDS:
                    key = field.replace(' ', '_')
                    value = (description if field in ('glaucoma risk assessment', 'confidence level') else
                             description['fundus_features'])[key]
                    normalized = normalized_value(value)
                    encoded = ' '.join(t if t in words else '<unk>' for t in normalized.split())
                    matched_fields += encoded == record['values'][field]
                    if encoded != record['values'][field]:
                        failures.append({'field': field, 'raw_normalized': encoded, 'target': record['values'][field]})
                if failures:
                    issues.append({'id': saved['id'], 'failures': failures})
        stats[name] = {'samples': len(rows), 'raw_pixel_matches': matched_pixels,
                       'raw_field_matches': matched_fields, 'raw_field_total': len(rows) * len(FIELDS),
                       'unknown_target_tokens': sum(r['tokens'].count(words['<unk>']) for r in rows),
                       'unique_reference_reports': len({r['reference'] for r in rows})}
    duplicates = [rows for rows in pixels.values() if len(rows) > 1]
    return {'splits': stats, 'issues': issues, 'exact_processed_pixel_duplicates': duplicates,
            'raw_fingerprints': source_hashes, 'test_images_read': False}


def parameter_updates(initial, final):
    if set(initial) != set(final):
        raise ValueError('Initial/final state keys differ')
    groups = defaultdict(lambda: {'tensors': 0, 'changed_tensors': 0, 'initial_squared_norm': 0., 'delta_squared_norm': 0.})
    for key, before in initial.items():
        after = final[key]
        if before.shape != after.shape or not torch.isfinite(after).all():
            raise ValueError(f'Invalid checkpoint tensor: {key}')
        group = '.'.join(key.split('.')[:2])
        item = groups[group]
        item['tensors'] += 1
        item['changed_tensors'] += not torch.equal(before, after)
        item['initial_squared_norm'] += before.double().square().sum().item()
        item['delta_squared_norm'] += (after.double() - before.double()).square().sum().item()
    for item in groups.values():
        item['relative_l2_change'] = (item['delta_squared_norm'] / max(item['initial_squared_norm'], 1e-30)) ** .5
    return dict(groups)


def distributions(first, second, coefficient):
    return {'primary': first.float().log_softmax(-1), 'secondary': second.float().log_softmax(-1),
            'mixture': mixture_log_probs(first, second, coefficient)}


def describe_scores(scores, record, words):
    targets = torch.tensor(record['tokens'][1:], device=scores['mixture'].device)
    kinds = ['template'] * len(targets)
    for start, end in record['spans'].values():
        kinds[start:end] = ['value'] * (end - start)
    punctuation = {words[t] for t in ('.', ',', ':', ';') if t in words}
    for i, token in enumerate(record['tokens'][1:]):
        if token in punctuation:
            kinds[i] = 'punctuation'
    kinds[-1] = 'eos'
    result = {}
    for head, log_probs in scores.items():
        probs = log_probs[:len(targets)].detach().cpu()
        target_cpu = targets.cpu()
        nll = -probs.gather(1, target_cpu[:, None]).squeeze(1)
        choices = probs.argmax(-1)
        fields = {}
        for field, (start, end) in record['spans'].items():
            fields[field] = {'first_id': choices[start].item(), 'first_correct': bool(choices[start] == target_cpu[start]),
                             'value_tokens_correct': bool(torch.equal(choices[start:end], target_cpu[start:end])),
                             'value_nll_sum': nll[start:end].sum().item(), 'value_tokens': end - start,
                             'first_target_probability': (-nll[start]).exp().item()}
        buckets = {}
        for kind in ('template', 'punctuation', 'value', 'eos'):
            mask = torch.tensor([k == kind for k in kinds])
            buckets[kind] = {'tokens': int(mask.sum()), 'correct': int((choices[mask] == target_cpu[mask]).sum()),
                             'nll_sum': nll[mask].sum().item()}
        result[head] = {'fields': fields, 'token_buckets': buckets}
    return result


def summarize_scores(cases, records, words):
    result = {}
    reverse = {index: word for word, index in words.items()}
    for condition in sorted({c for row in cases for c in row['conditions']}):
        selected = [(row, record) for row, record in zip(cases, records) if condition in row['conditions']]
        result[condition] = {}
        for head in HEADS:
            fields, buckets = {}, {}
            for field in FIELDS:
                scored = [row['conditions'][condition][head]['fields'][field] for row, _ in selected]
                first_targets = [reverse[record['tokens'][record['spans'][field][0] + 1]] for _, record in selected]
                first_choices = [reverse[s['first_id']] for s in scored]
                fields[field] = {'first_token': agreement(first_targets, first_choices),
                                 'teacher_forced_all_value_tokens_correct': sum(s['value_tokens_correct'] for s in scored),
                                 'value_token_nll': sum(s['value_nll_sum'] for s in scored) / sum(s['value_tokens'] for s in scored)}
            for kind in ('template', 'punctuation', 'value', 'eos'):
                sums = {key: sum(row['conditions'][condition][head]['token_buckets'][kind][key] for row, _ in selected)
                        for key in ('tokens', 'correct', 'nll_sum')}
                buckets[kind] = {**sums, 'nll': sums['nll_sum'] / sums['tokens'] if sums['tokens'] else None}
            result[condition][head] = {'fields': fields, 'token_buckets': buckets}
    contrasts = {}
    for condition in result:
        if condition == 'own_single':
            continue
        contrasts[condition] = {}
        for field in FIELDS:
            pairs = [(row['conditions']['own_single']['mixture']['fields'][field],
                      row['conditions'][condition]['mixture']['fields'][field]) for row in cases if condition in row['conditions']]
            contrasts[condition][field] = {'samples': len(pairs), 'first_choice_changes': sum(a['first_id'] != b['first_id'] for a, b in pairs),
                                          'correct_delta': sum(int(b['first_correct']) - int(a['first_correct']) for a, b in pairs),
                                          'mean_target_probability_delta': sum(b['first_target_probability'] - a['first_target_probability'] for a, b in pairs) / len(pairs)}
    return {'conditions': result, 'paired_mixture_contrasts_vs_own_single': contrasts}


@torch.no_grad()
def encode(model, split, device):
    memories, raw_cls, raw_mean = [], [], []

    def capture(_module, args):
        tokens = args[0]
        raw_cls.append(tokens[:, 0].cpu())
        raw_mean.append(tokens[:, 1:].mean(1).cpu())

    hook = model.encoder.projection.register_forward_pre_hook(capture)
    try:
        for i in range(len(split)):
            memory = model.encoder(split[i]['image'].unsqueeze(0).to(device)).cpu()
            if not torch.isfinite(memory).all():
                raise ValueError('Non-finite visual memory')
            memories.append(memory)
            if (i + 1) % 50 == 0 or i + 1 == len(split):
                print(f'Encoded {i + 1}/{len(split)}', flush=True)
    finally:
        hook.remove()
    return memories, {'raw_cls': torch.cat(raw_cls), 'raw_patch_mean': torch.cat(raw_mean),
                      'projected_cls': torch.cat([m[:, 0] for m in memories]),
                      'projected_patch_mean': torch.cat([m[:, 1:].mean(1) for m in memories])}


def neighbor_probe(features, records, k=5):
    result = {}
    for view in features['TRAIN']:
        train = F.normalize(features['TRAIN'][view].float(), dim=-1)
        result[view] = {}
        for name in ('TRAIN', 'VAL'):
            query = F.normalize(features[name][view].float(), dim=-1)
            similarities = query @ train.T
            if name == 'TRAIN':
                similarities.fill_diagonal_(-torch.inf)
            if len(train) < k + (name == 'TRAIN'):
                raise ValueError('Too few TRAIN samples for fixed kNN')
            # Stable sorting makes exact feature-similarity ties reproducible.
            neighbors = torch.argsort(similarities, dim=1, descending=True, stable=True)[:, :k].tolist()
            fields = {}
            for field in FIELDS:
                predicted = []
                for indices in neighbors:
                    counts = Counter(records['TRAIN'][i]['values'][field] for i in indices)
                    predicted.append(min(counts, key=lambda key: (-counts[key], key)))
                fields[field] = agreement([r['values'][field] for r in records[name]], predicted)
            result[view][name] = {'fields': fields, 'neighbors': [
                {'id': row['id'], 'train_ids': [records['TRAIN'][i]['id'] for i in indices]}
                for row, indices in zip(records[name], neighbors)]}
    return {'k': k, 'metric': 'cosine', 'train_self_excluded': True, 'views': result}


@torch.no_grad()
def teacher_audit(decoder, memories, records, mean_memory, words, config, device, batch_probes=False):
    cases, attention = [], []

    def capture(_module, args, output):
        context, info = output
        weights = info['combined_weights'].float()
        attention.append(torch.stack(((weights > 1e-8).sum().float(), weights.abs().sum(),
                                      context.float().square().mean().sqrt() / args[0].float().square().mean().sqrt().clamp_min(1e-12))).detach())

    for i, (memory, record) in enumerate(zip(memories, records)):
        conditions = {}
        tokens = torch.tensor([record['tokens']], device=device)
        lengths = torch.tensor([tokens.size(1)], device=device)
        for name, visual in (('own_single', memory), ('train_mean_memory', mean_memory)):
            hook = decoder.attention.register_forward_hook(capture) if name == 'own_single' else None
            try:
                a, b = decoder(visual.to(device), tokens, lengths)
            finally:
                if hook is not None:
                    hook.remove()
            conditions[name] = describe_scores({h: s[0] for h, s in distributions(a, b, config['lambda_parallel']).items()}, record, words)
        cases.append({'id': record['id'], 'conditions': conditions})
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f'Teacher forcing {i + 1}/{len(records)}', flush=True)
    if batch_probes:
        generator = torch.Generator().manual_seed(config['seed'])
        orders = {'saved_order_batch': list(range(len(records))),
                  'shuffled_batch': torch.randperm(len(records), generator=generator).tolist()}
        for condition, order in orders.items():
            for start in range(0, len(order), config['batch_size']):
                indices = order[start:start + config['batch_size']]
                lengths = torch.tensor([len(records[i]['tokens']) for i in indices], device=device)
                tokens = torch.full((len(indices), int(lengths.max())), words['<pad>'], device=device)
                for j, i in enumerate(indices):
                    tokens[j, :lengths[j]] = torch.tensor(records[i]['tokens'], device=device)
                memory = torch.cat([memories[i] for i in indices]).to(device)
                a, b = decoder(memory, tokens, lengths)
                scores = distributions(a, b, config['lambda_parallel'])
                for j, i in enumerate(indices):
                    cases[i]['conditions'][condition] = describe_scores({h: s[j] for h, s in scores.items()}, records[i], words)
                    cases[i].setdefault('batch_members', {})[condition] = [records[n]['id'] for n in indices]
    weights = torch.stack(attention).cpu()
    attention_summary = {'calls': len(weights), 'all_heads_zero_calls': int((weights[:, 0] == 0).sum()),
                         'mean_active_heads': weights[:, 0].mean().item(), 'mean_weight_sum': weights[:, 1].mean().item(),
                         'mean_context_to_memory_rms': weights[:, 2].mean().item()}
    return cases, attention_summary


class SingleHead:
    def __init__(self, decoder, head):
        if head not in ('primary', 'secondary'):
            raise ValueError('Unknown head')
        self.decoder, self.head = decoder, head

    @property
    def training(self):
        return self.decoder.training

    def initial_state(self, memory):
        return self.decoder.initial_state(memory)

    def step(self, *args):
        a, b, state = self.decoder.step(*args)
        selected = a if self.head == 'primary' else b
        return selected, selected, state


@torch.no_grad()
def generate(decoder, memories, records, words, config, device):
    rows = []
    for i, (memory, record) in enumerate(zip(memories, records)):
        result = beam_search(decoder, memory.to(device), words, coefficient=config['lambda_parallel'], width=1,
                             max_steps=config['max_decode_steps'])
        rows.append({'id': record['id'], 'reference': record['reference'], **result})
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f'Greedy generation {i + 1}/{len(records)}', flush=True)
    return rows


def generation_metrics(rows, records):
    summary, checks = mode_summary(rows, records)
    parsed = [parse_fields(row['prediction']) for row in rows]
    summary['field_agreement'] = {
        field: agreement([r['values'][field] for r in records],
                         [p[field][0]['value'] if len(p[field]) == 1 and p[field][0]['complete'] else '<invalid>'
                          for p in parsed]) for field in FIELDS}
    return summary, checks


def validate_comparison(path, source, config, checkpoint, hashes, records, words):
    manifest = read_json(path / 'manifest.json')
    if (read_json(path / 'status.json')['stage'] != 'complete' or
            manifest['experiment'] != 'paper_core_greedy_vs_beam5_v1' or
            manifest['checkpoint_sha256'] != fingerprint(source / 'last.pt') or
            manifest['checkpoint_epoch'] != checkpoint['epoch'] or
            manifest['max_decode_steps'] != config['max_decode_steps'] or
            read_json(path / 'config.json') != config or read_json(path / 'data_fingerprints.json') != hashes):
        raise ValueError('Greedy comparison provenance mismatch')
    result = {name: read_json(path / f'{name}_predictions.json') for name in ('beam5', 'greedy')}
    for rows in result.values():
        validate_predictions(rows, records, words, config['max_decode_steps'])
    baseline = source / f'epoch_{checkpoint["epoch"]:03d}' / 'predictions.json'
    if manifest['prediction_sha256'] != fingerprint(baseline) or result['beam5'] != read_json(baseline):
        raise ValueError('Saved beam predictions differ from the source')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'artifacts/paper_method/lem_ablation_seed123/lem_off')
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Raw-data/checkpoint audit only, no model construction or inference')
    args = parser.parse_args()
    source = resolve(args.run)
    prefix = source.parent.with_name(source.parent.name + '_' + source.name)
    output = resolve(args.output or prefix.with_name(prefix.name + '_pipeline_audit'))
    comparison = resolve(args.comparison or prefix.with_name(prefix.name + '_decoding_comparison'))
    config = read_json(source / 'config.json')
    directory = resolve(config['data_directory'])
    if (output.exists() or any(p == output or p in output.parents for p in (directory, PACKAGE, ROOT / 'data')) or
            any((p / 'status.json').is_file() for p in output.parents)):
        parser.error('Choose a new output directory outside existing runs, data and code')
    torch.set_num_threads(4)
    words, tags, splits, hashes = load_dataset(directory)
    try:
        config, checkpoint = load_source(source, words, tags, hashes)
        records = {name: records_for(splits[name], words) for name in ('TRAIN', 'VAL')}
        saved = validate_comparison(comparison, source, config, checkpoint, hashes, records['VAL'], words)
        initial_path = source.parent / 'initial.pt'
        initialization = read_json(source.parent / 'initialization.json')
        if fingerprint(initial_path) != initialization['file_sha256']:
            raise ValueError('Initial checkpoint fingerprint mismatch')
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        if state_digest(initial['model']) != checkpoint['initial_state_sha256']:
            raise ValueError('Initial state differs from the paired run')
        print('Checking raw data, target encoding and parameter updates...', flush=True)
        static = {'data_alignment': data_alignment(directory, splits, records, words, tags),
                  'parameter_updates': parameter_updates(initial['model'], checkpoint['model']),
                  'majority_baselines': majority_baselines(records),
                  'training_history': read_json(source / 'history.json')}
        del initial
        if static['data_alignment']['issues'] or static['data_alignment']['exact_processed_pixel_duplicates']:
            print(json.dumps(static['data_alignment'], indent=2), flush=True)
            raise ValueError('Data alignment requires inspection before model diagnostics')
        weights = resolve(config['pretrained_weights'])
        if not weights.is_file():
            parser.error('Cached pretrained weights required; no download')
        manifest = {'experiment': 'paper_core_pipeline_audit_v1', 'source_run': str(source),
                    'checkpoint_sha256': fingerprint(source / 'last.pt'), 'epoch': checkpoint['epoch'],
                    'comparison': str(comparison), 'samples': {n: len(r) for n, r in records.items()},
                    'comparison_prediction_hashes': {n: fingerprint(comparison / f'{n}_predictions.json') for n in saved},
                    'device': args.device, 'training': False, 'test_inference': False, 'amp': False, 'limits': LIMITS}
        if args.check:
            print(json.dumps({'manifest': manifest, 'data_alignment': static['data_alignment'],
                              'parameter_updates': static['parameter_updates']}, indent=2), flush=True)
            return
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA unavailable. Run on an allocated GPU node')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        seed_all(config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(output / 'manifest.json', manifest)
            write_json(output / 'config.json', config)
            write_json(output / 'data_fingerprints.json', hashes)
            write_json(output / 'static_audit.json', static)
            archive_sources(output)
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(weights)}, words, tags, device)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.eval().requires_grad_(False)
            del checkpoint
            memories, features = {}, {}
            for name in records:
                write_json(output / 'status.json', {'stage': 'encoding', 'split': name})
                print(f'Encoding {name} once', flush=True)
                memories[name], features[name] = encode(model, splits[name], device)
            replays = []
            for i, record in enumerate(records['VAL']):
                replays.append({'id': record['id'], **{name: replay_prediction(model.decoder, memories['VAL'][i].to(device),
                                  rows[i], config['lambda_parallel']) for name, rows in saved.items()}})
            write_json(output / 'prediction_replay.json', replays)
            write_json(output / 'feature_knn.json', neighbor_probe(features, records))
            del features
            mean_memory = torch.stack(memories['TRAIN']).mean(0)
            teacher, attention, generated = {}, {}, {}
            for name in records:
                write_json(output / 'status.json', {'stage': 'teacher_forcing', 'split': name})
                print(f'{name}: reference/mean-memory and branch audit', flush=True)
                cases, attention[name] = teacher_audit(model.decoder, memories[name], records[name], mean_memory,
                                                      words, config, device, batch_probes=name == 'VAL')
                write_json(output / f'{name.lower()}_teacher_cases.json', cases)
                teacher[name] = summarize_scores(cases, records[name], words)
                del cases
            modes = [('train_mixture', 'TRAIN', model.decoder),
                     ('val_primary', 'VAL', SingleHead(model.decoder, 'primary')),
                     ('val_secondary', 'VAL', SingleHead(model.decoder, 'secondary'))]
            for mode, name, decoder in modes:
                write_json(output / 'status.json', {'stage': 'generating', 'mode': mode})
                print(mode, flush=True)
                rows = generate(decoder, memories[name], records[name], words, config, device)
                write_json(output / f'{mode}_predictions.json', rows)
                generated[mode], checks = generation_metrics(rows, records[name])
                write_json(output / f'{mode}_report_checks.json', checks)
            for name, rows in saved.items():
                generated['val_' + name], _ = generation_metrics(rows, records['VAL'])
            summary = {'teacher_forcing': teacher, 'attention': attention, 'generation': generated, 'limits': LIMITS}
            write_json(output / 'summary.json', summary)
            write_json(output / 'status.json', {'stage': 'complete', 'samples': manifest['samples']})
            print(json.dumps({'output': str(output), 'stage': 'complete'}, indent=2), flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
