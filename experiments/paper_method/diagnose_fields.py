"""VAL field/image sensitivity for the final lem_off checkpoint; no training or TEST inference."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import tarfile

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.ablate_lem import FORMAT, arm_configs, seed_all
from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.diagnose_risk import HEADS, head_distributions, strict_generated_risk
from paper_method.evaluation import parse_fields
from paper_method.model import build_model, mixture_log_probs
from paper_method.runner import PACKAGE, ROOT, archive_sources, resolve
from paper_method.trace_risk_beam import same_score

FIELDS = ('optic disc size', 'cup to disc ratio', 'rim color')
CONTRASTS = {
    'image_swap_reference_prefix': ('own_image_reference_prefix', 'donor_image_reference_prefix'),
    'image_swap_generated_prefix': ('own_image_generated_prefix', 'donor_image_generated_prefix'),
    'reference_to_generated_prefix': ('own_image_reference_prefix', 'own_image_generated_prefix'),
}
LIMITS = [
    'Final lem_off checkpoint, VAL only, one image per decoder call, eval FP32; no optimization or new beam search.',
    'The first field marker ends each prefix; its value and all later text are withheld.',
    'Candidates are single-token field values from TRAIN references only; multiword and unseen values are reported as uncovered.',
    'Candidate probabilities retain full-vocabulary normalization; restricted rankings are not free-generation accuracy.',
    'Value-plus-period scores are diagnostic continuations, not rewritten reports or globally optimal beam paths.',
    'Donors differ in the target field and preferentially share the exact reference risk label; selection labels are not model inputs.',
    'Donors can be reused; same-risk and cross-risk pairings are reported separately and are not independent samples.',
    'A swapped image changes many visual properties, and image/text combinations can be out of distribution.',
    'A fixed-image prefix swap also changes wording and possibly length; it is not a single clinical-variable intervention.',
    'Saved generated paths must replay their log probabilities before probing; missing generated prefixes are never filled from references.',
    'This measures checkpoint sensitivity, not encoder-only recognition, independent clinical validity or a unique historical cause.',
]


def field_prefix(tokens, field, words):
    marker = [words[word] for word in (field + ' :').split()]
    positions = [i for i in range(len(tokens) - len(marker) + 1) if tokens[i:i + len(marker)] == marker]
    if not positions:
        return None
    prefix = tokens[:positions[0] + len(marker)]
    if (prefix[0] != words['<start>'] or prefix.count(words['<start>']) != 1 or
            words['<end>'] in prefix or words['<pad>'] in prefix):
        raise ValueError('Invalid tokens before first field marker')
    return prefix


def report_records(split, words):
    reverse = {index: word for word, index in words.items()}
    records = []
    for item, caption, length in zip(split.records, split.captions, split.lengths):
        tokens = caption[:length]
        text = ' '.join(reverse[token] for token in tokens[1:-1])
        risk = strict_generated_risk(text)
        parsed, fields = parse_fields(text), {}
        if not risk['valid']:
            raise ValueError(f'{item["id"]}: malformed reference risk')
        for field in FIELDS:
            entries, prefix = parsed[field], field_prefix(tokens, field, words)
            if len(entries) != 1 or not entries[0]['complete'] or prefix is None:
                raise ValueError(f'{item["id"]}: malformed reference field {field}')
            fields[field] = {'value': entries[0]['value'], 'prefix_ids': prefix}
        records.append({'id': item['id'], 'reference': text, 'risk': risk['label'], 'fields': fields})
    return records


def candidate_audit(train_records, val_records, words):
    candidates, audit = {}, {}
    for field in FIELDS:
        counts = Counter(row['fields'][field]['value'] for row in train_records)
        values = sorted(value for value in counts if len(value.split()) == 1 and value in words)
        if len(values) < 2:
            raise ValueError(f'{field}: need at least two single-token TRAIN values')
        candidates[field] = values
        uncovered = Counter(row['fields'][field]['value'] for row in val_records if row['fields'][field]['value'] not in values)
        audit[field] = {'candidates': values, 'train_counts': dict(counts),
                        'excluded_train_values': {key: count for key, count in counts.items() if key not in values},
                        'val_samples': len(val_records), 'uncovered_val_counts': dict(uncovered)}
    return candidates, audit


def pair_donors(records, candidates, seed):
    rng = random.Random(seed)
    result = {}
    for field in FIELDS:
        order = list(range(len(records)))
        rng.shuffle(order)
        selected, uses = [None] * len(records), Counter()
        for i in order:
            value = records[i]['fields'][field]['value']
            if value not in candidates[field]:
                continue
            eligible = [j for j, row in enumerate(records) if row['fields'][field]['value'] in candidates[field]
                        and row['fields'][field]['value'] != value]
            same_risk = [j for j in eligible if records[j]['risk'] == records[i]['risk']]
            pool = same_risk or eligible
            if not pool:
                continue
            fewest = min(uses[j] for j in pool)
            selected[i] = rng.choice([j for j in pool if uses[j] == fewest])
            uses[selected[i]] += 1
        result[field] = selected
    return result


def validate_predictions(predictions, records, words, max_steps):
    if [row['id'] for row in predictions] != [row['id'] for row in records]:
        raise ValueError('Saved VAL prediction IDs/order differ from references')
    reverse = {index: word for word, index in words.items()}
    for row, record in zip(predictions, records):
        tokens = row['token_ids']
        if (not 2 <= len(tokens) <= max_steps + 1 or any(type(t) is not int or t not in reverse for t in tokens) or
                tokens[0] != words['<start>'] or tokens.count(words['<start>']) != 1 or words['<pad>'] in tokens or
                words['<end>'] in tokens[:-1] or
                row['terminated_with_end'] != (tokens[-1] == words['<end>'])):
            raise ValueError('Invalid saved generated tokens or EOS flag')
        decoded = ' '.join(reverse[t] for t in tokens[1:] if t != words['<end>'])
        if decoded != row['prediction'] or record['reference'] != row['reference'] or not math.isfinite(row['log_probability']):
            raise ValueError('Saved prediction text/reference/score is inconsistent')


def load_source(source, words, tags, hashes):
    parent = source.parent
    config = read_json(source / 'config.json')
    baseline = read_json(parent / 'config.json')
    manifest, parent_manifest = read_json(source / 'manifest.json'), read_json(parent / 'manifest.json')
    if (config != arm_configs(baseline)['lem_off'] or manifest['format'] != FORMAT or manifest['arm'] != 'lem_off' or
            parent_manifest['experiment'] != FORMAT or manifest['tag_loss_weight'] != 0):
        raise ValueError('Requires the explicit lem_off arm of a paired ablation')
    for folder in (source, parent):
        if read_json(folder / 'status.json')['stage'] != 'complete' or read_json(folder / 'data_fingerprints.json') != hashes:
            raise ValueError('Source ablation is incomplete or its dataset has changed')
    checkpoint = torch.load(source / 'last.pt', map_location='cpu', weights_only=True)
    expected = {'format': FORMAT, 'arm': 'lem_off', 'config': config, 'words': words, 'tags': tags,
                'data_fingerprints': hashes, 'epoch': config['epochs'],
                'initial_state_sha256': manifest['initial_state_sha256']}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'Ablation checkpoint {key} mismatch')
    if (read_json(source / 'status.json')['completed_epochs'] != checkpoint['epoch'] or
            read_json(parent / 'status.json')['epochs_per_arm'] != checkpoint['epoch']):
        raise ValueError('Completed epoch does not match the final checkpoint')
    with tarfile.open(parent / 'source.tar.gz') as archive:
        for name in ('model.py', 'losses.py', 'decoding.py', 'data.py', 'evaluation.py', 'ablate_lem.py'):
            path = PACKAGE / name
            member = archive.extractfile(str(path.relative_to(ROOT)))
            if member is None or member.read() != path.read_bytes():
                raise ValueError(f'Source changed since ablation: {name}')
    return config, checkpoint


@torch.no_grad()
def replay_prediction(decoder, memory, prediction, coefficient):
    tokens = torch.tensor([prediction['token_ids']], device=memory.device)
    first, second = decoder(memory, tokens, torch.tensor([tokens.size(1)], device=memory.device))
    probs = mixture_log_probs(first, second, coefficient)
    score = probs[0].gather(1, tokens[0, 1:, None]).double().sum().item()
    same_score(score, prediction['log_probability'])
    return score


@torch.no_grad()
def score_field(decoder, memory, prefix, field, candidates, words, coefficient):
    if decoder.training or memory.size(0) != 1 or field_prefix(prefix, field, words) != prefix:
        raise ValueError('Field probe needs an eval decoder, one image and a value-free prefix')
    if len(set(candidates)) != len(candidates) or any(len(value.split()) != 1 or value not in words for value in candidates):
        raise ValueError('Candidates must be distinct single-token values')
    state = decoder.initial_state(memory)
    for token in prefix:
        first, second, state = decoder.step(memory, torch.tensor([token], device=memory.device), state)
    distributions = head_distributions(first, second, coefficient)
    sequence = {head: {} for head in HEADS}
    for value in candidates:
        first, second, _ = decoder.step(memory, torch.tensor([words[value]], device=memory.device), state)
        following = head_distributions(first, second, coefficient)
        for head in HEADS:
            sequence[head][value] = (distributions[head][words[value]] + following[head][words['.']]).item()
    reverse = {index: word for word, index in words.items()}
    result = {}
    for head, distribution in distributions.items():
        if not torch.isfinite(distribution).all() or not all(math.isfinite(x) for x in sequence[head].values()):
            raise ValueError('Non-finite field probabilities')
        probabilities = {value: distribution[words[value]].exp().item() for value in candidates}
        values, indices = distribution.topk(min(5, len(words)))
        result[head] = {'value_probabilities': probabilities, 'candidate_mass': sum(probabilities.values()),
                        'restricted_value_choice': max(probabilities, key=probabilities.get),
                        'value_period_log_probabilities': sequence[head],
                        'restricted_value_period_choice': max(sequence[head], key=sequence[head].get),
                        'unrestricted_top5': [{'token': reverse[index], 'probability': value}
                                              for index, value in zip(indices.tolist(), values.exp().tolist())]}
    return result


@torch.no_grad()
def diagnose(model, split, records, predictions, candidates, donors, words, config, device):
    model.eval()
    memories, replays = [], []
    for i, record in enumerate(records):
        sample = split[i]
        if sample['id'] != record['id']:
            raise ValueError('VAL image identity mismatch')
        memory = model.encoder(sample['image'].unsqueeze(0).to(device))
        if not torch.isfinite(memory).all():
            raise ValueError('Non-finite visual memory')
        replays.append({'id': record['id'], 'saved': predictions[i]['log_probability'],
                        'replayed': replay_prediction(model.decoder, memory, predictions[i], config['lambda_parallel'])})
        memories.append(memory.cpu())
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f'Encoded/replayed VAL {i + 1}/{len(records)}', flush=True)
    rows = []
    for field in FIELDS:
        cache = {}

        def scored(index, prefix):
            key = (index, tuple(prefix))
            if key not in cache:
                cache[key] = score_field(model.decoder, memories[index].to(device), prefix, field,
                                         candidates[field], words, config['lambda_parallel'])
            return cache[key]

        for i, record in enumerate(records):
            donor = donors[field][i]
            reference = record['fields'][field]
            prefix = field_prefix(predictions[i]['token_ids'], field, words)
            entries = parse_fields(predictions[i]['prediction'])[field]
            actual = entries[0]['value'] if len(entries) == 1 and entries[0]['complete'] else None
            conditions = {'own_image_reference_prefix': scored(i, reference['prefix_ids'])}
            if prefix is not None:
                conditions['own_image_generated_prefix'] = scored(i, prefix)
            if donor is not None:
                conditions['donor_image_reference_prefix'] = scored(donor, reference['prefix_ids'])
                if prefix is not None:
                    conditions['donor_image_generated_prefix'] = scored(donor, prefix)
            rows.append({'id': record['id'], 'field': field, 'reference_value': reference['value'],
                         'reference_risk': record['risk'], 'reference_prefix_ids': reference['prefix_ids'],
                         'generated_prefix_ids': prefix, 'generated_value': actual,
                         'generated_entries': entries, 'generated_value_covered': actual in candidates[field],
                         'donor_id': records[donor]['id'] if donor is not None else None,
                         'donor_value': records[donor]['fields'][field]['value'] if donor is not None else None,
                         'donor_risk': records[donor]['risk'] if donor is not None else None,
                         'donor_stratum': ('same_risk' if records[donor]['risk'] == record['risk'] else 'cross_risk')
                                          if donor is not None else 'unavailable',
                         'memory_difference_rms': (memories[i] - memories[donor]).float().square().mean().sqrt().item()
                                                  if donor is not None else None,
                         'conditions': conditions})
            if (i + 1) % 25 == 0 or i + 1 == len(records):
                print(f'{field}: {i + 1}/{len(records)}', flush=True)
    return rows, replays


def mean(values):
    return sum(values) / len(values) if values else None


def condition_summary(rows, condition, head):
    selected = [row for row in rows if condition in row['conditions']]
    profiles = [row['conditions'][condition][head] for row in selected]
    covered = [(row, profile) for row, profile in zip(selected, profiles) if row['reference_value'] in profile['value_probabilities']]
    return {'samples': len(selected), 'covered_reference_samples': len(covered),
            'restricted_value_counts': dict(Counter(p['restricted_value_choice'] for p in profiles)),
            'restricted_value_matches': sum(p['restricted_value_choice'] == r['reference_value'] for r, p in covered),
            'restricted_value_period_matches': sum(p['restricted_value_period_choice'] == r['reference_value'] for r, p in covered),
            'unrestricted_top1_matches': sum(p['unrestricted_top5'][0]['token'] == r['reference_value'] for r, p in covered),
            'mean_candidate_mass': mean([p['candidate_mass'] for p in profiles]),
            'mean_reference_value_probability': mean([p['value_probabilities'][r['reference_value']] for r, p in covered]),
            'local_value_period_correct_but_saved_generation_wrong': sum(
                p['restricted_value_period_choice'] == r['reference_value'] and r['generated_value'] != r['reference_value']
                for r, p in covered)}


def contrast_summary(rows, before, after, head):
    pairs = [(row, row['conditions'][before][head], row['conditions'][after][head]) for row in rows
             if before in row['conditions'] and after in row['conditions']]
    tv, target_delta, donor_delta = [], [], []
    for row, a, b in pairs:
        first, second = a['value_probabilities'], b['value_probabilities']
        if first.keys() != second.keys():
            raise ValueError('Paired candidate sets must match')
        tv.append(.5 * (sum(abs(second[k] - first[k]) for k in first) + abs(sum(second.values()) - sum(first.values()))))
        if row['reference_value'] in first:
            target_delta.append(second[row['reference_value']] - first[row['reference_value']])
        if row['donor_value'] in first:
            donor_delta.append(second[row['donor_value']] - first[row['donor_value']])
    return {'paired_samples': len(pairs),
            'value_choice_changes': sum(a['restricted_value_choice'] != b['restricted_value_choice'] for _, a, b in pairs),
            'value_period_choice_changes': sum(a['restricted_value_period_choice'] != b['restricted_value_period_choice'] for _, a, b in pairs),
            'recipient_value_matches_before': sum(a['restricted_value_choice'] == r['reference_value'] for r, a, _ in pairs),
            'recipient_value_matches_after': sum(b['restricted_value_choice'] == r['reference_value'] for r, _, b in pairs),
            'donor_value_matches_before': sum(a['restricted_value_choice'] == r['donor_value'] for r, a, _ in pairs),
            'donor_value_matches_after': sum(b['restricted_value_choice'] == r['donor_value'] for r, _, b in pairs),
            'mean_tv_candidates_plus_other': mean(tv), 'reference_probability_samples': len(target_delta),
            'mean_reference_probability_change': mean(target_delta), 'donor_probability_samples': len(donor_delta),
            'mean_donor_probability_change': mean(donor_delta)}


def summarize(rows):
    result = {}
    for field in FIELDS:
        selected = [row for row in rows if row['field'] == field]
        strata = {'all': selected, **{name: [row for row in selected if row['donor_stratum'] == name]
                                      for name in ('same_risk', 'cross_risk')}}
        conditions = sorted({key for row in selected for key in row['conditions']})
        result[field] = {'samples': len(selected), 'missing_generated_prefix': sum(r['generated_prefix_ids'] is None for r in selected),
                         'saved_generation': {'reference_matches': sum(r['generated_value'] == r['reference_value'] for r in selected),
                            'invalid_or_uncovered': sum(not r['generated_value_covered'] for r in selected),
                            'value_counts': dict(Counter(r['generated_value'] or '<invalid>' for r in selected))},
                         'donor_strata': dict(Counter(r['donor_stratum'] for r in selected)),
                         'donor_reuse_counts': dict(Counter(r['donor_id'] for r in selected if r['donor_id'] is not None)),
                         'conditions': {name: {head: condition_summary(selected, name, head) for head in HEADS} for name in conditions},
                         'contrasts': {name: {stratum: {head: contrast_summary(members, before, after, head) for head in HEADS}
                                               for stratum, members in strata.items()}
                                       for name, (before, after) in CONTRASTS.items()}}
    return {'fields': result, 'limits': LIMITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/paper_method/lem_ablation_seed123/lem_off'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Data/checkpoint/prefix checks only; no model construction')
    args = parser.parse_args()
    source = resolve(args.run)
    config = read_json(source / 'config.json')
    directory = resolve(config['data_directory'])
    output = resolve(args.output or source.parent.with_name(source.parent.name + '_lem_off_field_diagnostic'))
    if (output.exists() or any(parent == output or parent in output.parents for parent in (source.parent, directory, PACKAGE)) or
            any((parent / 'status.json').is_file() for parent in output.parents)):
        parser.error('Choose a new output directory outside saved runs, data and code')
    words, tags, splits, hashes = load_dataset(directory)
    try:
        config, checkpoint = load_source(source, words, tags, hashes)
        train, records = (report_records(splits[name], words) for name in ('TRAIN', 'VAL'))
        candidates, coverage = candidate_audit(train, records, words)
        prediction_path = source / f'epoch_{checkpoint["epoch"]:03d}' / 'predictions.json'
        predictions = read_json(prediction_path)
        validate_predictions(predictions, records, words, config['max_decode_steps'])
        donors = pair_donors(records, candidates, config['seed'])
        weights = resolve(config['pretrained_weights'])
        if not weights.is_file():
            parser.error('Cached pretrained weights required; no download is attempted')
        manifest = {'diagnostic': 'paper_core_field_sensitivity_v1', 'source_run': str(source),
                    'checkpoint_file': 'last.pt', 'checkpoint_epoch': checkpoint['epoch'],
                    'checkpoint_sha256': fingerprint(source / 'last.pt'),
                    'prediction_file': str(prediction_path), 'prediction_sha256': fingerprint(prediction_path),
                    'split': 'VAL', 'samples': len(records), 'fields': FIELDS, 'output': str(output),
                    'device': args.device, 'cuda_available': torch.cuda.is_available(), 'torch': str(torch.__version__),
                    'training': False, 'test_inference': False, 'inference_amp': False, 'limits': LIMITS}
        if args.check:
            print(json.dumps({'manifest': manifest, 'coverage': coverage, 'donor_strata': {
                field: dict(Counter('unavailable' if j is None else 'same_risk' if records[i]['risk'] == records[j]['risk']
                                    else 'cross_risk' for i, j in enumerate(indices))) for field, indices in donors.items()}}, indent=2))
            return
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA unavailable. Run on your allocated GPU node')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        seed_all(config['seed'])
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(output / 'manifest.json', manifest)
            write_json(output / 'config.json', config)
            write_json(output / 'data_fingerprints.json', hashes)
            write_json(output / 'candidate_audit.json', coverage)
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'diagnosing'})
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(weights)}, words, tags, device)
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            rows, replays = diagnose(model, splits['VAL'], records, predictions, candidates, donors, words, config, device)
            write_json(output / 'cases.json', rows)
            write_json(output / 'prediction_replay.json', replays)
            write_json(output / 'summary.json', summarize(rows))
            write_json(output / 'status.json', {'stage': 'complete', 'samples': len(records), 'field_cases': len(rows)})
            print(f'Field diagnostic complete: {output}', flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
