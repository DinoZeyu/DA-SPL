"""Read-only VAL risk-conditioning audit; no training or TEST inference."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.decoding import beam_search
from paper_method.evaluation import parse_fields
from paper_method.model import build_model, mixture_log_probs
from paper_method.runner import archive_sources, load_checkpoint, resolve, validate_config

RISKS = ('healthy', 'very healthy', 'moderate risk', 'high risk')
RISK_FIELD = 'glaucoma risk assessment'
HEADS = ('primary', 'secondary', 'mixture')


def risk_group(label):
    if label not in RISKS:
        raise ValueError(f'Unknown reference risk: {label}')
    return 'healthy' if label in ('healthy', 'very healthy') else 'at_risk'


def extract_prefix(tokens, words):
    """Stop after the FIRST risk marker, never include its value or later confidence."""
    marker = [words[word] for word in (RISK_FIELD + ' :').split()]
    positions = [i for i in range(len(tokens) - len(marker) + 1) if tokens[i:i + len(marker)] == marker]
    if not positions:
        return None
    prefix = tokens[:positions[0] + len(marker)]
    if (not prefix or prefix[0] != words['<start>'] or prefix.count(words['<start>']) != 1 or
            words['<end>'] in prefix or words['<pad>'] in prefix):
        raise ValueError('Invalid tokens before the first risk field')
    return prefix


def candidate_tokens(words):
    candidates = {label: [words[word] for word in (label + ' .').split()] for label in RISKS}
    if len({tokens[0] for tokens in candidates.values()}) != len(RISKS):
        raise ValueError('Risk candidates must have distinct first words')
    return candidates


def reference_cases(split, words):
    reverse = {index: word for word, index in words.items()}
    cases = []
    for record, caption, length in zip(split.records, split.captions, split.lengths):
        tokens = caption[:length]
        text = ' '.join(reverse[token] for token in tokens[1:-1])
        entries = parse_fields(text)[RISK_FIELD]
        if len(entries) != 1 or not entries[0]['complete'] or entries[0]['value'] not in RISKS:
            raise ValueError(f"Invalid reference risk: {record['id']}")
        prefix = extract_prefix(tokens, words)
        if prefix is None:
            raise ValueError(f"Reference risk marker missing: {record['id']}")
        cases.append({'id': record['id'], 'reference': text, 'reference_risk': entries[0]['value'],
                      'reference_prefix_ids': prefix})
    return cases


def pair_donors(cases, seed):
    """Round-robin shuffled opposite-group donors, without consulting predictions."""
    groups = {name: [i for i, row in enumerate(cases) if risk_group(row['reference_risk']) == name]
              for name in ('healthy', 'at_risk')}
    if any(not indices for indices in groups.values()):
        raise ValueError('VAL must contain both healthy and at-risk references for this audit')
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    donors = {}
    for group, indices in groups.items():
        opposite = groups['at_risk' if group == 'healthy' else 'healthy']
        for order, index in enumerate(indices):
            donors[index] = opposite[order % len(opposite)]
    return [donors[i] for i in range(len(cases))]


def head_distributions(first, second, coefficient):
    return {'primary': first.float().log_softmax(-1)[0],
            'secondary': second.float().log_softmax(-1)[0],
            'mixture': mixture_log_probs(first, second, coefficient)[0]}


@torch.no_grad()
def score_prefix(decoder, memory, prefix, words, coefficient):
    if decoder.training or memory.size(0) != 1:
        raise ValueError('Audit requires eval mode and one image per decoder call')
    if extract_prefix(prefix, words) != prefix:
        raise ValueError('Prefix must end immediately after the first risk marker')
    candidates = candidate_tokens(words)
    state = decoder.initial_state(memory)
    for token in prefix:
        first, second, state = decoder.step(memory, torch.tensor([token], device=memory.device), state)
    initial = head_distributions(first, second, coefficient)
    sequence = {head: {} for head in HEADS}
    # Every candidate starts from exactly the same prefix state; no argmax feedback.
    for label, tokens in candidates.items():
        current, branch_state = initial, state
        totals = {head: 0. for head in HEADS}
        for position, token in enumerate(tokens):
            for head in HEADS:
                totals[head] += current[head][token].item()
            if position + 1 < len(tokens):
                first, second, branch_state = decoder.step(
                    memory, torch.tensor([token], device=memory.device), branch_state)
                current = head_distributions(first, second, coefficient)
        for head in HEADS:
            sequence[head][label] = totals[head]
    reverse = {index: word for word, index in words.items()}
    result = {}
    for head in HEADS:
        if not torch.isfinite(initial[head]).all() or not all(
                torch.isfinite(torch.tensor(value)) for value in sequence[head].values()):
            raise ValueError('Non-finite diagnostic scores')
        probabilities = {label: initial[head][tokens[0]].exp().item() for label, tokens in candidates.items()}
        top_values, top_indices = initial[head].topk(min(5, len(words)))
        result[head] = {
            'risk_start_probabilities': probabilities,
            'risk_start_total_mass': sum(probabilities.values()),
            'restricted_first_token_choice': max(probabilities, key=probabilities.get),
            'sequence_log_probabilities': sequence[head],
            'restricted_sequence_choice': max(sequence[head], key=sequence[head].get),
            'sequence_token_counts': {label: len(tokens) for label, tokens in candidates.items()},
            'unrestricted_next_token_top5': [{'token': reverse[index], 'probability': value}
                                            for index, value in zip(top_indices.tolist(), top_values.exp().tolist())],
        }
    return result


def strict_generated_risk(text):
    entries = parse_fields(text)[RISK_FIELD]
    valid = len(entries) == 1 and entries[0]['complete'] and entries[0]['value'] in RISKS
    return {'entries': entries, 'valid': valid, 'label': entries[0]['value'] if valid else None}


def average(values):
    return sum(values) / len(values) if values else None


def condition_summary(rows, condition, head):
    selected = [row for row in rows if condition in row['conditions']]
    profiles = [row['conditions'][condition][head] for row in selected]
    return {
        'samples': len(selected),
        'restricted_first_token_choices': dict(Counter(p['restricted_first_token_choice'] for p in profiles)),
        'first_token_confusion': [{'reference': ref, 'candidate': pred, 'count': count}
                                 for (ref, pred), count in sorted(Counter(
                                     (row['reference_risk'], p['restricted_first_token_choice'])
                                     for row, p in zip(selected, profiles)).items())],
        'first_token_reference_matches': sum(p['restricted_first_token_choice'] == row['reference_risk']
                                             for row, p in zip(selected, profiles)),
        'sequence_reference_matches': sum(p['restricted_sequence_choice'] == row['reference_risk']
                                          for row, p in zip(selected, profiles)),
        'mean_risk_start_total_mass': average([p['risk_start_total_mass'] for p in profiles]),
        'mean_reference_risk_start_probability': average([p['risk_start_probabilities'][row['reference_risk']]
                                                         for row, p in zip(selected, profiles)]),
        'mean_reference_sequence_nll': average([-p['sequence_log_probabilities'][row['reference_risk']]
                                                for row, p in zip(selected, profiles)]),
    }


def contrast_summary(rows, before, after, head):
    selected = [row for row in rows if before in row['conditions'] and after in row['conditions']]
    changes, total_variations, target_changes, toward_donor = [], [], [], []
    sequence_changes, matches_before, matches_after = [], [], []
    for row in selected:
        first, second = (row['conditions'][key][head] for key in (before, after))
        a, b = first['risk_start_probabilities'], second['risk_start_probabilities']
        changes.append(first['restricted_first_token_choice'] != second['restricted_first_token_choice'])
        sequence_changes.append(first['restricted_sequence_choice'] != second['restricted_sequence_choice'])
        matches_before.append(first['restricted_first_token_choice'] == row['reference_risk'])
        matches_after.append(second['restricted_first_token_choice'] == row['reference_risk'])
        # TV of five buckets: four risk first-words and all remaining vocabulary.
        total_variations.append(.5 * (sum(abs(b[k] - a[k]) for k in RISKS) + abs(sum(b.values()) - sum(a.values()))))
        target_changes.append(b[row['reference_risk']] - a[row['reference_risk']])
        group = risk_group(row['donor_reference_risk'])
        toward_donor.append(sum(b[k] - a[k] for k in RISKS if risk_group(k) == group))
    return {'paired_samples': len(selected), 'first_token_choice_changes': sum(changes),
            'sequence_choice_changes': sum(sequence_changes),
            'first_token_reference_matches_before': sum(matches_before),
            'first_token_reference_matches_after': sum(matches_after),
            'mean_risk_start_tv_with_other_bucket': average(total_variations),
            'mean_reference_risk_start_probability_change': average(target_changes),
            'mean_absolute_reference_risk_start_probability_change': average([abs(x) for x in target_changes]),
            'mean_donor_group_risk_start_mass_change': average(toward_donor)}


def summarize(rows):
    conditions = sorted({key for row in rows for key in row['conditions']})
    contrasts = {
        'image_swap_reference_prefix': ('own_image_reference_prefix', 'donor_image_reference_prefix'),
        'prefix_swap_own_image': ('own_image_reference_prefix', 'own_image_donor_prefix'),
        'image_swap_generated_prefix': ('own_image_generated_prefix', 'donor_image_generated_prefix'),
        'reference_to_generated_prefix': ('own_image_reference_prefix', 'own_image_generated_prefix'),
    }
    generated = [row['generated_risk'] for row in rows]
    return {
        'split': 'VAL', 'samples': len(rows),
        'reference_risk_counts': dict(Counter(row['reference_risk'] for row in rows)),
        'donor_reuse_counts': dict(Counter(row['donor_id'] for row in rows)),
        'generation': {
            'unique_reports': len({row['generation']['prediction'] for row in rows}),
            'unfinished': sum(not row['generation']['terminated_with_end'] for row in rows),
            'invalid_or_missing_risk_values': sum(not item['valid'] for item in generated),
            'missing_risk_prefix': sum(row['generated_prefix_ids'] is None for row in rows),
            'strict_risk_counts': dict(Counter(item['label'] if item['valid'] else '<invalid>' for item in generated)),
            'strict_reference_risk_matches': sum(item['label'] == row['reference_risk'] for item, row in zip(generated, rows)),
            'confusion': [{'reference': ref, 'generated': pred, 'count': count}
                          for (ref, pred), count in sorted(Counter(
                              (row['reference_risk'], item['label'] if item['valid'] else '<invalid>')
                              for row, item in zip(rows, generated)).items())],
        },
        'conditions': {name: {head: condition_summary(rows, name, head) for head in HEADS} for name in conditions},
        'contrasts': {name: {head: contrast_summary(rows, before, after, head) for head in HEADS}
                      for name, (before, after) in contrasts.items()},
        'interpretation_limits': [
            'Restricted risk scores are diagnostic rankings, not free-generation accuracy or clinical confidence.',
            'First-word probabilities use the full vocabulary; four risk words need not sum to one.',
            'Whole-label log probabilities include the terminating period and are not length-normalized.',
            'Opposite-group image/text pairs can be out of distribution; donors are reused, not independent trials.',
            'Whole-prefix swaps also change findings, phrasing and length; no single-factor clinical causality claim.',
            'Missing generated prefixes are excluded from paired contrasts, never replaced with reference text.',
            'VAL was used for checkpoint selection. This is debugging, not independent validation.',
        ],
    }


@torch.no_grad()
def diagnose(model, split, words, config, device, cases, donors):
    model.eval()
    memories = []
    for index in range(len(split)):
        row = split[index]
        if row['id'] != cases[index]['id']:
            raise ValueError('Image/metadata order mismatch')
        memory = model.encoder(row['image'].unsqueeze(0).to(device))
        if not torch.isfinite(memory).all():
            raise ValueError('Non-finite encoded image')
        memories.append(memory.cpu())
        if (index + 1) % 10 == 0 or index + 1 == len(split):
            print(f'Encoded VAL {index + 1}/{len(split)}', flush=True)
    rows = []
    for index, case in enumerate(cases):
        donor = donors[index]
        own, other = memories[index].to(device), memories[donor].to(device)
        prefix, donor_prefix = case['reference_prefix_ids'], cases[donor]['reference_prefix_ids']
        conditions = {}
        for name, memory, text in (
                ('own_image_reference_prefix', own, prefix),
                ('donor_image_reference_prefix', other, prefix),
                ('own_image_donor_prefix', own, donor_prefix)):
            conditions[name] = score_prefix(model.decoder, memory, text, words, config['lambda_parallel'])
        generation = beam_search(model.decoder, own, words, config['lambda_parallel'],
                                 config['beam_size'], config['max_decode_steps'])
        generated_prefix = extract_prefix(generation['token_ids'], words)
        if generated_prefix is not None:
            for name, memory in (('own_image_generated_prefix', own), ('donor_image_generated_prefix', other)):
                conditions[name] = score_prefix(model.decoder, memory, generated_prefix, words, config['lambda_parallel'])
        delta = own.float() - other.float()
        rows.append({**case, 'donor_id': cases[donor]['id'],
                     'donor_reference_risk': cases[donor]['reference_risk'],
                     'donor_prefix_ids': donor_prefix, 'generated_prefix_ids': generated_prefix,
                     'memory_difference_rms': delta.square().mean().sqrt().item(),
                     'conditions': conditions, 'generation': generation,
                     'generated_risk': strict_generated_risk(generation['prediction'])})
        print(f"Audited VAL {index + 1}/{len(cases)}: {case['id']}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/paper_method/image_core_seed123'))
    parser.add_argument('--output', type=Path, help='New sibling directory by default; never overwrite')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Validate checkpoint/data/pairs only; no model construction')
    args = parser.parse_args()
    source = resolve(args.run)
    output = resolve(args.output or source.with_name(source.name + '_risk_diagnostic'))
    config = read_json(source / 'config.json')
    validate_config(config)
    directory = resolve(config['data_directory'])
    if output.exists() or any(parent == output or parent in output.parents for parent in
                              (source, directory, Path(__file__).resolve().parent)):
        parser.error('Choose a new output directory outside the source run, data and code')
    words, tags, splits, hashes = load_dataset(directory)
    try:
        checkpoint = load_checkpoint(source, config, words, tags, hashes)
        candidate_tokens(words)
        cases = reference_cases(splits['VAL'], words)
        donors = pair_donors(cases, config['seed'])
        manifest = {
            'diagnostic': 'paper_core_risk_conditioning_v1', 'split': 'VAL', 'samples': len(cases),
            'reference_risk_counts': dict(Counter(row['reference_risk'] for row in cases)),
            'checkpoint_epoch': checkpoint['epoch'], 'source_run': str(source),
            'checkpoint_sha256': fingerprint(source / 'best.pt'), 'output': str(output),
            'device': args.device, 'cuda_available': torch.cuda.is_available(), 'torch': str(torch.__version__),
            'seed': config['seed'], 'inference_amp': False, 'batch_size': 1,
            'training': False, 'test_inference': False, 'paper_comparable': False,
            'prefix_boundary': 'Immediately after first glaucoma risk assessment colon; no risk value/confidence',
        }
        if args.check:
            print(json.dumps(manifest, indent=2), flush=True)
            return
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA unavailable. Run on your allocated GPU node')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(output / 'manifest.json', manifest)
            write_json(output / 'config.json', config)
            write_json(output / 'data_fingerprints.json', hashes)
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'diagnosing'})
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(resolve(config['pretrained_weights']))}, words, tags, device)
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            rows = diagnose(model, splits['VAL'], words, config, device, cases, donors)
            write_json(output / 'cases.json', rows)
            summary = summarize(rows)
            write_json(output / 'summary.json', summary)
            write_json(output / 'status.json', {'stage': 'complete', 'samples': len(rows)})
            print(json.dumps({'generation': summary['generation'],
                              'mixture_contrasts': {name: heads['mixture'] for name, heads in summary['contrasts'].items()}},
                             indent=2), flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
