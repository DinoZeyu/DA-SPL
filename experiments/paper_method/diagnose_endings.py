"""Audit reference endings, branch learning, batch effects and controlled prefixes."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.diagnose_risk import HEADS, RISKS, extract_prefix
from paper_method.model import build_model, mixture_log_probs
from paper_method.runner import archive_sources, load_checkpoint, resolve, validate_config

SITES = ('risk_period', 'confidence_value', 'terminal_period', 'eos')


def ending_record(sample_id, caption, length, words):
    tokens = caption[:length]
    if (len(tokens) != length or length < 2 or tokens[0] != words['<start>'] or
            tokens[-2:] != [words['.'], words['<end>']] or tokens.count(words['<end>']) != 1 or
            tokens.count(words['<start>']) != 1 or words['<pad>'] in tokens or
            any(token != words['<pad>'] for token in caption[length:])):
        raise ValueError(f'{sample_id}: invalid START/period/EOS/PAD placement')
    prefix = extract_prefix(tokens, words)
    marker = [words[token] for token in 'confidence level :'.split()]
    positions = [i for i in range(len(tokens) - len(marker) + 1) if tokens[i:i + len(marker)] == marker]
    if prefix is None or len(positions) != 1:
        raise ValueError(f'{sample_id}: missing risk or non-unique confidence marker')
    confidence = positions[0]
    if confidence <= len(prefix) or tokens[confidence - 1] != words['.'] or confidence + len(marker) != length - 3:
        raise ValueError(f'{sample_id}: expected a single confidence token followed by period and EOS')
    reverse = {index: word for word, index in words.items()}
    risk = ' '.join(reverse[token] for token in tokens[len(prefix):confidence - 1])
    if risk not in RISKS:
        raise ValueError(f'{sample_id}: malformed risk value')
    return {'id': sample_id, 'tokens': tokens, 'risk': risk, 'confidence': reverse[tokens[length - 3]],
            'risk_prefix_ids': prefix, 'sites': {'risk_period': confidence - 1,
             'confidence_value': length - 3, 'terminal_period': length - 2, 'eos': length - 1}}


def reference_records(split, words):
    return [ending_record(row['id'], caption, length, words)
            for row, caption, length in zip(split.records, split.captions, split.lengths)]


def data_audit(records, words):
    reverse = {index: word for word, index in words.items()}
    by_risk, lengths, tails = defaultdict(Counter), defaultdict(list), Counter()
    for row in records:
        by_risk[row['risk']][row['confidence']] += 1
        lengths[row['risk']].append(len(row['tokens']) - 1)
        tail = row['tokens'][row['sites']['confidence_value']:]
        tails[' '.join(reverse[token] for token in tail)] += 1
    count = sum(len(row['tokens']) - 1 for row in records)
    return {'samples': len(records), 'risk_confidence_counts': dict(by_risk), 'terminal_suffix_counts': dict(tails),
            'mean_target_length_by_risk': {key: sum(values) / len(values) for key, values in lengths.items()},
            'supervised_token_count': count, 'eos_target_count': len(records),
            'eos_fraction_of_token_ce_targets': len(records) / count if count else None,
            'all_endings_have_one_period_then_eos': True,
            'note': 'EOS fraction is a target-token count, not its share of the full CE plus LEM objective.'}


def profile(log_probs, target, words):
    if not torch.isfinite(log_probs).all():
        raise ValueError('Non-finite ending probabilities')
    reverse = {index: word for word, index in words.items()}
    values, indices = log_probs.topk(min(5, len(words)))
    return {'target_token': reverse[target], 'target_log_probability': log_probs[target].item(),
            'target_probability': log_probs[target].exp().item(),
            'target_rank_lower_bound': 1 + (log_probs > log_probs[target]).sum().item(),
            'target_is_top1': log_probs.argmax().item() == target,
            'eos_probability': log_probs[words['<end>']].exp().item(),
            'colon_probability': log_probs[words[':']].exp().item(),
            'top5': [{'token': reverse[index], 'probability': value}
                     for index, value in zip(indices.tolist(), values.exp().tolist())]}


@torch.no_grad()
def score_batch(decoder, memories, records, words, coefficient, device):
    if decoder.training or len(memories) != len(records) or not records:
        raise ValueError('Need nonempty, aligned memories/reports in eval mode')
    lengths = torch.tensor([len(row['tokens']) for row in records], device=device)
    captions = torch.full((len(records), int(lengths.max())), words['<pad>'], dtype=torch.long, device=device)
    for i, row in enumerate(records):
        captions[i, :len(row['tokens'])] = torch.tensor(row['tokens'], device=device)
    first, second = decoder(torch.cat(memories).to(device), captions, lengths)
    distributions = {'primary': first.float().log_softmax(-1).cpu(),
                     'secondary': second.float().log_softmax(-1).cpu(),
                     'mixture': mixture_log_probs(first, second, coefficient).cpu()}
    results = []
    for index, row in enumerate(records):
        sites = {}
        for site, position in row['sites'].items():
            if not 1 <= position < len(row['tokens']):
                raise ValueError('Site must be an unmasked next-token target')
            sites[site] = {head: profile(scores[index, position - 1], row['tokens'][position], words)
                           for head, scores in distributions.items()}
        results.append({'sites': sites, 'batch_ids': [item['id'] for item in records],
                        'active_ids_at_site': {site: [other['id'] for other in records if len(other['tokens']) > position]
                                               for site, position in row['sites'].items()}})
    return results


def batch_orders(size, seed):
    native, shuffled = list(range(size)), list(range(size))
    random.Random(seed).shuffle(shuffled)
    return {'reference_native_batch': native, 'reference_shuffled_batch': shuffled}


def controlled_pair(reference, traced, words):
    healthy = traced['constrained_healthy_continuation']
    if not healthy['terminated_with_end']:
        raise ValueError('Controlled ending comparison requires a completed explored continuation')
    generated = ending_record(reference['id'], healthy['token_ids'], len(healthy['token_ids']), words)
    if generated['risk_prefix_ids'] != traced['generated_prefix_ids']:
        raise ValueError('Traced prefix does not match healthy continuation')
    suffix = generated['tokens'][len(generated['risk_prefix_ids']):]
    controlled_tokens = reference['risk_prefix_ids'] + suffix
    controlled = ending_record(reference['id'], controlled_tokens, len(controlled_tokens), words)
    return generated, controlled


def verify_trace_replay(scored, record, traced):
    old = traced['constrained_healthy_continuation']['scores']['token_scores']
    for site, position in record['sites'].items():
        saved = old[position - 1]
        if saved['token_id'] != record['tokens'][position]:
            raise ValueError('Traced token position mismatch')
        for head in HEADS:
            if not math.isclose(scored['sites'][site][head]['target_log_probability'],
                                saved['head_log_probabilities'][head], rel_tol=1e-6, abs_tol=1e-4):
                raise ValueError(f'Trace score replay mismatch: {record["id"]} {site} {head}')


@torch.no_grad()
def diagnose_split(model, split, records, words, config, device, focus):
    model.eval()
    memories = []
    for i, record in enumerate(records):
        sample = split[i]
        if sample['id'] != record['id']:
            raise ValueError('Image/metadata identity mismatch')
        memory = model.encoder(sample['image'].unsqueeze(0).to(device))
        if not torch.isfinite(memory).all():
            raise ValueError('Non-finite image features')
        memories.append(memory.cpu())
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f'Encoded {i + 1}/{len(records)} images', flush=True)
    rows = [{'id': record['id'], 'reference_risk': record['risk'], 'reference_confidence': record['confidence'],
             'target_tokens': record['tokens'], 'target_sites': record['sites'], 'conditions': {}} for record in records]
    for i, record in enumerate(records):
        rows[i]['conditions']['reference_single'] = score_batch(
            model.decoder, [memories[i]], [record], words, config['lambda_parallel'], device)[0]
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f'Scored single-image references {i + 1}/{len(records)}', flush=True)
    for mode, order in batch_orders(len(records), config['seed']).items():
        for start in range(0, len(order), config['batch_size']):
            group = order[start:start + config['batch_size']]
            scores = score_batch(model.decoder, [memories[i] for i in group], [records[i] for i in group],
                                 words, config['lambda_parallel'], device)
            for i, score in zip(group, scores):
                rows[i]['conditions'][mode] = score
        print(f'Completed {mode}', flush=True)
    for i, record in enumerate(records):
        if record['id'] not in focus:
            continue
        generated, controlled = controlled_pair(record, focus[record['id']], words)
        for mode, target in (('generated_body_fixed_suffix', generated), ('reference_body_fixed_suffix', controlled)):
            score = score_batch(model.decoder, [memories[i]], [target], words, config['lambda_parallel'], device)[0]
            if mode == 'generated_body_fixed_suffix':
                verify_trace_replay(score, target, focus[record['id']])
            rows[i]['conditions'][mode] = score
        rows[i]['controlled_tokens'] = {'generated_body': generated['tokens'], 'reference_body': controlled['tokens']}
    return rows


def mean(values):
    return sum(values) / len(values) if values else None


def statistics(rows, condition):
    chosen = [row for row in rows if condition in row['conditions']]
    return {'samples': len(chosen), 'sites': {site: {head: {
                'mean_target_probability': mean([row['conditions'][condition]['sites'][site][head]['target_probability'] for row in chosen]),
                'mean_target_nll': mean([-row['conditions'][condition]['sites'][site][head]['target_log_probability'] for row in chosen]),
                'target_top1_count': sum(row['conditions'][condition]['sites'][site][head]['target_is_top1'] for row in chosen),
                'mean_colon_probability': mean([row['conditions'][condition]['sites'][site][head]['colon_probability'] for row in chosen]),
            } for head in HEADS} for site in SITES}}


def contrasts(rows, before, after):
    chosen = [row for row in rows if before in row['conditions'] and after in row['conditions']]
    sites = {}
    for site in SITES:
        sites[site] = {}
        for head in HEADS:
            a = [row['conditions'][before]['sites'][site][head] for row in chosen]
            b = [row['conditions'][after]['sites'][site][head] for row in chosen]
            if any(first['target_token'] != second['target_token'] for first, second in zip(a, b)):
                raise ValueError('Paired comparison must keep the target token unchanged')
            delta = [second['target_probability'] - first['target_probability'] for first, second in zip(a, b)]
            sites[site][head] = {'mean_target_probability_change': mean(delta),
                                 'mean_absolute_target_probability_change': mean([abs(value) for value in delta]),
                                 'top1_before': sum(item['target_is_top1'] for item in a),
                                 'top1_after': sum(item['target_is_top1'] for item in b)}
    return {'paired_samples': len(chosen), 'before': before, 'after': after, 'sites': sites}


def summarize(rows):
    groups = {'all': rows}
    groups.update({risk: [row for row in rows if row['reference_risk'] == risk] for risk in RISKS})
    for risk, confidence in sorted({(row['reference_risk'], row['reference_confidence']) for row in rows}):
        groups[f'{risk}; confidence={confidence}'] = [row for row in rows if
                                                     (row['reference_risk'], row['reference_confidence']) == (risk, confidence)]
    conditions = sorted({key for row in rows for key in row['conditions']})
    pairs = {'batch_to_single': ('reference_native_batch', 'reference_single'),
             'batch_composition': ('reference_native_batch', 'reference_shuffled_batch'),
             'reference_to_generated_body_same_suffix': ('reference_body_fixed_suffix', 'generated_body_fixed_suffix')}
    return {'samples': len(rows),
            'conditions': {mode: {group: statistics(members, mode) for group, members in groups.items()} for mode in conditions},
            'contrasts': {name: {group: contrasts(members, before, after) for group, members in groups.items()}
                          for name, (before, after) in pairs.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/paper_method/image_core_seed123'))
    parser.add_argument('--trace', type=Path, help='Defaults to the source run sibling ending in _risk_beam_trace')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Validate targets, trace and checkpoint; no model construction')
    args = parser.parse_args()
    source = resolve(args.run)
    traced = resolve(args.trace or source.with_name(source.name + '_risk_beam_trace'))
    output = resolve(args.output or source.with_name(source.name + '_ending_diagnostic'))
    config = read_json(source / 'config.json')
    validate_config(config)
    directory = resolve(config['data_directory'])
    if output.exists() or any(parent == output or parent in output.parents for parent in
                              (source, traced, directory, Path(__file__).resolve().parent)):
        parser.error('Choose a new output directory outside saved runs, data and code')
    trace_manifest = read_json(traced / 'manifest.json')
    checkpoint_hash = fingerprint(source / 'best.pt')
    if (read_json(traced / 'status.json')['stage'] != 'complete' or trace_manifest['split'] != 'VAL' or
            trace_manifest['diagnostic'] != 'paper_core_risk_beam_trace_v1' or
            trace_manifest['checkpoint_sha256'] != checkpoint_hash or resolve(trace_manifest['source_run']) != source):
        parser.error('Requires a completed VAL beam trace from the same run/checkpoint')
    words, tags, splits, hashes = load_dataset(directory)
    try:
        checkpoint = load_checkpoint(source, config, words, tags, hashes)
        records = {name: reference_records(splits[name], words) for name in ('TRAIN', 'VAL')}
        audits = {name: data_audit(items, words) for name, items in records.items()}
        trace_rows = read_json(traced / 'cases.json')
        focus = {row['id']: row for row in trace_rows}
        if not focus or len(focus) != len(trace_rows) or set(focus) != set(trace_manifest['selected_ids']):
            parser.error('Trace IDs are empty, duplicated or inconsistent with its manifest')
        val = {row['id']: row for row in records['VAL']}
        if not set(focus) <= set(val):
            parser.error('Focused cases must belong only to VAL')
        for key, row in focus.items():
            controlled_pair(val[key], row, words)
        manifest = {'diagnostic': 'paper_core_ending_learning_v1', 'source_run': str(source), 'source_trace': str(traced),
                    'checkpoint_sha256': checkpoint_hash, 'trace_cases_sha256': fingerprint(traced / 'cases.json'),
                    'checkpoint_epoch': checkpoint['epoch'], 'samples': {key: len(value) for key, value in records.items()},
                    'focused_val_cases': sorted(focus), 'batch_size': config['batch_size'], 'seed': config['seed'],
                    'device': args.device, 'cuda_available': torch.cuda.is_available(), 'torch': str(torch.__version__),
                    'training': False, 'test_inference': False, 'inference_amp': False, 'output': str(output),
                    'batch_probe': 'Saved-order and seeded-shuffle batches; NOT replay of historical training batches',
                    'controlled_probe': 'Identical image and fixed risk/confidence/EOS suffix; only the preceding body changes'}
        if args.check:
            print(json.dumps({'manifest': manifest, 'data_audit': audits}, indent=2), flush=True)
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
            write_json(output / 'data_audit.json', audits)
            write_json(output / 'data_fingerprints.json', hashes)
            write_json(output / 'config.json', config)
            archive_sources(output)
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(resolve(config['pretrained_weights']))}, words, tags, device)
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            results = {}
            for name in ('TRAIN', 'VAL'):
                write_json(output / 'status.json', {'stage': 'diagnosing', 'split': name})
                rows = diagnose_split(model, splits[name], records[name], words, config, device, focus if name == 'VAL' else {})
                write_json(output / f'{name.lower()}_cases.json', rows)
                results[name] = summarize(rows)
            write_json(output / 'summary.json', {'splits': results, 'limits': [
                'All scores are teacher-forced target probabilities, not free-generation or clinical accuracy.',
                'Frozen encoder features are reused. Batch probes change only decoder batch composition, in eval FP32.',
                'Two fixed batch orders probe dependence; they are not historical training batch replay.',
                'Only selected VAL failures receive the controlled-body comparison; the entire suffix is held fixed.',
                'Data counts, branch differences and loss dilution are observations, not proof of a unique training cause.',
                'No training, new beam search, target rewriting or TEST inference occurs.',
            ]})
            write_json(output / 'status.json', {'stage': 'complete', 'samples': manifest['samples']})
            print(f'Ending diagnostic complete: {output}', flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
