"""Measure CE/LEM gradient conflict at report logits; never update parameters."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import string
import sys

import torch
from torch.nn import functional as F

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.diagnose_endings import reference_records
from paper_method.losses import objective
from paper_method.model import build_model, mixture_log_probs
from paper_method.runner import archive_sources, load_checkpoint, resolve, validate_config

HEADS = ('primary', 'secondary')
SITES = ('terminal_period', 'eos')
LIMITS = [
    'Teacher-forced TRAIN references at the saved checkpoint, not free-generation accuracy.',
    'Gradients are partial derivatives with respect to report logits, holding all other logits fixed.',
    'A positive target-minus-colon gradient means logit-space descent would reduce that margin.',
    'These are not parameter-space gradients, actual Adam updates, or historical training replay.',
    'CE and LEM use the saved loss weights and batch reductions; loss magnitudes are not gradient strengths.',
    'Native saved-order batches, eval FP32, native LEM LSTM backward; no AMP or dropout.',
    'No optimizer, parameter change, output rewriting, VAL inference or TEST inference.',
]


def token_audit(records, words):
    reverse = {index: word for word, index in words.items()}
    counts = Counter(reverse[token] for row in records for token in row['tokens'][1:])
    punctuation = {word: count for word, count in counts.items()
                   if word and all(char in string.punctuation for char in word)}
    total = sum(counts.values())
    return {'reports': len(records), 'supervised_tokens': total, 'token_counts': dict(counts),
            'punctuation_counts': punctuation, 'punctuation_fraction': sum(punctuation.values()) / total,
            'eos_fraction': counts['<end>'] / total,
            'note': 'Token frequency, not the fraction of loss or gradient magnitude.'}


def loss_gradients(model, first, second, captions, lengths, tags, coefficient, tag_weight):
    if model.training or first.shape != second.shape:
        raise ValueError('Expected matching branch logits from an eval-mode model')
    leaves = tuple(value.detach().float().requires_grad_(True) for value in (first, second))
    # cuDNN eval LSTM forward cannot be used for backward; native LSTM has the same equations.
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        distribution = mixture_log_probs(*leaves, coefficient).exp()
        embeddings = distribution @ model.decoder.embedding.weight.detach().float()
        tag_logits = model.label_enhancement(embeddings, lengths.reshape(-1) - 1)
        outputs = {'primary_logits': leaves[0], 'secondary_logits': leaves[1], 'tag_logits': tag_logits}
        total, parts, count = objective(outputs, captions, lengths, tags, coefficient, tag_weight)
        lem = tag_weight * F.binary_cross_entropy_with_logits(tag_logits.float(), tags.float())
        if not torch.isfinite(total):
            raise ValueError('Non-finite diagnostic objective')
        total_grads = torch.autograd.grad(total, leaves, retain_graph=True)
        lem_grads = torch.autograd.grad(lem, leaves)
    gradients = {}
    for head, all_grad, label_grad in zip(HEADS, total_grads, lem_grads):
        gradients[head] = {'ce': all_grad - label_grad, 'lem': label_grad, 'total': all_grad}
        if any(not torch.isfinite(value).all() for value in gradients[head].values()):
            raise ValueError('Non-finite diagnostic gradients')
    values = {key: value.item() for key, value in parts.items()}
    values.update(total=total.item(), weighted_lem=lem.item(), supervised_tokens=count)
    return gradients, values


def site_statistics(scores, gradients, target, colon):
    probabilities = scores.float().softmax(-1)
    ce, lem = gradients['ce'], gradients['lem']
    ce_norm, lem_norm = ce.norm().item(), lem.norm().item()
    margins = {name: (value[target] - value[colon]).item() for name, value in gradients.items()}
    return {'target_probability': probabilities[target].item(), 'colon_probability': probabilities[colon].item(),
            'target_is_top1': scores.argmax().item() == target, 'colon_is_top1': scores.argmax().item() == colon,
            'weighted_ce_gradient_norm': ce_norm, 'weighted_lem_gradient_norm': lem_norm,
            'lem_to_ce_gradient_norm_ratio': lem_norm / ce_norm if ce_norm else None,
            'ce_lem_gradient_cosine': (ce @ lem).item() / (ce_norm * lem_norm) if ce_norm and lem_norm else None,
            'target_minus_colon_gradients': margins,
            'lem_opposes_ce_margin': margins['ce'] < 0 < margins['lem'],
            'total_opposes_ce_margin': margins['ce'] < 0 < margins['total']}


def diagnose(model, split, records, words, config, device):
    model.eval()
    model.requires_grad_(False)
    rows, batches = [], []
    for start in range(0, len(records), config['batch_size']):
        group = records[start:start + config['batch_size']]
        samples = [split[start + i] for i in range(len(group))]
        if [sample['id'] for sample in samples] != [record['id'] for record in group]:
            raise ValueError('Image/target identity mismatch')
        images = torch.stack([sample['image'] for sample in samples]).to(device)
        captions = torch.stack([sample['caption'] for sample in samples]).to(device)
        lengths = torch.stack([sample['length'] for sample in samples]).to(device)
        tags = torch.stack([sample['tags'] for sample in samples]).to(device)
        with torch.no_grad():
            first, second = model.decoder(model.encoder(images), captions, lengths)
        gradients, losses = loss_gradients(model, first, second, captions, lengths, tags,
                                           config['lambda_parallel'], config['tag_loss_weight'])
        batch_ids = [record['id'] for record in group]
        batches.append({'ids': batch_ids, 'losses': losses})
        for i, record in enumerate(group):
            sites = {}
            for site in SITES:
                position = record['sites'][site]
                target = record['tokens'][position]
                sites[site] = {head: site_statistics(scores[i, position - 1],
                    {key: value[i, position - 1] for key, value in gradients[head].items()}, target, words[':'])
                    for head, scores in zip(HEADS, (first, second))}
            rows.append({'id': record['id'], 'reference_risk': record['risk'],
                         'reference_confidence': record['confidence'], 'batch_index': len(batches) - 1, 'sites': sites})
        print(f'Gradient audit {len(rows)}/{len(records)} TRAIN references', flush=True)
    return rows, batches


def summarize(rows):
    groups = {'all': rows}
    for risk in sorted({row['reference_risk'] for row in rows}):
        groups[risk] = [row for row in rows if row['reference_risk'] == risk]
    for risk, confidence in sorted({(row['reference_risk'], row['reference_confidence']) for row in rows}):
        groups[f'{risk}; confidence={confidence}'] = [row for row in rows if
            (row['reference_risk'], row['reference_confidence']) == (risk, confidence)]
    result = {}
    for key, group in groups.items():
        result[key] = {'samples': len(group), 'sites': {}}
        for site in SITES:
            result[key]['sites'][site] = {}
            for head in HEADS:
                values = [row['sites'][site][head] for row in group]
                stats = {name + '_count': sum(value[name] for value in values) for name in
                         ('target_is_top1', 'colon_is_top1', 'lem_opposes_ce_margin', 'total_opposes_ce_margin')}
                for name in ('target_probability', 'colon_probability', 'weighted_ce_gradient_norm',
                             'weighted_lem_gradient_norm', 'lem_to_ce_gradient_norm_ratio', 'ce_lem_gradient_cosine'):
                    valid = [value[name] for value in values if value[name] is not None]
                    stats['mean_' + name] = sum(valid) / len(valid) if valid else None
                stats['mean_target_minus_colon_gradients'] = {
                    name: sum(value['target_minus_colon_gradients'][name] for value in values) / len(values)
                    for name in ('ce', 'lem', 'total')}
                result[key]['sites'][site][head] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/paper_method/image_core_seed123'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Validate checkpoint and targets; no model construction')
    args = parser.parse_args()
    source = resolve(args.run)
    output = resolve(args.output or source.with_name(source.name + '_loss_conflict'))
    config = read_json(source / 'config.json')
    validate_config(config)
    directory = resolve(config['data_directory'])
    if output.exists() or any(parent == output or parent in output.parents for parent in
                              (source, directory, Path(__file__).resolve().parent)):
        parser.error('Choose a new output directory outside saved runs, data and code')
    words, tags, splits, hashes = load_dataset(directory)
    try:
        checkpoint = load_checkpoint(source, config, words, tags, hashes)
        records = reference_records(splits['TRAIN'], words)
        counts = token_audit(records, words)
        manifest = {'diagnostic': 'paper_core_loss_conflict_v1', 'source_run': str(source),
                    'checkpoint_sha256': fingerprint(source / 'best.pt'), 'checkpoint_epoch': checkpoint['epoch'],
                    'split': 'TRAIN', 'samples': len(records), 'batch_size': config['batch_size'],
                    'device': args.device, 'torch': str(torch.__version__), 'cuda_available': torch.cuda.is_available(),
                    'parameter_updates': False, 'inference_amp': False, 'output': str(output), 'limits': LIMITS}
        if args.check:
            print(json.dumps({'manifest': manifest, 'token_audit': counts}, indent=2), flush=True)
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
            write_json(output / 'token_audit.json', counts)
            write_json(output / 'data_fingerprints.json', hashes)
            write_json(output / 'config.json', config)
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'diagnosing'})
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(resolve(config['pretrained_weights']))}, words, tags, device)
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            rows, batches = diagnose(model, splits['TRAIN'], records, words, config, device)
            write_json(output / 'cases.json', rows)
            write_json(output / 'batches.json', batches)
            write_json(output / 'summary.json', {'groups': summarize(rows), 'limits': LIMITS})
            write_json(output / 'status.json', {'stage': 'complete', 'samples': len(rows)})
            print(f'Loss conflict diagnostic complete: {output}', flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
