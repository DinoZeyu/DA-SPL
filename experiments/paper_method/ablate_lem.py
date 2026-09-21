"""Paired fresh-start LEM-on/off training with per-epoch VAL audits, never TEST inference."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.decoding import beam_search
from paper_method.diagnose_endings import reference_records, score_batch, summarize as summarize_endings
from paper_method.diagnose_risk import risk_group, strict_generated_risk
from paper_method.evaluation import structural_checks
from paper_method.model import build_model
from paper_method.runner import PACKAGE, ROOT, archive_sources, resolve, run_epoch, save_checkpoint, validate_config

FORMAT = 'da_spl_paper_lem_ablation_v1'
ARMS = ('lem_on', 'lem_off')
LIMITS = [
    'Fresh matched runs, not continuation or exact replay of the earlier five-epoch checkpoint.',
    'Only tag_loss_weight changes: original positive weight versus zero; architecture and decoding stay fixed.',
    'The off arm still computes LEM for logging but receives zero LEM loss gradient; no other module is removed.',
    'Both arms reload identical model tensors, use fresh Adam/scaler state and identical planned training batches.',
    'Per-epoch RNG resets isolate training from validation; order and initialization hashes are checked.',
    'Compare matched epochs, especially the final epoch; the arms have different total objectives.',
    'Best checkpoints use the same VAL primary_ce + lambda * secondary_ce criterion, excluding LEM.',
    'VAL generation and single-image teacher-forced endings use eval FP32; no TEST inference or TEST-based selection.',
    'Risk/field agreement is with reconstructed references, not independently established clinical accuracy.',
    'A single-seed short ablation tests this reconstruction, not the published multimodal method or a unique cause.',
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_digest(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def training_orders(size, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(size, generator=generator).tolist() for _ in range(epochs)]


def arm_configs(config):
    validate_config(config)
    return {'lem_on': dict(config), 'lem_off': {**config, 'tag_loss_weight': 0.}}


class OrderedBatches:
    """Check IDs actually delivered to the unchanged training loop against the paired plan."""

    def __init__(self, split, order, batch_size, seed):
        if sorted(order) != list(range(len(split))):
            raise ValueError('Each epoch must visit every TRAIN sample exactly once')
        self.split, self.order, self.batch_size = split, order, batch_size
        self.loader = DataLoader(split, batch_size=batch_size, sampler=order, num_workers=0,
                                 generator=torch.Generator().manual_seed(seed))
        self.seen = []

    def __iter__(self):
        self.seen = []
        for batch in self.loader:
            start = len(self.seen)
            expected = [self.split.records[i]['id'] for i in self.order[start:start + self.batch_size]]
            if list(batch['id']) != expected:
                raise ValueError('Training batch IDs differ from the shared plan')
            self.seen.extend(batch['id'])
            yield batch
        if len(self.seen) != len(self.order):
            raise ValueError('Training epoch did not consume all planned samples')


def generation_summary(predictions, records):
    if [row['id'] for row in predictions] != [row['id'] for row in records]:
        raise ValueError('VAL predictions and reference records are misaligned')
    generated = [strict_generated_risk(row['prediction']) for row in predictions]
    correct = sum(item['label'] == record['risk'] for item, record in zip(generated, records))
    healthy = [i for i, row in enumerate(records) if risk_group(row['risk']) == 'healthy']
    counts = Counter(row['prediction'] for row in predictions)
    return {'samples': len(records), 'risk_matches': correct, 'risk_match_rate': correct / len(records),
            'invalid_risk': sum(not item['valid'] for item in generated),
            'reference_risk_counts': dict(Counter(row['risk'] for row in records)),
            'generated_risk_counts': dict(Counter(item['label'] if item['valid'] else '<invalid>' for item in generated)),
            'confusion': [{'reference': ref, 'generated': pred, 'count': count} for (ref, pred), count in sorted(Counter(
                (row['risk'], item['label'] if item['valid'] else '<invalid>')
                for row, item in zip(records, generated)).items())],
            'healthy_reference_samples': len(healthy),
            'healthy_to_high_risk': sum(generated[i]['label'] == 'high risk' for i in healthy),
            'healthy_to_invalid_risk': sum(not generated[i]['valid'] for i in healthy),
            'unique_reports': len(counts), 'largest_identical_group': max(counts.values()),
            'unfinished': sum(not row['terminated_with_end'] for row in predictions),
            'full_report_matches': sum(row['prediction'] == row['reference'] for row in predictions)}


@torch.no_grad()
def audit_validation(model, split, records, words, config, device, output):
    model.eval()
    reverse = {index: word for word, index in words.items()}
    predictions, endings = [], []
    for i, record in enumerate(records):
        sample = split[i]
        if sample['id'] != record['id']:
            raise ValueError('VAL image/target identity mismatch')
        memory = model.encoder(sample['image'].unsqueeze(0).to(device))
        generation = beam_search(model.decoder, memory, words, config['lambda_parallel'],
                                 config['beam_size'], config['max_decode_steps'])
        predictions.append({'id': record['id'],
                            'reference': ' '.join(reverse[token] for token in record['tokens'][1:-1]), **generation})
        scored = score_batch(model.decoder, [memory], [record], words, config['lambda_parallel'], device)[0]
        endings.append({'id': record['id'], 'reference_risk': record['risk'],
                        'reference_confidence': record['confidence'], 'conditions': {'reference_single': scored}})
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f'VAL generation/endings {i + 1}/{len(records)}', flush=True)
    structure = structural_checks(predictions)
    summary = {'generation': generation_summary(predictions, records), 'structure': structure['summary'],
               'endings': summarize_endings(endings)['conditions']['reference_single']}
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'ending_cases.json', endings)
    write_json(output / 'report_checks.json', structure)
    write_json(output / 'summary.json', summary)
    return summary


def train_arm(model, arm, config, splits, val_records, words, tags, hashes, orders, initial_hash, output, device):
    output.mkdir()
    write_json(output / 'config.json', config)
    write_json(output / 'data_fingerprints.json', hashes)
    write_json(output / 'manifest.json', {'format': FORMAT, 'arm': arm, 'initial_state_sha256': initial_hash,
               'training_order_sha256': fingerprint(output.parent / 'training_order.json'),
               'tag_loss_weight': config['tag_loss_weight'], 'limits': LIMITS})
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    history, best = [], float('inf')
    try:
        for epoch, order in enumerate(orders, 1):
            epoch_seed = (config['seed'] + epoch) % 2 ** 32
            seed_all(epoch_seed)
            loader = OrderedBatches(splits['TRAIN'], order, config['batch_size'], epoch_seed)
            epoch_output = output / f'epoch_{epoch:03d}'
            epoch_output.mkdir()
            write_json(output / 'status.json', {'stage': 'training', 'epoch': epoch})
            print(f'{arm}: epoch {epoch}/{config["epochs"]} training', flush=True)
            train_scores = run_epoch(model, loader, config, device, optimizer, scaler)
            if len(loader.seen) != len(order):
                raise ValueError('Training did not consume the complete shared order')
            write_json(epoch_output / 'train_ids.json', loader.seen)
            val_loader = DataLoader(splits['VAL'], batch_size=config['batch_size'], shuffle=False, num_workers=0,
                                    generator=torch.Generator().manual_seed(epoch_seed))
            write_json(output / 'status.json', {'stage': 'validating', 'epoch': epoch})
            val_scores = run_epoch(model, val_loader, {**config, 'amp': False}, device)
            selection_score = val_scores['primary_ce'] + config['lambda_parallel'] * val_scores['secondary_ce']
            checkpoint = {'format': FORMAT, 'arm': arm, 'model': model.state_dict(), 'config': config,
                          'words': words, 'tags': tags, 'data_fingerprints': hashes, 'epoch': epoch,
                          'initial_state_sha256': initial_hash, 'validation': val_scores,
                          'selection_metric': 'validation_report_ce', 'selection_score': selection_score}
            if selection_score < best:
                best = selection_score
                save_checkpoint(output / 'best.pt', checkpoint)
            save_checkpoint(output / 'last.pt', checkpoint)
            print(f'{arm}: epoch {epoch} VAL generation and ending audit', flush=True)
            validation = audit_validation(model, splits['VAL'], val_records, words, config, device, epoch_output)
            history.append({'epoch': epoch, 'train': train_scores, 'validation_losses': val_scores,
                            'validation_report_ce': selection_score, 'validation': validation})
            write_json(output / 'history.json', history)
            write_json(output / 'status.json', {'stage': 'epoch_complete', 'completed_epochs': epoch})
            print(json.dumps({'arm': arm, 'epoch': epoch, 'validation_report_ce': selection_score,
                              'generation': validation['generation']}), flush=True)
        write_json(output / 'status.json', {'stage': 'complete', 'completed_epochs': len(history)})
        return history
    except BaseException as error:
        write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


def compare_histories(histories):
    first, second = (histories[arm] for arm in ARMS)
    if [row['epoch'] for row in first] != [row['epoch'] for row in second] or not first:
        raise ValueError('Comparison requires both arms at matching epochs')
    comparisons = []
    for on, off in zip(first, second):
        a, b = on['validation']['generation'], off['validation']['generation']
        if a['samples'] != b['samples'] or a['reference_risk_counts'] != b['reference_risk_counts']:
            raise ValueError('Arms must evaluate the same reference population')
        keys = ('risk_matches', 'risk_match_rate', 'healthy_to_high_risk', 'invalid_risk', 'unfinished',
                'unique_reports', 'largest_identical_group', 'full_report_matches')
        comparisons.append({'epoch': on['epoch'], 'lem_on': on['validation'], 'lem_off': off['validation'],
                            'off_minus_on': {key: b[key] - a[key] for key in keys}})
    return {'epochs': comparisons, 'primary_comparison_epoch': first[-1]['epoch'], 'limits': LIMITS}


def run_pair(config, words, tags, splits, hashes, records, orders, output, device):
    seed_all(config['seed'])
    model = build_model({**config, 'pretrained_weights': str(resolve(config['pretrained_weights']))}, words, tags, device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_hash = state_digest(initial)
    save_checkpoint(output / 'initial.pt', {'format': FORMAT, 'model': initial, 'config': config,
                                           'words': words, 'tags': tags, 'data_fingerprints': hashes})
    write_json(output / 'initialization.json', {'state_sha256': initial_hash, 'file_sha256': fingerprint(output / 'initial.pt')})
    histories = {}
    for arm, settings in arm_configs(config).items():
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        if state_digest(model.state_dict()) != initial_hash:
            raise ValueError('Arm does not start from the shared initialization')
        write_json(output / 'status.json', {'stage': 'running', 'arm': arm})
        histories[arm] = train_arm(model, arm, settings, splits, records, words, tags, hashes,
                                   orders, initial_hash, output / arm, device)
    comparison = compare_histories(histories)
    write_json(output / 'comparison.json', comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PACKAGE / 'config.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check configuration/data/cache only; no model or training')
    parser.add_argument('--accept-reconstruction', action='store_true')
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.no_amp:
        config['amp'] = False
    configs = arm_configs(config)
    directory = resolve(config['data_directory'])
    output = resolve(args.output or ROOT / 'artifacts/paper_method' / f'lem_ablation_seed{config["seed"]}')
    if (output.exists() or any(parent == output or parent in output.parents for parent in (directory, PACKAGE)) or
            any((parent / 'status.json').is_file() for parent in output.parents)):
        parser.error('Choose a new output directory outside saved runs, data and code')
    weights = resolve(config['pretrained_weights'])
    if not weights.is_file():
        parser.error('Cached pretrained weights are required; no download is attempted')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    words, tags, splits, hashes = load_dataset(directory)
    try:
        records = reference_records(splits['VAL'], words)
        required_steps = max(splits['TRAIN'].lengths) - 1
        if config['max_decode_steps'] < required_steps:
            parser.error('Decode cap is shorter than a training reference including EOS')
        orders = training_orders(len(splits['TRAIN']), config['epochs'], config['seed'])
        plan = [{'epoch': epoch, 'ids': [splits['TRAIN'].records[i]['id'] for i in order]}
                for epoch, order in enumerate(orders, 1)]
        manifest = {'experiment': FORMAT, 'config_source': str(config_path), 'config_source_sha256': fingerprint(config_path),
                    'output': str(output), 'epochs_per_arm': config['epochs'], 'seed': config['seed'],
                    'arms': {arm: value['tag_loss_weight'] for arm, value in configs.items()},
                    'samples': {name: len(split) for name, split in splits.items()},
                    'device': args.device, 'amp_training': config['amp'], 'amp_validation': False,
                    'cuda_available': torch.cuda.is_available(), 'torch': str(torch.__version__),
                    'test_inference': False, 'paper_comparable': False, 'limits': LIMITS}
        if args.check:
            print(json.dumps(manifest, indent=2), flush=True)
            return
        if not args.accept_reconstruction:
            parser.error('Pass --accept-reconstruction after reading docs/paper_method_contract.md')
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA unavailable. Run on your allocated GPU node')
        if args.device == 'cpu' and config['amp']:
            parser.error('CPU training requires --no-amp')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(output / 'config.json', config)
            write_json(output / 'manifest.json', {**manifest, 'pretrained_sha256': fingerprint(weights)})
            write_json(output / 'data_fingerprints.json', hashes)
            write_json(output / 'training_order.json', plan)
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'initializing'})
            comparison = run_pair(config, words, tags, splits, hashes, records, orders, output, torch.device(args.device))
            write_json(output / 'status.json', {'stage': 'complete', 'epochs_per_arm': config['epochs']})
            print(json.dumps({'output': str(output), 'final_off_minus_on': comparison['epochs'][-1]['off_minus_on']}, indent=2))
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
