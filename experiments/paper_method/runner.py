"""Isolated reconstruction runner. No automatic downloads or historical checkpoint reuse."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import tarfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import fingerprint, load_dataset, read_json, write_json
from .evaluation import generate_reports, structural_checks, text_metrics
from .losses import objective
from .model import build_model

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
FORMAT = 'da_spl_paper_image_core_v1'


def resolve(path):
    return (ROOT / path).resolve()


def validate_config(config):
    expected = set(read_json(PACKAGE / 'config.json'))
    if set(config) != expected:
        raise ValueError(f'Config keys differ: {sorted(set(config) ^ expected)}')
    if config['method'] != 'ieee_da_spl_image_core' or config['reproduction_status'] != \
            'paper_guided_reconstruction_with_documented_assumptions':
        raise ValueError('Only the explicitly qualified image-core reconstruction is implemented')
    for key in ('epochs', 'batch_size', 'hidden_dim', 'attention_heads', 'beam_size', 'max_decode_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2 ** 32:
        raise ValueError('Invalid random seed')
    for key in ('learning_rate', 'weight_decay', 'lambda_parallel', 'tag_loss_weight', 'gradient_clip_norm'):
        if type(config[key]) not in (float, int) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Invalid finite nonnegative {key}')
    if (config['learning_rate'] == 0 or not 0 < config['lambda_parallel'] <= 1 or
            config['tag_loss_weight'] == 0 or config['gradient_clip_norm'] == 0):
        raise ValueError('Learning rate, loss weights and clip norm must be positive')
    if config['hidden_dim'] % config['attention_heads']:
        raise ValueError('hidden_dim must be divisible by attention_heads')
    if any(type(config[key]) is not bool for key in ('amp', 'freeze_backbone')):
        raise ValueError('amp/freeze_backbone must be boolean')


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(source, config, words, tags, hashes):
    checkpoint = torch.load(source / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('format') != FORMAT:
        raise ValueError('Historical GitHub/repair checkpoints are not compatible with this reconstruction')
    for key, expected in (('config', config), ('words', words), ('tags', tags), ('data_fingerprints', hashes)):
        if checkpoint.get(key) != expected:
            raise ValueError(f'Checkpoint {key} mismatch')
    with tarfile.open(source / 'source.tar.gz') as archive:
        for name in ('model.py', 'losses.py', 'decoding.py', 'data.py', 'evaluation.py'):
            path = PACKAGE / name
            if archive.extractfile(str(path.relative_to(ROOT))).read() != path.read_bytes():
                raise ValueError(f'Source changed since checkpoint creation: {name}')
    return checkpoint


def run_epoch(model, loader, config, device, optimizer=None, scaler=None):
    model.train(optimizer is not None)
    total_tokens, total_reports, totals = 0, 0, {}
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            images, caps, lengths, tags = (batch[key].to(device) for key in ('image', 'caption', 'length', 'tags'))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config['amp']):
                outputs = model(images, caps, lengths)
                loss, parts, count = objective(outputs, caps, lengths, tags,
                                               config['lambda_parallel'], config['tag_loss_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss; stopping instead of saving an invalid result')
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            for key, value in {'total': loss.detach(), **parts}.items():
                denominator = len(caps) if key.endswith('eos_ce') else count
                totals[key] = totals.get(key, 0.) + value.item() * denominator
            total_tokens += count
            total_reports += len(caps)
    if not total_tokens:
        raise ValueError('Empty epoch')
    return {key: value / (total_reports if key.endswith('eos_ce') else total_tokens) for key, value in totals.items()}


def train(model, splits, config, words, tags, hashes, output, device):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    generator = torch.Generator().manual_seed(config['seed'])
    loaders = {name: DataLoader(splits[name], batch_size=config['batch_size'], shuffle=name == 'TRAIN',
                               num_workers=0, generator=generator if name == 'TRAIN' else None)
               for name in ('TRAIN', 'VAL')}
    best, history = math.inf, []
    for epoch in range(1, config['epochs'] + 1):
        train_scores = run_epoch(model, loaders['TRAIN'], config, device, optimizer, scaler)
        val_scores = run_epoch(model, loaders['VAL'], config, device)
        history.append({'epoch': epoch, 'train': train_scores, 'validation': val_scores})
        print(json.dumps(history[-1]), flush=True)
        checkpoint = {'format': FORMAT, 'model': model.state_dict(), 'config': config,
                      'words': words, 'tags': tags, 'data_fingerprints': hashes, 'epoch': epoch,
                      'validation': val_scores}
        if val_scores['total'] < best:
            best = val_scores['total']
            save_checkpoint(output / 'best.pt', checkpoint)
        save_checkpoint(output / 'last.pt', {**checkpoint, 'optimizer': optimizer.state_dict(),
                                             'scaler': scaler.state_dict()})
        write_json(output / 'history.json', history)
        write_json(output / 'status.json', {'stage': 'training', 'completed_epochs': epoch})
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    return checkpoint['epoch']


def archive_sources(output):
    paths = [path for path in PACKAGE.rglob('*') if path.is_file() and path.suffix in ('.py', '.json', '.sh', '.md')]
    paths += [ROOT / 'docs/paper_method_contract.md', ROOT / 'environment.yml']
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Defaults to paper_method/config.json')
    parser.add_argument('--run-dir', type=Path, help='New directory; never overwrites an existing run')
    parser.add_argument('--evaluate-run', type=Path, help='Evaluate only a checkpoint from this independent model')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--check', action='store_true', help='Data/import/cache checks only; no model construction or inference')
    parser.add_argument('--accept-reconstruction', action='store_true', help='Acknowledge docs/paper_method_contract.md, not exact paper reproduction')
    args = parser.parse_args()
    source = resolve(args.evaluate_run) if args.evaluate_run is not None else None
    if source and (args.train_only or args.config or args.epochs is not None or args.batch_size is not None or args.seed is not None or args.no_amp):
        parser.error('Evaluation reuses saved configuration; do not pass training overrides')
    config = read_json(source / 'config.json' if source else resolve(args.config or PACKAGE / 'config.json'))
    for key in ('epochs', 'batch_size', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.no_amp:
        config['amp'] = False
    validate_config(config)
    default = source.with_name(source.name + '_evaluation') if source else ROOT / 'artifacts/paper_method' / f"image_core_seed{config['seed']}"
    output = resolve(args.run_dir or default)
    if output.exists() or (source and source in output.parents):
        parser.error('Choose a new output directory outside any source run')
    weights = resolve(config['pretrained_weights'])
    if not weights.is_file():
        parser.error(f'Pretrained cache missing; no automatic download: {weights}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    words, tags, splits, hashes = load_dataset(resolve(config['data_directory']))
    try:
        checkpoint = load_checkpoint(source, config, words, tags, hashes) if source else None
        import timm
        from nlgeval import NLGEval  # Import check only; no metric model construction.

        required_steps = max(splits['TRAIN'].lengths) - 1
        if config['max_decode_steps'] < required_steps:
            parser.error('Decode cap is shorter than a training reference including EOS')
        report = {'method': config['method'], 'reproduction_status': config['reproduction_status'],
                  'paper_comparable': False, 'scope': 'image_only_core', 'output': str(output),
                  'samples': {name: len(split) for name, split in splits.items()},
                  'epochs': config['epochs'], 'amp': config['amp'], 'cuda_available': torch.cuda.is_available(),
                  'pretrained_weights_cached': True, 'torch': str(torch.__version__), 'timm': timm.__version__,
                  'longest_training_target_steps': required_steps, 'max_decode_steps': config['max_decode_steps'],
                  'mode': 'evaluation_only' if source else 'train_only' if args.train_only else 'train_and_evaluate',
                  'assumptions': 'docs/paper_method_contract.md'}
        if args.check:
            print(json.dumps(report, indent=2), flush=True)
            return
        if not args.accept_reconstruction:
            parser.error('Read docs/paper_method_contract.md and pass --accept-reconstruction to acknowledge the unresolved details')
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA unavailable. Run on your allocated GPU node')
        if args.device == 'cpu' and config['amp'] and source is None:
            parser.error('CPU training requires --no-amp')
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(output / 'config.json', config)
            write_json(output / 'data_fingerprints.json', hashes)
            write_json(output / 'manifest.json', {**report, 'pretrained_sha256': fingerprint(weights),
                       'source_run': str(source) if source else None,
                       'source_checkpoint_sha256': fingerprint(source / 'best.pt') if source else None,
                       'assumptions_acknowledged': True})
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'initializing'})
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(weights)}, words, tags, device)
            if checkpoint:
                model.load_state_dict(checkpoint['model'], strict=True)
                selected_epoch = checkpoint['epoch']
            else:
                selected_epoch = train(model, splits, config, words, tags, hashes, output, device)
            if args.train_only:
                write_json(output / 'status.json', {'stage': 'training_complete', 'selected_epoch': selected_epoch})
                return
            write_json(output / 'status.json', {'stage': 'evaluating', 'selected_epoch': selected_epoch})
            predictions = generate_reports(model, splits['TEST'], words, config, device)
            write_json(output / 'predictions.json', predictions)
            checks = structural_checks(predictions)
            write_json(output / 'report_checks.json', checks)
            scores = text_metrics(predictions)
            write_json(output / 'metrics.json', {'paper_comparable': False, 'selected_epoch': selected_epoch,
                       'reproduction_status': config['reproduction_status'], **checks['summary'], 'metrics_raw': scores,
                       'metrics_x100': {key: value * 100 for key, value in scores.items()}})
            write_json(output / 'status.json', {'stage': 'complete', 'selected_epoch': selected_epoch})
            print(json.dumps(checks['summary'], indent=2), flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()
