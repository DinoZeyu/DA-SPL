"""Longer LEM-off training from the archived initialization, with TRAIN/VAL monitoring."""

import argparse
import json
import os
from pathlib import Path
import sys
import tarfile

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.ablate_lem import FORMAT as SOURCE_FORMAT, OrderedBatches, seed_all, state_digest, training_orders
from paper_method.audit_pipeline import (describe_scores, distributions, generation_metrics,
                                         majority_baselines, records_for, summarize_scores)
from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.decoding import beam_search
from paper_method.diagnose_fields import load_source
from paper_method.evaluation import FIELDS
from paper_method.model import build_model
from paper_method.runner import PACKAGE, ROOT, archive_sources, resolve, run_epoch, save_checkpoint

FORMAT = 'da_spl_paper_lem_off_long_v1'
CONTENT_FIELDS = tuple(field for field in FIELDS if field not in ('confidence level', 'additional observations'))
LIMITS = [
    'Only epoch count changes relative to the source lem_off configuration; no LEM-on arm is trained.',
    'Fresh Adam/scaler from the archived shared initialization, not a 25-epoch continuation of last.pt.',
    'Initial tensors, data, vocabulary and the original training-order prefix are verified before training.',
    'Per-epoch RNG resets isolate training from monitoring; identical inputs do not guarantee cross-device bitwise replay.',
    'Frozen backbone, trainable projection, teacher forcing, loss weights and original decoder remain unchanged.',
    'VAL beam-5 and greedy every epoch; full TRAIN greedy at multiples of five, the reference epoch, and the final epoch.',
    'Monitoring is eval FP32; teacher-forced token scores are not free-generation accuracy.',
    'best.pt retains lowest VAL report CE; best_fields.pt is a separate exploratory VAL-greedy content-field selection.',
    'Content-field selection excludes confidence and additional observations; all 14 fields are still reported.',
    'No TEST inference, early stopping, scheduler, reweighting, sampling, forced EOS or field repair.',
    'Repeatedly inspected VAL agreement is not independent clinical accuracy or exact paper reproduction.',
]


def load_training_setup(source, epochs, words, tags, splits, hashes):
    baseline, checkpoint = load_source(source, words, tags, hashes)
    if type(epochs) is not int or epochs <= baseline['epochs']:
        raise ValueError('Total epochs must exceed the completed source run')
    if baseline['beam_size'] != 5:
        raise ValueError('Expected the original beam-5 configuration')
    with tarfile.open(source.parent / 'source.tar.gz') as archive:
        path = PACKAGE / 'runner.py'
        if archive.extractfile(str(path.relative_to(ROOT))).read() != path.read_bytes():
            raise ValueError('Training loop changed since source run: runner.py')
    initial_path = source.parent / 'initial.pt'
    initialization = read_json(source.parent / 'initialization.json')
    if fingerprint(initial_path) != initialization['file_sha256']:
        raise ValueError('Initial checkpoint fingerprint mismatch')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    expected = {'format': SOURCE_FORMAT, 'config': read_json(source.parent / 'config.json'),
                'words': words, 'tags': tags, 'data_fingerprints': hashes}
    for key, value in expected.items():
        if initial.get(key) != value:
            raise ValueError(f'Initial checkpoint {key} mismatch')
    initial_hash = state_digest(initial['model'])
    if initial_hash != checkpoint['initial_state_sha256'] or initial_hash != initialization['state_sha256']:
        raise ValueError('Initial model tensors differ from the source ablation')
    orders = training_orders(len(splits['TRAIN']), epochs, baseline['seed'])
    plan = [{'epoch': epoch, 'ids': [splits['TRAIN'].records[i]['id'] for i in order]}
            for epoch, order in enumerate(orders, 1)]
    source_plan = read_json(source.parent / 'training_order.json')
    if plan[:baseline['epochs']] != source_plan:
        raise ValueError('Training-order prefix differs from the source ablation')
    for row in source_plan:
        if read_json(source / f'epoch_{row["epoch"]:03d}' / 'train_ids.json') != row['ids']:
            raise ValueError('Source did not consume its recorded training order')
    reference = {'source_run': str(source), 'epoch': checkpoint['epoch'], 'config': baseline,
                 'checkpoint_sha256': fingerprint(source / 'last.pt'),
                 'model_state_sha256': state_digest(checkpoint['model']),
                 'initial_file_sha256': initialization['file_sha256'], 'initial_state_sha256': initial_hash,
                 'training_order_sha256': fingerprint(source.parent / 'training_order.json'),
                 'history': read_json(source / 'history.json')}
    return {**baseline, 'epochs': epochs}, initial['model'], orders, plan, reference


def content_match_rate(summary):
    matches = summary['structure']['reference_field_exact_matches']
    return sum(matches[field] for field in CONTENT_FIELDS) / (len(CONTENT_FIELDS) * summary['generation']['samples'])


def train_monitor_epochs(total, reference_epoch):
    return sorted(set(range(5, total + 1, 5)) | {reference_epoch, total})


@torch.no_grad()
def monitor_split(model, split, records, words, config, device, output, include_beam):
    model.eval()
    widths = {'beam5': config['beam_size'], 'greedy': 1} if include_beam else {'greedy': 1}
    predictions = {mode: [] for mode in widths}
    cases = []
    for i, record in enumerate(records):
        sample = split[i]
        if sample['id'] != record['id']:
            raise ValueError('Monitoring image/reference identity mismatch')
        memory = model.encoder(sample['image'].unsqueeze(0).to(device))
        if not torch.isfinite(memory).all():
            raise ValueError('Non-finite monitoring image features')
        for mode, width in widths.items():
            generated = beam_search(model.decoder, memory, words, coefficient=config['lambda_parallel'],
                                    width=width, max_steps=config['max_decode_steps'])
            predictions[mode].append({'id': record['id'], 'reference': record['reference'], **generated})
        tokens = torch.tensor([record['tokens']], device=device)
        lengths = torch.tensor([tokens.size(1)], device=device)
        a, b = model.decoder(memory, tokens, lengths)
        scores = distributions(a, b, config['lambda_parallel'])
        cases.append({'id': record['id'], 'conditions': {'own_single': describe_scores(
            {head: value[0] for head, value in scores.items()}, record, words)}})
        if (i + 1) % 25 == 0 or i + 1 == len(records):
            print(f'{output.name}: monitored {i + 1}/{len(records)}', flush=True)
    summaries = {}
    for mode, rows in predictions.items():
        summaries[mode], checks = generation_metrics(rows, records)
        summaries[mode]['content_field_match_rate'] = content_match_rate(summaries[mode])
        write_json(output / f'{mode}_predictions.json', rows)
        write_json(output / f'{mode}_report_checks.json', checks)
    result = {'generation': summaries, 'teacher_forcing': summarize_scores(cases, records, words)}
    write_json(output / 'teacher_cases.json', cases)
    write_json(output / 'summary.json', result)
    return result


def run_training(model, config, splits, records, words, tags, hashes, orders, reference, output, device):
    model.zero_grad(set_to_none=True)
    if state_digest(model.state_dict()) != reference['initial_state_sha256']:
        raise ValueError('Training must start from the verified initial state')
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    history, best_ce, best_fields = [], None, None
    train_epochs = train_monitor_epochs(config['epochs'], reference['epoch'])
    for epoch, order in enumerate(orders, 1):
        seed = (config['seed'] + epoch) % 2 ** 32
        seed_all(seed)
        loader = OrderedBatches(splits['TRAIN'], order, config['batch_size'], seed)
        epoch_output = output / f'epoch_{epoch:03d}'
        epoch_output.mkdir()
        write_json(output / 'status.json', {'stage': 'training', 'epoch': epoch})
        print(f'LEM off: epoch {epoch}/{config["epochs"]} training', flush=True)
        train_losses = run_epoch(model, loader, config, device, optimizer, scaler)
        if len(loader.seen) != len(order):
            raise ValueError('Training did not consume the complete epoch')
        write_json(epoch_output / 'train_ids.json', loader.seen)
        val_loader = DataLoader(splits['VAL'], batch_size=config['batch_size'], shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
        write_json(output / 'status.json', {'stage': 'validating', 'epoch': epoch})
        val_losses = run_epoch(model, val_loader, {**config, 'amp': False}, device)
        ce = val_losses['primary_ce'] + config['lambda_parallel'] * val_losses['secondary_ce']
        checkpoint = {'format': FORMAT, 'arm': 'lem_off', 'model': model.state_dict(), 'config': config,
                      'words': words, 'tags': tags, 'data_fingerprints': hashes, 'epoch': epoch,
                      'initial_state_sha256': reference['initial_state_sha256'],
                      'source_checkpoint_sha256': reference['checkpoint_sha256'],
                      'validation': val_losses, 'selection_metric': 'validation_report_ce', 'selection_score': ce}
        # Preserve optimizer/scaler before the longer monitoring stage, even if it fails.
        save_checkpoint(output / 'last.pt', {**checkpoint, 'optimizer': optimizer.state_dict(),
                                             'scaler': scaler.state_dict()})
        if best_ce is None or ce < best_ce['score']:
            best_ce = {'epoch': epoch, 'score': ce}
            save_checkpoint(output / 'best.pt', checkpoint)
        reference_match = None
        if epoch == reference['epoch']:
            digest = state_digest(model.state_dict())
            reference_match = {'model_state_sha256': digest,
                               'matches_source_final_state': digest == reference['model_state_sha256']}
            save_checkpoint(output / 'reference_epoch.pt', checkpoint)
            write_json(output / 'reference_epoch_comparison.json', reference_match)
            print(f'Reference epoch {epoch} exact tensor replay: {reference_match["matches_source_final_state"]}', flush=True)
        monitored = {}
        names = ('VAL', 'TRAIN') if epoch in train_epochs else ('VAL',)
        for name in names:
            destination = epoch_output / name.lower()
            destination.mkdir()
            write_json(output / 'status.json', {'stage': 'monitoring', 'epoch': epoch, 'split': name})
            monitored[name] = monitor_split(model, splits[name], records[name], words, config, device,
                                            destination, include_beam=name == 'VAL')
        greedy = monitored['VAL']['generation']['greedy']
        score = greedy['content_field_match_rate']
        if best_fields is None or score > best_fields['score']:
            best_fields = {'epoch': epoch, 'score': score}
            save_checkpoint(output / 'best_fields.pt', {**checkpoint,
                'selection_metric': 'validation_greedy_content_field_match_rate', 'selection_score': score,
                'selection_fields': list(CONTENT_FIELDS)})
        row = {'epoch': epoch, 'train_losses': train_losses, 'validation_losses': val_losses,
               'validation_report_ce': ce, 'validation': monitored['VAL'], 'train_monitor': monitored.get('TRAIN'),
               'reference_epoch_comparison': reference_match}
        history.append(row)
        write_json(output / 'history.json', history)
        write_json(output / 'selection.json', {'best_report_ce': best_ce, 'best_content_fields': best_fields,
                                              'content_fields': list(CONTENT_FIELDS), 'tie_policy': 'earliest_epoch'})
        write_json(output / 'status.json', {'stage': 'epoch_complete', 'completed_epochs': epoch})
        risk = greedy['field_agreement']['glaucoma risk assessment']
        print(json.dumps({'epoch': epoch, 'validation_report_ce': ce,
                          'greedy_field_match_rate': greedy['field_match_rate'], 'content_field_match_rate': score,
                          'risk_match_rate': risk['rate'], 'risk_macro_recall': risk['macro_recall'],
                          'risk_class_recall': risk['class_recall'],
                          'unique_reports': greedy['generation']['unique_reports'],
                          'largest_identical_group': greedy['generation']['largest_identical_group']}), flush=True)
    result = {'epochs': config['epochs'], 'best_report_ce': best_ce, 'best_content_fields': best_fields,
              'final_validation': history[-1]['validation'], 'final_train': history[-1]['train_monitor'],
              'train_monitor_epochs': train_epochs, 'limits': LIMITS}
    write_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT / 'artifacts/paper_method/lem_ablation_seed123/lem_off')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=30, help='Total fresh-start epochs, not extra epochs')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Verify inputs only; no model construction, training or output writes')
    parser.add_argument('--accept-reconstruction', action='store_true')
    args = parser.parse_args()
    source = resolve(args.source_run)
    baseline = read_json(source / 'config.json')
    directory = resolve(baseline['data_directory'])
    output = resolve(args.output or ROOT / 'artifacts/paper_method' / f'lem_off_{args.epochs}epochs_seed{baseline["seed"]}')
    if (output.exists() or any(p == output or p in output.parents for p in (ROOT / 'data', directory, PACKAGE)) or
            any((p / 'status.json').is_file() for p in output.parents)):
        parser.error('Choose a new output directory outside saved runs, data and code')
    if args.epochs <= baseline['epochs']:
        parser.error('Total epochs must exceed the source run; default is 30')
    torch.set_num_threads(4)
    words, tags, splits, hashes = load_dataset(directory)
    try:
        config, initial, orders, plan, reference = load_training_setup(source, args.epochs, words, tags, splits, hashes)
        weights = resolve(config['pretrained_weights'])
        if not weights.is_file():
            parser.error('Cached pretrained weights required; no download is attempted')
        if config['max_decode_steps'] < max(splits['TRAIN'].lengths) - 1:
            parser.error('Decode cap is shorter than a TRAIN target')
        records = {name: records_for(splits[name], words) for name in ('TRAIN', 'VAL')}
        manifest = {'experiment': FORMAT, 'source_run': str(source), 'source_epoch': reference['epoch'],
                    'source_checkpoint_sha256': reference['checkpoint_sha256'],
                    'initial_state_sha256': reference['initial_state_sha256'],
                    'initial_file_sha256': reference['initial_file_sha256'], 'output': str(output),
                    'config_changes': {key: {'before': baseline[key], 'after': config[key]}
                                       for key in baseline if baseline[key] != config[key]},
                    'epochs': config['epochs'], 'training_start': 'archived_initial_state_fresh_optimizer',
                    'samples': {name: len(rows) for name, rows in records.items()},
                    'train_monitor_epochs': train_monitor_epochs(config['epochs'], reference['epoch']),
                    'val_monitor_every_epoch': True, 'device': args.device, 'amp_training': config['amp'],
                    'amp_monitoring': False, 'test_inference': False, 'torch': str(torch.__version__),
                    'cuda_available': torch.cuda.is_available(), 'limits': LIMITS}
        if args.check:
            print(json.dumps(manifest, indent=2), flush=True)
            return
        if not args.accept_reconstruction:
            parser.error('Read docs/paper_method_contract.md and pass --accept-reconstruction')
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA unavailable. Run on your allocated GPU node')
        if args.device == 'cpu' and config['amp']:
            parser.error('The source uses AMP; run on CUDA to preserve its settings')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        seed_all(config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(output / 'config.json', config)
            write_json(output / 'manifest.json', {**manifest, 'pretrained_sha256': fingerprint(weights)})
            write_json(output / 'data_fingerprints.json', hashes)
            write_json(output / 'training_order.json', plan)
            write_json(output / 'reference_run.json', reference)
            write_json(output / 'majority_baselines.json', majority_baselines(records))
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'initializing'})
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(weights)}, words, tags, device)
            model.load_state_dict(initial, strict=True)
            del initial
            result = run_training(model, config, splits, records, words, tags, hashes, orders, reference, output, device)
            write_json(output / 'status.json', {'stage': 'complete', 'completed_epochs': config['epochs']})
            print(json.dumps({'output': str(output), 'best_report_ce': result['best_report_ce'],
                              'best_content_fields': result['best_content_fields']}, indent=2), flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
