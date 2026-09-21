"""Compare unchanged beam-5 reports with greedy decoding on the final lem_off VAL checkpoint."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.ablate_lem import generation_summary, seed_all
from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.decoding import beam_search
from paper_method.diagnose_fields import load_source, replay_prediction, report_records, validate_predictions
from paper_method.evaluation import FIELDS, parse_fields, structural_checks
from paper_method.model import build_model
from paper_method.runner import PACKAGE, archive_sources, resolve

LIMITS = [
    'Same final lem_off checkpoint, VAL images, probability mixture, token budget, eval mode and FP32.',
    'Only search width changes from five to one; width one is greedy decoding with the existing decoder.',
    'Beam-5 predictions are the saved final-epoch outputs, whose log probabilities are replayed before comparison.',
    'Greedy starts from BOS and sees only image features and its own previous tokens, never reference prefixes or candidate values.',
    'No field repair, forced EOS, sampling, reranking, length penalty, loss change, parameter update or TEST inference.',
    'More distinct reports are not automatically better; field/risk agreement, omissions, repetition and termination are also reported.',
    'Local field preferences can differ from complete-sequence rankings without a search implementation bug.',
    'This is a decoding ablation of the reconstruction, not a new paper-standard default or a clinical accuracy claim.',
]


def field_value(parsed, field):
    entries = parsed[field]
    return entries[0]['value'] if len(entries) == 1 and entries[0]['complete'] else None


def mode_summary(predictions, records):
    checks = structural_checks(predictions)
    matches = checks['summary']['reference_field_exact_matches']
    return {'generation': generation_summary(predictions, records), 'structure': checks['summary'],
            'field_matches': sum(matches.values()), 'field_total': len(FIELDS) * len(predictions),
            'field_match_rate': sum(matches.values()) / (len(FIELDS) * len(predictions))}, checks


def compare_predictions(baseline, greedy, records):
    expected_ids = [row['id'] for row in records]
    if ([row['id'] for row in baseline] != expected_ids or [row['id'] for row in greedy] != expected_ids or
            len(set(expected_ids)) != len(expected_ids)):
        raise ValueError('Comparison requires unique, aligned VAL IDs')
    modes, checks = {}, {}
    for name, predictions in (('beam5', baseline), ('greedy', greedy)):
        if [row['reference'] for row in predictions] != [row['reference'] for row in records]:
            raise ValueError('Comparison reference text mismatch')
        modes[name], checks[name] = mode_summary(predictions, records)
    pairs, transitions = [], {field: Counter() for field in FIELDS}
    for before, after in zip(baseline, greedy):
        reference, a, b = (parse_fields(text) for text in (before['reference'], before['prediction'], after['prediction']))
        fields = {}
        for field in FIELDS:
            target, left, right = (field_value(parsed, field) for parsed in (reference, a, b))
            if target is None:
                raise ValueError('Expected complete reference fields')
            transition = ('both_correct' if left == target and right == target else
                          'greedy_fixes' if right == target else 'greedy_regresses' if left == target else 'both_wrong')
            transitions[field][transition] += 1
            fields[field] = {'reference': target, 'beam5': left, 'greedy': right, 'transition': transition}
        pairs.append({'id': before['id'], 'report_changed': before['prediction'] != after['prediction'],
                      'beam5_ended': before['terminated_with_end'], 'greedy_ended': after['terminated_with_end'],
                      'beam5_log_probability': before['log_probability'], 'greedy_log_probability': after['log_probability'],
                      'beam5_token_count': len(before['token_ids']) - 1, 'greedy_token_count': len(after['token_ids']) - 1,
                      'fields': fields})
    keys = ('risk_matches', 'risk_match_rate', 'healthy_to_high_risk', 'invalid_risk', 'unfinished',
            'unique_reports', 'largest_identical_group', 'full_report_matches')
    delta = {key: modes['greedy']['generation'][key] - modes['beam5']['generation'][key] for key in keys}
    delta['field_matches'] = modes['greedy']['field_matches'] - modes['beam5']['field_matches']
    delta['field_match_rate'] = modes['greedy']['field_match_rate'] - modes['beam5']['field_match_rate']
    summary = {'samples': len(records), 'modes': modes, 'greedy_minus_beam5': delta,
               'changed_reports': sum(row['report_changed'] for row in pairs),
               'field_transitions': {field: {key: counts[key] for key in
                   ('both_correct', 'greedy_fixes', 'greedy_regresses', 'both_wrong')} for field, counts in transitions.items()},
               'limits': LIMITS}
    return summary, pairs, checks


@torch.no_grad()
def generate_greedy(model, split, records, baseline, words, config, device):
    model.eval()
    predictions, replays = [], []
    for i, record in enumerate(records):
        sample = split[i]
        if sample['id'] != record['id'] or baseline[i]['id'] != record['id']:
            raise ValueError('VAL image/reference/baseline identity mismatch')
        memory = model.encoder(sample['image'].unsqueeze(0).to(device))
        if not torch.isfinite(memory).all():
            raise ValueError('Non-finite visual memory')
        score = replay_prediction(model.decoder, memory, baseline[i], config['lambda_parallel'])
        generated = beam_search(model.decoder, memory, words, coefficient=config['lambda_parallel'],
                                width=1, max_steps=config['max_decode_steps'])
        predictions.append({'id': record['id'], 'reference': record['reference'], **generated})
        replays.append({'id': record['id'], 'saved_beam5_score': baseline[i]['log_probability'], 'replayed_beam5_score': score})
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f'Greedy VAL comparison {i + 1}/{len(records)}', flush=True)
    validate_predictions(predictions, records, words, config['max_decode_steps'])
    return predictions, replays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/paper_method/lem_ablation_seed123/lem_off'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Validate data/checkpoint/baseline only; no model construction')
    args = parser.parse_args()
    source = resolve(args.run)
    config = read_json(source / 'config.json')
    directory = resolve(config['data_directory'])
    output = resolve(args.output or source.parent.with_name(source.parent.name + '_lem_off_decoding_comparison'))
    if (output.exists() or any(parent == output or parent in output.parents for parent in (source.parent, directory, PACKAGE)) or
            any((parent / 'status.json').is_file() for parent in output.parents)):
        parser.error('Choose a new output directory outside saved runs, data and code')
    words, tags, splits, hashes = load_dataset(directory)
    try:
        config, checkpoint = load_source(source, words, tags, hashes)
        if config['beam_size'] != 5:
            parser.error('This comparison requires the saved baseline beam width to be five')
        records = report_records(splits['VAL'], words)
        prediction_path = source / f'epoch_{checkpoint["epoch"]:03d}' / 'predictions.json'
        baseline = read_json(prediction_path)
        validate_predictions(baseline, records, words, config['max_decode_steps'])
        baseline_summary, _ = mode_summary(baseline, records)
        weights = resolve(config['pretrained_weights'])
        if not weights.is_file():
            parser.error('Cached pretrained weights required; no download is attempted')
        manifest = {'experiment': 'paper_core_greedy_vs_beam5_v1', 'source_run': str(source),
                    'checkpoint_file': 'last.pt', 'checkpoint_epoch': checkpoint['epoch'],
                    'checkpoint_sha256': fingerprint(source / 'last.pt'),
                    'prediction_file': str(prediction_path), 'prediction_sha256': fingerprint(prediction_path),
                    'decoding_widths': {'beam5': 5, 'greedy': 1}, 'max_decode_steps': config['max_decode_steps'],
                    'split': 'VAL', 'samples': len(records), 'output': str(output), 'device': args.device,
                    'cuda_available': torch.cuda.is_available(), 'torch': str(torch.__version__),
                    'training': False, 'test_inference': False, 'inference_amp': False, 'limits': LIMITS}
        if args.check:
            print(json.dumps({'manifest': manifest, 'baseline': baseline_summary}, indent=2), flush=True)
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
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'generating'})
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(weights)}, words, tags, device)
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            greedy, replays = generate_greedy(model, splits['VAL'], records, baseline, words, config, device)
            summary, pairs, checks = compare_predictions(baseline, greedy, records)
            write_json(output / 'beam5_predictions.json', baseline)
            write_json(output / 'greedy_predictions.json', greedy)
            write_json(output / 'prediction_replay.json', replays)
            write_json(output / 'paired_cases.json', pairs)
            write_json(output / 'report_checks.json', checks)
            write_json(output / 'summary.json', summary)
            write_json(output / 'status.json', {'stage': 'complete', 'samples': len(records)})
            print(json.dumps({'output': str(output), 'greedy_minus_beam5': summary['greedy_minus_beam5']}, indent=2))
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
