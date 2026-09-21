"""Trace the selected VAL risk reversals without changing the production search."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.data import fingerprint, load_dataset, read_json, write_json
from paper_method.decoding import beam_search
from paper_method.diagnose_risk import candidate_tokens, extract_prefix, head_distributions, strict_generated_risk
from paper_method.model import build_model
from paper_method.runner import archive_sources, load_checkpoint, resolve, validate_config


def same_score(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-4):
        raise ValueError(f'Score replay mismatch: {actual} versus {expected}')


def starts_with(tokens, prefix):
    return tokens[:len(prefix)] == prefix


def select_cases(rows, words):
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate audit sample IDs')
    selected = []
    for row in rows:
        prefix = row['generated_prefix_ids']
        if prefix is None:
            continue
        if extract_prefix(row['generation']['token_ids'], words) != prefix:
            raise ValueError('Saved generated prefix does not match generated tokens')
        profile = row['conditions']['own_image_generated_prefix']['mixture']
        if (profile['restricted_first_token_choice'] == 'very healthy' and
                profile['restricted_sequence_choice'] == 'very healthy' and
                strict_generated_risk(row['generation']['prediction'])['label'] == 'high risk'):
            selected.append(row)
    if not selected:
        raise ValueError('No local-healthy/final-high-risk cases in this audit')
    return selected


@dataclass
class ObservedState:
    core: object
    tokens: list
    score: float
    log_probs: object = None


class ObservedDecoder:
    """Opaque-state adapter: returns the original logits and never masks them."""

    def __init__(self, decoder, words, coefficient, width, watched_path):
        self.decoder, self.words, self.coefficient = decoder, words, coefficient
        self.width, self.watched_path, self.events = width, watched_path, []

    @property
    def training(self):
        return self.decoder.training

    def initial_state(self, memory):
        self.events = []
        return ObservedState(self.decoder.initial_state(memory), [], 0.)

    def step(self, memory, previous_word, state):
        token = previous_word.item()
        prefix = state.tokens + [token]
        score = state.score + (state.log_probs[token].item() if state.tokens else 0.)
        first, second, core = self.decoder.step(memory, previous_word, state.core)
        distributions = head_distributions(first, second, self.coefficient)
        scores = distributions['mixture'].clone()
        scores[[self.words['<start>'], self.words['<pad>']]] = -torch.inf
        values, indices = scores.topk(min(self.width, scores.numel() - 2))
        event = {'parent_token_ids': prefix, 'prefix_log_probability': score,
                 'extensions': [{'token_id': index, 'log_probability': value,
                                 'head_log_probabilities': {head: probs[index].item() for head, probs in distributions.items()}}
                                for value, index in zip(values.tolist(), indices.tolist())],
                 'eos_log_probabilities': {head: probs[self.words['<end>']].item() for head, probs in distributions.items()}}
        if len(prefix) < len(self.watched_path) and self.watched_path[:len(prefix)] == prefix:
            desired = self.watched_path[len(prefix)]
            event['watched_next'] = {'token_id': desired, 'log_probability': scores[desired].item(),
                                     'local_rank_lower_bound': 1 + (scores > scores[desired]).sum().item(),
                                     'in_parent_topk': desired in indices.tolist()}
        self.events.append(event)
        return first, second, ObservedState(core, prefix, score, distributions['mixture'])


def reconstruct_search(events, result, words, width, max_steps, prefix, healthy_path):
    """Reconstruct rankings from observed expansions; no second model/search run."""
    bos, eos = words['<start>'], words['<end>']
    reverse = {index: word for word, index in words.items()}
    beams = [{'token_ids': [bos], 'score': 0., 'ended': False}]
    cursor, frames, first_elimination = 0, [], None

    def healthy(candidate):
        tokens = candidate['token_ids']
        end = min(len(tokens), len(healthy_path))
        return len(tokens) > len(prefix) and tokens[:end] == healthy_path[:end]

    def snapshot(candidate, rank):
        return {**candidate, 'rank': rank,
                'tail': ' '.join(reverse[token] for token in candidate['token_ids'][len(prefix):])}

    for step in range(1, max_steps + 1):
        proposals, expanded = [], []
        had_healthy = any(healthy(beam) for beam in beams)
        for beam in beams:
            if beam['ended']:
                proposals.append(beam)
                continue
            if cursor >= len(events):
                raise ValueError('Missing decoder event')
            event = events[cursor]
            cursor += 1
            if event['parent_token_ids'] != beam['token_ids']:
                raise ValueError('Observed beam order differs from reconstructed order')
            same_score(event['prefix_log_probability'], beam['score'])
            expanded.append(event)
            for extension in event['extensions']:
                token = extension['token_id']
                proposals.append({'token_ids': beam['token_ids'] + [token],
                                  'score': beam['score'] + extension['log_probability'], 'ended': token == eos})
        ranked = sorted(proposals, key=lambda item: item['score'], reverse=True)
        beams = ranked[:width]
        if step >= len(prefix):
            proposed_healthy = [(rank, beam) for rank, beam in enumerate(ranked, 1) if healthy(beam)]
            retained_healthy = [(rank, beam) for rank, beam in enumerate(beams, 1) if healthy(beam)]
            frame = {'step': step, 'retained': [snapshot(beam, rank) for rank, beam in enumerate(beams, 1)],
                     'best_healthy_proposal': snapshot(proposed_healthy[0][1], proposed_healthy[0][0]) if proposed_healthy else None,
                     'best_healthy_retained': snapshot(retained_healthy[0][1], retained_healthy[0][0]) if retained_healthy else None,
                     'expansions': expanded}
            if first_elimination is None and step < len(healthy_path):
                parent = healthy_path[:step]
                event = next((item for item in expanded if item['parent_token_ids'] == parent), None)
                desired_path = healthy_path[:step + 1]
                survived = any(beam['token_ids'] == desired_path for beam in beams)
                if not survived:
                    first_elimination = {
                        'step': step, 'stage': 'risk_phrase', 'required_token': reverse[healthy_path[step]],
                        'reason': 'ancestor_absent' if event is None else
                                  'outside_parent_topk' if not event['watched_next']['in_parent_topk'] else 'global_beam_pruning',
                        'watched_next': event.get('watched_next') if event else None,
                        'best_healthy_proposal': frame['best_healthy_proposal'],
                        'retained_cutoff_score': beams[-1]['score'],
                    }
            elif first_elimination is None and had_healthy and not retained_healthy:
                first_elimination = {'step': step, 'stage': 'after_risk_phrase',
                                     'reason': 'all_healthy_phrase_continuations_pruned',
                                     'best_healthy_proposal': frame['best_healthy_proposal'],
                                     'retained_cutoff_score': beams[-1]['score']}
            frames.append(frame)
        if all(beam['ended'] for beam in beams):
            break
    if cursor != len(events) or beams[0]['token_ids'] != result['token_ids']:
        raise ValueError('Reconstructed final beam differs from production search')
    same_score(beams[0]['score'], result['log_probability'])
    return {'first_healthy_path_elimination': first_elimination, 'frames': frames,
            'healthy_path_survived_to_final_beam': any(healthy(beam) for beam in beams)}


class PrefixContinuation:
    """Virtual BOS adapter for a separately labelled fixed-prefix counterfactual."""

    def __init__(self, decoder, fixed_prefix, bos):
        self.decoder, self.fixed_prefix, self.bos = decoder, fixed_prefix, bos

    @property
    def training(self):
        return self.decoder.training

    def initial_state(self, memory):
        core = self.decoder.initial_state(memory)
        for token in self.fixed_prefix[:-1]:
            _, _, core = self.decoder.step(memory, torch.tensor([token], device=memory.device), core)
        return core, True

    def step(self, memory, previous_word, state):
        core, first_call = state
        if first_call:
            if previous_word.item() != self.bos:
                raise ValueError('Continuation must start with virtual BOS')
            previous_word = torch.tensor([self.fixed_prefix[-1]], device=memory.device)
        first, second, core = self.decoder.step(memory, previous_word, core)
        return first, second, (core, False)


@torch.no_grad()
def score_path(decoder, memory, tokens, words, coefficient, prefix_length, risk_length):
    state, total, records = decoder.initial_state(memory), 0., []
    reverse = {index: word for word, index in words.items()}
    marker = [words[token] for token in 'confidence level :'.split()]
    confidence_start = next((i for i in range(prefix_length + risk_length, len(tokens) - len(marker) + 1)
                             if tokens[i:i + len(marker)] == marker), None)
    for position in range(1, len(tokens)):
        first, second, state = decoder.step(memory, torch.tensor([tokens[position - 1]], device=memory.device), state)
        distributions = head_distributions(first, second, coefficient)
        token = tokens[position]
        scores = {head: probs[token].item() for head, probs in distributions.items()}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError('Non-finite path score')
        total += scores['mixture']
        stage = ('report_prefix' if position < prefix_length else
                 'risk_phrase' if position < prefix_length + risk_length else
                 'eos' if token == words['<end>'] else
                 'post_risk_before_confidence' if confidence_start is None or position < confidence_start else
                 'confidence_marker' if position < confidence_start + len(marker) else 'after_confidence_marker')
        records.append({'position': position, 'token_id': token, 'token': reverse[token], 'stage': stage,
                        'head_log_probabilities': scores, 'cumulative_log_probability': total,
                        'eos_probability': distributions['mixture'][words['<end>']].exp().item()})
    return {'token_scores': records, 'log_probability': total,
            'stage_log_probabilities': {stage: sum(item['head_log_probabilities']['mixture'] for item in records if item['stage'] == stage)
                                        for stage in sorted({item['stage'] for item in records})}}


def compare_paths(healthy, original, prefix_length, risk_length):
    a, b = healthy['token_scores'], original['token_scores']
    timeline, first_reversal = [], None
    risk_end = prefix_length + risk_length - 1
    for position in range(risk_end, max(len(a), len(b)) + 1):
        left, right = a[min(position, len(a)) - 1], b[min(position, len(b)) - 1]
        margin = left['cumulative_log_probability'] - right['cumulative_log_probability']
        item = {'position': position, 'healthy_minus_original_log_probability': margin,
                'healthy_token': left['token'], 'original_token': right['token'],
                'healthy_stage': left['stage'], 'original_stage': right['stage'],
                'healthy_score_carried': position > len(a), 'original_score_carried': position > len(b)}
        if timeline and first_reversal is None and timeline[-1]['healthy_minus_original_log_probability'] >= 0 and margin < 0:
            first_reversal = item
        timeline.append(item)
    return {'first_score_reversal_on_explored_continuation': first_reversal,
            'timeline': timeline, 'final_healthy_minus_original_log_probability': healthy['log_probability'] - original['log_probability']}


@torch.no_grad()
def trace_case(model, memory, case, words, config):
    if model.training or memory.size(0) != 1:
        raise ValueError('Trace requires eval mode and a single image')
    prefix = case['generated_prefix_ids']
    healthy_tokens = candidate_tokens(words)['very healthy']
    healthy_path = prefix + healthy_tokens
    coefficient, width, cap = config['lambda_parallel'], config['beam_size'], config['max_decode_steps']
    observed = ObservedDecoder(model.decoder, words, coefficient, width, healthy_path)
    result = beam_search(observed, memory, words, coefficient, width, cap)
    saved = case['generation']
    if any(result[key] != saved[key] for key in ('token_ids', 'prediction', 'terminated_with_end')):
        raise ValueError('Production search no longer reproduces the saved diagnostic report')
    same_score(result['log_probability'], saved['log_probability'])
    search = reconstruct_search(observed.events, result, words, width, cap, prefix, healthy_path)
    original_scores = score_path(model.decoder, memory, result['token_ids'], words, coefficient, len(prefix), len(healthy_tokens))
    same_score(original_scores['log_probability'], result['log_probability'])
    remaining = cap - (len(healthy_path) - 1)
    if remaining < 1:
        raise ValueError('No decode budget left after the fixed healthy phrase')
    adapter = PrefixContinuation(model.decoder, healthy_path, words['<start>'])
    suffix = beam_search(adapter, memory, words, coefficient, width, remaining)
    alternative_tokens = healthy_path + suffix['token_ids'][1:]
    reverse = {index: word for word, index in words.items()}
    alternative_text = ' '.join(reverse[token] for token in alternative_tokens
                                if token not in (words['<start>'], words['<end>'], words['<pad>']))
    healthy_scores = score_path(model.decoder, memory, alternative_tokens, words, coefficient, len(prefix), len(healthy_tokens))
    forced_score = healthy_scores['token_scores'][len(healthy_path) - 2]['cumulative_log_probability']
    same_score(healthy_scores['log_probability'], forced_score + suffix['log_probability'])
    profile = case['conditions']['own_image_generated_prefix']['mixture']
    same_score(original_scores['stage_log_probabilities']['risk_phrase'], profile['sequence_log_probabilities']['high risk'])
    same_score(healthy_scores['stage_log_probabilities']['risk_phrase'], profile['sequence_log_probabilities']['very healthy'])
    comparison = compare_paths(healthy_scores, original_scores, len(prefix), len(healthy_tokens))
    parsed = strict_generated_risk(alternative_text)
    verdict = ('healthy_continuation_unfinished' if not suffix['terminated_with_end'] else
               'healthy_continuation_has_invalid_risk' if parsed['label'] != 'very healthy' else
               'completed_valid_healthy_path_scores_higher' if comparison['final_healthy_minus_original_log_probability'] > 1e-4 else
               'explored_completed_healthy_path_scores_lower_or_equal')
    return {'id': case['id'], 'reference_risk': case['reference_risk'], 'generated_prefix_ids': prefix,
            'original_generation': result, 'original_search': search, 'original_path_scores': original_scores,
            'constrained_healthy_continuation': {'prediction': alternative_text, 'token_ids': alternative_tokens,
                'terminated_with_end': suffix['terminated_with_end'], 'risk': parsed, 'scores': healthy_scores,
                'remaining_decode_steps': remaining},
            'comparison': comparison, 'diagnostic_outcome': verdict}


def summarize(rows):
    return {'samples': len(rows), 'split': 'VAL',
            'outcomes': dict(Counter(row['diagnostic_outcome'] for row in rows)),
            'original_search_elimination_stages': dict(Counter(
                (row['original_search']['first_healthy_path_elimination'] or {}).get('stage', 'survived_to_final_beam')
                for row in rows)),
            'cases': [{'id': row['id'], 'diagnostic_outcome': row['diagnostic_outcome'],
                       'original_search_first_elimination': row['original_search']['first_healthy_path_elimination'],
                       'explored_continuation_first_score_reversal': row['comparison']['first_score_reversal_on_explored_continuation'],
                       'final_healthy_minus_original_log_probability': row['comparison']['final_healthy_minus_original_log_probability']}
                      for row in rows],
            'limits': [
                'Production beam search is unmodified; original-search observations and constrained continuations are separate.',
                'A completed valid higher-scoring healthy continuation witnesses a missed candidate, not clinical correctness.',
                'A lower-scoring explored continuation does not rule out other better healthy paths; beam width is finite.',
                'Scores include real token probabilities, periods and EOS, with the original total decode budget.',
                'Counterfactual continuation fixes the generated prefix and healthy phrase, not confidence or EOS.',
                'After-confidence-marker stage may include repetition or later fields, not just a confidence number.',
                'Only preselected VAL failures are traced. No overall accuracy or independent validation claim.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path('artifacts/paper_method/image_core_seed123_risk_diagnostic'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check', action='store_true', help='Metadata/checkpoint preflight only; no model construction')
    args = parser.parse_args()
    audit = resolve(args.audit)
    manifest = read_json(audit / 'manifest.json')
    if (read_json(audit / 'status.json')['stage'] != 'complete' or manifest['split'] != 'VAL' or
            manifest['diagnostic'] != 'paper_core_risk_conditioning_v1'):
        parser.error('Requires a completed VAL risk-conditioning audit')
    source = resolve(manifest['source_run'])
    output = resolve(args.output or source.with_name(source.name + '_risk_beam_trace'))
    config = read_json(source / 'config.json')
    validate_config(config)
    directory = resolve(config['data_directory'])
    if output.exists() or any(parent == output or parent in output.parents for parent in
                              (source, audit, directory, Path(__file__).resolve().parent)):
        parser.error('Choose a new output directory outside source runs, data and code')
    if read_json(audit / 'config.json') != config or fingerprint(source / 'best.pt') != manifest['checkpoint_sha256']:
        parser.error('Audit configuration/checkpoint mismatch')
    words, tags, splits, hashes = load_dataset(directory)
    try:
        checkpoint = load_checkpoint(source, config, words, tags, hashes)
        if read_json(audit / 'data_fingerprints.json') != hashes:
            parser.error('Audit data fingerprints do not match the checkpoint dataset')
        selected = select_cases(read_json(audit / 'cases.json'), words)
        indices = {row['id']: i for i, row in enumerate(splits['VAL'].records)}
        if any(row['id'] not in indices for row in selected):
            parser.error('Selected case does not belong to VAL')
        report = {'diagnostic': 'paper_core_risk_beam_trace_v1', 'source_run': str(source), 'source_audit': str(audit),
                  'checkpoint_sha256': manifest['checkpoint_sha256'], 'audit_cases_sha256': fingerprint(audit / 'cases.json'),
                  'split': 'VAL', 'selected_ids': [row['id'] for row in selected], 'samples': len(selected),
                  'device': args.device, 'cuda_available': torch.cuda.is_available(), 'inference_amp': False,
                  'training': False, 'test_inference': False, 'output': str(output), 'torch': str(torch.__version__)}
        if args.check:
            print(json.dumps(report, indent=2), flush=True)
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
            write_json(output / 'manifest.json', report)
            archive_sources(output)
            write_json(output / 'status.json', {'stage': 'tracing'})
            device = torch.device(args.device)
            model = build_model({**config, 'pretrained_weights': str(resolve(config['pretrained_weights']))}, words, tags, device)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.eval()
            del checkpoint
            rows = []
            with torch.no_grad():
                for case in selected:
                    sample = splits['VAL'][indices[case['id']]]
                    if sample['id'] != case['id']:
                        raise ValueError('Image/metadata identity mismatch')
                    memory = model.encoder(sample['image'].unsqueeze(0).to(device))
                    if not torch.isfinite(memory).all():
                        raise ValueError('Non-finite image encoding')
                    rows.append(trace_case(model, memory, case, words, config))
                    write_json(output / 'cases.json', rows)
                    print(f"Traced {len(rows)}/{len(selected)}: {case['id']}", flush=True)
            summary = summarize(rows)
            write_json(output / 'summary.json', summary)
            write_json(output / 'status.json', {'stage': 'complete', 'samples': len(rows)})
            print(json.dumps(summary, indent=2), flush=True)
        except BaseException as error:
            write_json(output / 'status.json', {'stage': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
    finally:
        for split in splits.values():
            split.close()


if __name__ == '__main__':
    main()
