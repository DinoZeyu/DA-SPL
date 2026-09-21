"""Text metrics and transparent structural checks, never postprocess predictions."""

from collections import Counter

from .decoding import beam_search

FIELDS = ('optic disc size', 'cup to disc ratio', 'neuroretinal rim', 'isnt rule followed',
          'rim pallor', 'rim color', 'bayoneting', 'sharp edge', 'laminar dot sign', 'notching',
          'rim thinning', 'additional observations', 'glaucoma risk assessment', 'confidence level')


def parse_fields(text):
    tokens = text.split()
    boundaries = []
    for name in FIELDS:
        marker = name.split() + [':']
        boundaries.extend((i, name, len(marker)) for i in range(len(tokens) - len(marker) + 1)
                          if tokens[i:i + len(marker)] == marker)
    boundaries.sort()
    parsed = {name: [] for name in FIELDS}
    for number, (i, name, width) in enumerate(boundaries):
        end = boundaries[number + 1][0] if number + 1 < len(boundaries) else len(tokens)
        value = tokens[i + width:end]
        complete = bool(value and value[-1] == '.' and any(token != '.' for token in value))
        while value and value[-1] == '.':
            value = value[:-1]
        parsed[name].append({'value': ' '.join(value), 'complete': complete})
    return parsed


def structural_checks(predictions):
    if not predictions or len({row['id'] for row in predictions}) != len(predictions):
        raise ValueError('Expected nonempty predictions with unique IDs')
    details, matches = [], Counter()
    for row in predictions:
        reference, generated = parse_fields(row['reference']), parse_fields(row['prediction'])
        if any(len(entries) != 1 or not entries[0]['complete'] for entries in reference.values()):
            raise ValueError('Structural checks require the retained complete 14-field reference format')
        missing = [key for key, entries in generated.items() if not entries]
        repeated = [key for key, entries in generated.items() if len(entries) > 1]
        incomplete = [key for key, entries in generated.items() if any(not item['complete'] for item in entries)]
        conflicting = [key for key, entries in generated.items() if
                       len({item['value'] for item in entries if item['complete']}) > 1]
        for key in FIELDS:
            if (len(generated[key]) == 1 and generated[key][0]['complete'] and
                    generated[key][0]['value'] == reference[key][0]['value']):
                matches[key] += 1
        details.append({'id': row['id'], 'missing_fields': missing, 'repeated_fields': repeated,
                        'incomplete_fields': incomplete, 'conflicting_fields': conflicting,
                        'all_fields_once_and_complete': not (missing or repeated or incomplete),
                        'ended_with_incomplete_structure': row['terminated_with_end'] and bool(missing or incomplete)})
    counts = Counter(row['prediction'] for row in predictions)
    lengths = [len(row['prediction'].split()) for row in predictions]
    summary = {'samples': len(predictions), 'unique_predictions': len(counts),
               'largest_identical_group': max(counts.values()),
               'unfinished_predictions': sum(not row['terminated_with_end'] for row in predictions),
               'prediction_tokens': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths) / len(lengths)},
               'repeated_confidence': sum('confidence level' in row['repeated_fields'] for row in details),
               **{key: sum(bool(row[key]) for row in details) for key in
                  ('missing_fields', 'repeated_fields', 'incomplete_fields', 'conflicting_fields',
                   'all_fields_once_and_complete', 'ended_with_incomplete_structure')},
               'reference_field_exact_matches': {key: matches[key] for key in FIELDS}}
    return {'summary': summary, 'cases': details,
            'notes': ['Literal reconstructed-template checks, not clinical correctness.',
                      'EOS with incomplete structure can reflect omissions, not necessarily early stopping.']}


def text_metrics(predictions):
    from nlgeval import NLGEval

    evaluator = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['METEOR', 'SPICE'])
    result = evaluator.compute_metrics([[row['reference'] for row in predictions]],
                                       [row['prediction'] for row in predictions])
    return {key: float(value) for key, value in result.items()}


def generate_reports(model, split, words, config, device):
    import torch

    model.eval()
    reverse = {index: word for word, index in words.items()}
    predictions = []
    with torch.no_grad():
        for index in range(len(split)):
            row = split[index]
            memory = model.encoder(row['image'].unsqueeze(0).to(device))
            result = beam_search(model.decoder, memory, words, config['lambda_parallel'],
                                 config['beam_size'], config['max_decode_steps'])
            caption = split.captions[index][1:split.lengths[index] - 1]
            predictions.append({'id': row['id'], 'reference': ' '.join(reverse[token] for token in caption), **result})
            if (index + 1) % 10 == 0:
                print(f'Generated {index + 1}/{len(split)} reports', flush=True)
    return predictions
