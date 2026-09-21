"""Per-image log-probability beam search using the exact training decoder step."""

from dataclasses import dataclass

import torch

from .model import mixture_log_probs


@dataclass
class Candidate:
    tokens: list
    score: float
    state: tuple
    ended: bool = False


@torch.no_grad()
def beam_search(decoder, memory, words, coefficient=.5, width=5, max_steps=161):
    if memory.size(0) != 1 or width < 1 or max_steps < 1:
        raise ValueError('Beam search requires one image and positive limits')
    if decoder.training:
        raise ValueError('Generation requires eval mode')
    bos, eos, pad = (words[key] for key in ('<start>', '<end>', '<pad>'))
    beams = [Candidate([bos], 0., decoder.initial_state(memory))]
    for _ in range(max_steps):
        candidates = []
        for beam in beams:
            if beam.ended:
                candidates.append(beam)
                continue
            token = torch.tensor([beam.tokens[-1]], device=memory.device)
            first, second, state = decoder.step(memory, token, beam.state)
            scores = mixture_log_probs(first, second, coefficient)[0]
            scores[[bos, pad]] = -torch.inf
            values, indices = scores.topk(min(width, scores.numel() - 2))
            for value, index in zip(values.tolist(), indices.tolist()):
                candidates.append(Candidate(beam.tokens + [index], beam.score + value, state, index == eos))
        beams = sorted(candidates, key=lambda beam: beam.score, reverse=True)[:width]
        if all(beam.ended for beam in beams):
            break
    chosen = max(beams, key=lambda beam: beam.score)
    reverse = {index: word for word, index in words.items()}
    text = ' '.join(reverse[token] for token in chosen.tokens if token not in (bos, eos, pad))
    return {'prediction': text, 'token_ids': chosen.tokens, 'terminated_with_end': chosen.ended,
            'log_probability': chosen.score}
