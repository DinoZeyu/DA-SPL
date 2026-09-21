"""Independent same-next-token branch supervision and report-derived label loss."""

import torch
from torch.nn import functional as F


def objective(outputs, captions, lengths, tags, coefficient=.5, tag_weight=5.):
    first, second = outputs['primary_logits'], outputs['secondary_logits']
    if first.shape != second.shape or first.shape[:2] != (captions.size(0), int(lengths.max()) - 1):
        raise ValueError('Branch logits do not match caption lengths')
    steps = lengths.reshape(-1).to(first.device) - 1
    mask = torch.arange(first.size(1), device=first.device)[None] < steps[:, None]
    targets = captions[:, 1:1 + first.size(1)]
    loss1 = F.cross_entropy(first[mask].float(), targets[mask])
    loss2 = F.cross_entropy(second[mask].float(), targets[mask])
    tag_loss = F.binary_cross_entropy_with_logits(outputs['tag_logits'].float(), tags.float())
    total = loss1 + coefficient * loss2 + tag_weight * tag_loss
    eos_rows = torch.arange(len(steps), device=first.device)
    eos_targets = targets[eos_rows, steps - 1]
    metrics = {'primary_ce': loss1.detach(), 'secondary_ce': loss2.detach(), 'tag_bce': tag_loss.detach(),
               'primary_eos_ce': F.cross_entropy(first[eos_rows, steps - 1].float(), eos_targets).detach(),
               'secondary_eos_ce': F.cross_entropy(second[eos_rows, steps - 1].float(), eos_targets).detach()}
    return total, metrics, int(mask.sum())
