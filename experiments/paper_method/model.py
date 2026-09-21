"""ConViT, dual-weight attention, three-cell PLN, and report-based LEM.

See docs/paper_method_contract.md for equation mappings and unresolved choices.
No imports from the retired implementation or experimental repair branches.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class VisualEncoder(nn.Module):
    def __init__(self, backbone, feature_dim, hidden_dim, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            tokens = self.backbone.forward_features(images)
        if tokens.ndim != 3 or tokens.size(1) < 2:
            raise ValueError('Expected CLS and spatial ConViT tokens, not one pooled vector')
        return self.projection(tokens)


class DualAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError('Feature dimension must be divisible by head count')
        self.heads, self.head_dim = heads, dim // heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.head_logits = nn.Parameter(torch.zeros(heads))

    def forward(self, memory, query, active=None):
        batch, patches, dim = memory.shape
        q = self.query(query).reshape(batch, self.heads, self.head_dim)
        k = self.key(memory).reshape(batch, patches, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(memory).reshape(batch, patches, self.heads, self.head_dim).transpose(1, 2)
        probabilities = ((q.unsqueeze(2) * k).sum(-1) / self.head_dim ** .5).softmax(-1)
        head_context = (probabilities.unsqueeze(-1) * v).sum(2)
        learned = self.head_logits.softmax(0) * self.heads
        base = self.head_logits.argmax()
        cosine = F.cosine_similarity(head_context.float(), head_context[:, base:base + 1].float(), dim=-1, eps=1e-8)
        if active is None:
            active = torch.ones(batch, dtype=torch.bool, device=memory.device)
        if active.shape != (batch,) or not active.any():
            raise ValueError('Attention needs at least one active report')
        # Eq. 6 says batch average; Eq. 7 is a geometric mean over heads.
        mean_cosine = cosine[active].mean(0)
        beta = mean_cosine.abs().clamp_min(1e-8).log().mean().exp()
        correction = (beta - mean_cosine).relu()
        weights = learned * correction
        weighted = head_context * weights.to(head_context.dtype)[None, :, None]
        context = self.output(weighted.reshape(batch, dim))
        return context, {'patch_probabilities': probabilities, 'learned_weights': learned,
                         'mean_cosine': mean_cosine, 'beta': beta, 'cosine_weights': correction,
                         'combined_weights': weights}


def mixture_log_probs(first, second, coefficient):
    """Same-step probability mixture, an explicit inference choice, not a lag."""
    if not 0 < coefficient <= 1:
        raise ValueError('Parallel coefficient must be in (0, 1]')
    first, second = first.float().log_softmax(-1), second.float().log_softmax(-1)
    weight = first.new_tensor(coefficient)
    return torch.logaddexp(first, second + weight.log()) - (1 + weight).log()


class ParallelDecoder(nn.Module):
    def __init__(self, vocab_size, dim, heads, pad_id):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.attention = DualAttention(dim, heads)
        self.visual_cell = nn.LSTMCell(2 * dim, dim)
        self.primary_cell = nn.LSTMCell(3 * dim, dim)
        self.secondary_cell = nn.LSTMCell(3 * dim, dim)
        self.primary_head = nn.Linear(dim, vocab_size)
        self.secondary_head = nn.Linear(dim, vocab_size)

    def initial_state(self, memory):
        return tuple(memory.new_zeros(memory.size(0), self.dim) for _ in range(4))

    def step(self, memory, previous_word, state, active=None):
        h1, c1, h2, c2 = state
        embedding = self.embedding(previous_word)
        global_feature = memory[:, 0]
        context1, _ = self.attention(memory, h1, active)
        h1, c1 = self.visual_cell(torch.cat((global_feature, context1), -1), (h1, c1))
        h2, c2 = self.primary_cell(torch.cat((global_feature, h1, embedding), -1), (h2, c2))
        context2, _ = self.attention(memory, h2, active)
        # Eq. 12 conditions the second cell on the current primary state.
        h3, _ = self.secondary_cell(torch.cat((context2, h2, embedding), -1), (h2, c2))
        return self.primary_head(h2), self.secondary_head(h3), (h1, c1, h2, c2)

    def forward(self, memory, captions, lengths):
        steps = lengths.reshape(-1).to(memory.device) - 1
        if (steps < 1).any() or steps.max() >= captions.size(1):
            raise ValueError('Caption lengths must include START and END')
        state, first, second = self.initial_state(memory), [], []
        for t in range(int(steps.max())):
            a, b, state = self.step(memory, captions[:, t], state, active=t < steps)
            first.append(a)
            second.append(b)
        return torch.stack(first, 1), torch.stack(second, 1)


class LabelEnhancement(nn.Module):
    def __init__(self, dim, tag_count):
        super().__init__()
        self.category_lstm = nn.LSTM(dim, dim, batch_first=True)
        self.classifier = nn.Linear(dim, tag_count)

    def forward(self, report_embeddings, lengths):
        packed = pack_padded_sequence(report_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.category_lstm(packed)
        return self.classifier(hidden[-1])


class DASPL(nn.Module):
    def __init__(self, encoder, vocab_size, tag_count, dim=512, heads=8, pad_id=0, coefficient=.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = ParallelDecoder(vocab_size, dim, heads, pad_id)
        self.label_enhancement = LabelEnhancement(dim, tag_count)
        self.coefficient = coefficient

    def from_memory(self, memory, captions, lengths):
        first, second = self.decoder(memory, captions, lengths)
        distribution = mixture_log_probs(first, second, self.coefficient).exp()
        # Soft generated embeddings keep LEM gradients connected to both report heads.
        embeddings = distribution @ self.decoder.embedding.weight.float()
        tags = self.label_enhancement(embeddings, lengths.reshape(-1) - 1)
        return {'primary_logits': first, 'secondary_logits': second, 'tag_logits': tags}

    def forward(self, images, captions, lengths):
        return self.from_memory(self.encoder(images), captions, lengths)


def build_model(config, words, tags, device):
    import timm

    path = Path(config['pretrained_weights'])
    if not path.is_file():
        raise ValueError(f'Cached ConViT weights required; no download is attempted: {path}')
    backbone = timm.create_model('convit_base', pretrained=False, num_classes=0)
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = state.get('model', state)
    state = {key: value for key, value in state.items() if key not in ('head.weight', 'head.bias')}
    backbone.load_state_dict(state, strict=True)
    encoder = VisualEncoder(backbone, backbone.num_features, config['hidden_dim'], config['freeze_backbone'])
    return DASPL(encoder, len(words), len(tags) - 4, config['hidden_dim'], config['attention_heads'],
                 words['<pad>'], config['lambda_parallel']).to(device)
