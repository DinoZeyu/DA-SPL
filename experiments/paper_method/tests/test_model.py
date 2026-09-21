"""Synthetic equations/gradient/decoding tests; no pretrained inference or optimization."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_method.model import DASPL, DualAttention, ParallelDecoder, VisualEncoder, mixture_log_probs
from paper_method.losses import objective
from paper_method.decoding import beam_search


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.dropout = nn.Dropout(.5)

    def forward_features(self, images):
        return self.dropout(self.linear(images))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        self.words = {word: i for i, word in enumerate(('<pad>', '<start>', '<end>', 'a', 'b', 'c', 'd', '<unk>'))}

    def captions(self):
        return torch.tensor([[1, 3, 4, 2, 0], [1, 2, 0, 0, 0], [1, 5, 2, 0, 0]]), torch.tensor([4, 2, 3])

    def model(self):
        encoder = VisualEncoder(ToyBackbone(), 8, 8, True)
        return DASPL(encoder, len(self.words), 3, dim=8, heads=2, pad_id=0)

    def test_spatial_attention_and_dual_weight_equations_match_explicit_calculation(self):
        attention = DualAttention(8, 2)
        memory, query = torch.randn(3, 5, 8), torch.randn(3, 8)
        with torch.no_grad():
            attention.head_logits.copy_(torch.tensor([.4, -.2]))
        context, detail = attention(memory, query)
        q = attention.query(query).reshape(3, 2, 4)
        k = attention.key(memory).reshape(3, 5, 2, 4).transpose(1, 2)
        v = attention.value(memory).reshape(3, 5, 2, 4).transpose(1, 2)
        p = ((q[:, :, None] * k).sum(-1) / 2).softmax(-1)
        heads = (p[..., None] * v).sum(2)
        cos = F.cosine_similarity(heads, heads[:, :1], dim=-1).mean(0)
        beta = cos.abs().clamp_min(1e-8).log().mean().exp()
        weights = 2 * attention.head_logits.softmax(0) * (beta - cos).relu()
        expected = attention.output((heads * weights[None, :, None]).reshape(3, 8))
        torch.testing.assert_close(context, expected)
        torch.testing.assert_close(detail['patch_probabilities'].sum(-1), torch.ones(3, 2))
        self.assertEqual(detail['patch_probabilities'].shape, (3, 2, 5))
        torch.testing.assert_close(detail['combined_weights'], weights)

    def test_attention_permutation_mask_and_no_forward_parameter_mutation(self):
        attention = DualAttention(8, 2)
        memory, query = torch.randn(3, 5, 8), torch.randn(3, 8)
        before = {name: value.clone() for name, value in attention.state_dict().items()}
        result, _ = attention(memory, query)
        order = torch.tensor([2, 0, 1])
        permuted, _ = attention(memory[order], query[order])
        torch.testing.assert_close(permuted, result[order])
        active = torch.tensor([True, False, True])
        masked, _ = attention(memory, query, active)
        selected, _ = attention(memory[active], query[active])
        torch.testing.assert_close(masked[active], selected)
        for name, value in attention.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        result.square().sum().backward()
        for parameter in (attention.query.weight, attention.value.weight, attention.head_logits):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_literal_cosine_rectification_zeroes_identical_heads(self):
        attention = DualAttention(4, 2)
        with torch.no_grad():
            attention.query.weight.fill_(.25)
            attention.key.weight.fill_(.25)
            attention.value.weight.fill_(.25)
        context, detail = attention(torch.ones(2, 3, 4), torch.ones(2, 4))
        torch.testing.assert_close(detail['cosine_weights'], torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(context, torch.zeros_like(context), atol=1e-6, rtol=0)

    def test_frozen_backbone_stays_in_eval_while_projection_learns(self):
        model = self.model().train()
        self.assertFalse(model.encoder.backbone.training)
        self.assertTrue(model.encoder.projection.training)
        model.encoder(torch.randn(2, 4, 4)).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.backbone.parameters()))
        self.assertGreater(model.encoder.projection.weight.grad.abs().sum().item(), 0)

    def test_forward_and_exact_stepwise_prefix_have_identical_head_logits(self):
        decoder = ParallelDecoder(8, 8, 2, 0)
        memory = torch.randn(3, 4, 8)
        caps, lengths = self.captions()
        expected1, expected2 = decoder(memory, caps, lengths)
        state = decoder.initial_state(memory)
        for t in range(3):
            first, second, state = decoder.step(memory, caps[:, t], state, t < lengths - 1)
            torch.testing.assert_close(first, expected1[:, t], rtol=0, atol=0)
            torch.testing.assert_close(second, expected2[:, t], rtol=0, atol=0)

    def test_loss_uses_same_next_token_in_both_heads_including_eos(self):
        caps, lengths = self.captions()
        first, second, labels = torch.randn(3, 3, 8, requires_grad=True), torch.randn(3, 3, 8, requires_grad=True), torch.randn(3, 3)
        tags = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.]])
        mask = torch.arange(3)[None] < (lengths - 1)[:, None]
        targets = caps[:, 1:4][mask]
        expected = F.cross_entropy(first[mask], targets) + .5 * F.cross_entropy(second[mask], targets)
        expected += 5 * F.binary_cross_entropy_with_logits(labels, tags)
        actual, parts, count = objective({'primary_logits': first, 'secondary_logits': second, 'tag_logits': labels}, caps, lengths, tags)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(count, 6)
        self.assertIn('primary_eos_ce', parts)
        actual.backward()
        for row, length in enumerate(lengths.tolist()):
            end = length - 2
            self.assertLess(first.grad[row, end, 2].item(), 0)
            self.assertLess(second.grad[row, end, 2].item(), 0)
        torch.testing.assert_close(first.grad[~mask], torch.zeros_like(first.grad[~mask]))
        torch.testing.assert_close(second.grad[~mask], torch.zeros_like(second.grad[~mask]))

    def test_lem_is_differentiable_to_both_report_heads_and_ignores_padding(self):
        model = self.model().eval()
        caps, lengths = self.captions()
        memory = torch.randn(3, 4, 8, requires_grad=True)
        output = model.from_memory(memory, caps, lengths)
        label_loss = F.binary_cross_entropy_with_logits(output['tag_logits'], torch.ones(3, 3))
        label_loss.backward()
        for parameter in (model.decoder.primary_head.weight, model.decoder.secondary_head.weight,
                          model.label_enhancement.classifier.weight, model.decoder.embedding.weight, memory):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        padded = caps.clone()
        for i, length in enumerate(lengths):
            padded[i, length:] = 6
        changed = model.from_memory(memory.detach(), padded, lengths)
        torch.testing.assert_close(changed['tag_logits'], output['tag_logits'])

    def test_model_batch_permutation_and_train_eval_consistency_without_dropout(self):
        model = self.model()
        caps, lengths = self.captions()
        images = torch.randn(3, 4, 4)
        model.train()
        train = model(images, caps, lengths)
        model.eval()
        evaluation = model(images, caps, lengths)
        order = torch.tensor([2, 0, 1])
        permuted = model(images[order], caps[order], lengths[order])
        for key in train:
            torch.testing.assert_close(train[key], evaluation[key])
            torch.testing.assert_close(train[key][order], permuted[key])

    def test_different_image_memories_change_prefix_scores(self):
        model = self.model().eval()
        caps, lengths = self.captions()
        memory = torch.randn(1, 4, 8)
        first = model.from_memory(memory, caps[:1], lengths[:1])['primary_logits']
        changed = model.from_memory(memory + 2, caps[:1], lengths[:1])['primary_logits']
        self.assertGreater((first - changed).abs().max().item(), 1e-4)

    def test_probability_mixture_is_normalized_and_uses_current_heads(self):
        first, second = torch.randn(2, 8), torch.randn(2, 8)
        actual = mixture_log_probs(first, second, .5)
        torch.testing.assert_close(actual.exp(), (first.softmax(-1) + .5 * second.softmax(-1)) / 1.5)
        torch.testing.assert_close(actual.exp().sum(-1), torch.ones(2))

    def scripted_decoder(self, transitions):
        words = self.words

        class Scripted(nn.Module):
            def initial_state(self, memory):
                return ()

            def step(self, memory, previous_word, state):
                scores = memory.new_full((1, len(words)), -100)
                for word, score in transitions[int(previous_word)].items():
                    scores[0, words[word]] = score
                return scores, scores, ()

        return Scripted().eval()

    def test_beam_uses_log_probabilities_and_stops_at_eos(self):
        decoder = self.scripted_decoder({1: {'a': 2., 'b': 1.}, 3: {'<end>': 3.}, 4: {'c': 1.}, 5: {'<end>': 1.}})
        result = beam_search(decoder, torch.zeros(1, 3, 8), self.words, width=2, max_steps=5)
        self.assertEqual(result['prediction'], 'a')
        self.assertEqual(result['token_ids'], [1, 3, 2])
        self.assertTrue(result['terminated_with_end'])
        self.assertLess(result['log_probability'], 0)

    def test_loop_is_reported_unfinished_without_forced_eos_or_rewriting(self):
        decoder = self.scripted_decoder({1: {'a': 5.}, 3: {'a': 5.}})
        result = beam_search(decoder, torch.zeros(1, 3, 8), self.words, width=1, max_steps=5)
        self.assertEqual(result['prediction'], 'a a a a a')
        self.assertFalse(result['terminated_with_end'])

    def test_generation_does_not_leak_previous_image_state_and_has_no_minimum_length(self):
        decoder = self.scripted_decoder({1: {'<end>': 5.}})
        a = beam_search(decoder, torch.zeros(1, 3, 8), self.words, width=1)
        b = beam_search(decoder, torch.ones(1, 3, 8), self.words, width=1)
        self.assertEqual(a, b)
        self.assertEqual(a['prediction'], '')
        decoder.train()
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            beam_search(decoder, torch.zeros(1, 3, 8), self.words)


if __name__ == '__main__':
    unittest.main()
