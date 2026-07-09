import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.sampler import sample_next_token, top_k_filter, top_p_filter  # noqa: E402


class SamplerTest(unittest.TestCase):
    def test_greedy_returns_argmax(self):
        logits = torch.tensor([[0.1, 0.2, 3.0, -1.0]])
        input_ids = torch.tensor([[0, 1]])
        token = sample_next_token(logits, input_ids, greedy=True)
        self.assertEqual(token.item(), 2)

    def test_top_k_filter_keeps_only_k_tokens(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        filtered = top_k_filter(logits, top_k=2)
        self.assertTrue(torch.isneginf(filtered[0, 0]))
        self.assertTrue(torch.isneginf(filtered[0, 1]))
        self.assertFalse(torch.isneginf(filtered[0, 2]))
        self.assertFalse(torch.isneginf(filtered[0, 3]))

    def test_top_p_filter_keeps_at_least_one_token(self):
        logits = torch.tensor([[10.0, 1.0, 0.5, 0.0]])
        filtered = top_p_filter(logits, top_p=0.1)
        self.assertFalse(torch.isneginf(filtered[0, 0]))
        self.assertTrue(torch.isneginf(filtered[0, 1:]).all().item())

    def test_sampling_shape(self):
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[0, 1]])
        token = sample_next_token(logits, input_ids, temperature=1.0, top_k=2)
        self.assertEqual(token.shape, (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
