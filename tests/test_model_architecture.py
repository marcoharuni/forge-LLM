import gc
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import (  # noqa: E402
    FiftyMillionConfig,
    FiveMillionConfig,
    HundredMillionConfig,
    LLMConfig,
    TwentyFiveMillionConfig,
)
from models import MinimalLLM  # noqa: E402
from utils import count_parameters  # noqa: E402


class ModelArchitectureTest(unittest.TestCase):
    def test_expected_parameter_counts(self):
        expected = [
            (LLMConfig, 88_630_528),
            (FiveMillionConfig, 6_652_800),
            (TwentyFiveMillionConfig, 25_366_272),
            (FiftyMillionConfig, 48_244_224),
            (HundredMillionConfig, 100_169_472),
        ]
        for config_cls, count in expected:
            with self.subTest(config=config_cls.__name__):
                model = MinimalLLM(config_cls())
                self.assertEqual(count_parameters(model), count)
                del model
                gc.collect()

    def test_weight_tying(self):
        model = MinimalLLM(FiveMillionConfig())
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_forward_shape_and_no_nans(self):
        config = LLMConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            n_kv_heads=2,
            max_seq_len=16,
            vocab_size=128,
            dropout=0.0,
        )
        model = MinimalLLM(config)
        x = torch.randint(0, config.vocab_size, (2, 8))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 8, config.vocab_size))
        self.assertFalse(torch.isnan(logits).any().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
