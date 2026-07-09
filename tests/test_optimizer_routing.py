import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import LLMConfig  # noqa: E402
from models import MinimalLLM  # noqa: E402
from optimizers import Muon  # noqa: E402
from training.trainer import setup_muon_optimizer  # noqa: E402


class OptimizerRoutingTest(unittest.TestCase):
    def _tiny_model(self):
        config = LLMConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            n_kv_heads=2,
            max_seq_len=16,
            vocab_size=128,
            device="cpu",
        )
        return MinimalLLM(config), config

    def test_qkvo_and_ffn_route_to_muon(self):
        model, config = self._tiny_model()
        optimizers = setup_muon_optimizer(model, config)
        muon = next(optimizer for optimizer in optimizers if isinstance(optimizer, Muon))
        muon_ids = {id(param) for group in muon.param_groups for param in group["params"]}
        params = dict(model.named_parameters())

        self.assertIn(id(params["transformer_blocks.0.attention.qkvo_proj"]), muon_ids)
        self.assertIn(id(params["transformer_blocks.0.feed_forward.up_proj.weight"]), muon_ids)
        self.assertIn(id(params["transformer_blocks.0.feed_forward.down_proj.weight"]), muon_ids)

    def test_embeddings_and_norms_route_to_adamw(self):
        model, config = self._tiny_model()
        optimizers = setup_muon_optimizer(model, config)
        adamw = next(optimizer for optimizer in optimizers if isinstance(optimizer, torch.optim.AdamW))
        adamw_ids = {id(param) for group in adamw.param_groups for param in group["params"]}
        params = dict(model.named_parameters())

        self.assertIn(id(params["token_embedding.weight"]), adamw_ids)
        self.assertIn(id(params["norm.weight"]), adamw_ids)
        self.assertIn(id(params["transformer_blocks.0.norm1.weight"]), adamw_ids)

    def test_muon_step_runs_on_2d_parameter(self):
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = Muon([param], lr=0.01)
        param.grad = torch.randn_like(param)
        before = param.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, param.detach()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
