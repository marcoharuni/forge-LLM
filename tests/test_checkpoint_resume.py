import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import LLMConfig  # noqa: E402
from models import MinimalLLM  # noqa: E402
from training.trainer import (  # noqa: E402
    _load_training_state,
    _save_training_checkpoint,
    setup_muon_optimizer,
)


class CheckpointResumeTest(unittest.TestCase):
    def test_training_state_contains_resume_fields(self):
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
        model = MinimalLLM(config)
        optimizers = setup_muon_optimizer(model, config)
        schedulers = [LambdaLR(optimizer, lambda _: 1.0) for optimizer in optimizers]
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        metrics = [
            {"type": "eval", "step": 1, "val_loss": 3.0},
            {"type": "eval", "step": 2, "val_loss": 2.5},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_state.pt"
            _save_training_checkpoint(
                path=path,
                model=model,
                config=config,
                optimizers=optimizers,
                schedulers=schedulers,
                scaler=scaler,
                steps=2,
                tokens_seen=512,
                micro_steps=2,
                metrics_history=metrics,
                eval_seen={1, 2},
                training_time=12.5,
                extra_config={"test": True},
            )
            state = _load_training_state(path)

        self.assertIn("model_state_dict", state)
        self.assertEqual(len(state["optimizer_state_dicts"]), len(optimizers))
        self.assertEqual(len(state["scheduler_state_dicts"]), len(schedulers))
        self.assertEqual(state["steps"], 2)
        self.assertEqual(state["tokens_seen"], 512)
        self.assertEqual(state["best_val"]["val_loss"], 2.5)
        self.assertIn("rng_state", state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
