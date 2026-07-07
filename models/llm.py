import math

import torch
import torch.nn as nn

from configs import LLMConfig
from .layers import TransformerBlock


class MinimalLLM(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    n_kv_heads=config.n_kv_heads,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        del batch_size
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}."
            )

        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits
