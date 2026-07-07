import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = torch.square(F.relu(x))
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
