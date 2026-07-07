import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

from .components import SquaredReLUFeedForward


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=10000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rope(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float,
        n_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.num_key_value_groups = n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        self.q_size = d_model
        self.kv_size = self.n_kv_heads * self.d_k
        self.o_size = d_model
        self.qkv_size = self.q_size + 2 * self.kv_size
        self.dropout = dropout

        self.qkvo_proj = nn.Parameter(torch.empty(self.qkv_size + self.o_size, d_model))
        nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.rope = Rotary(dim=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = F.linear(x, self.qkvo_proj[: self.qkv_size])
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)

        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = F.linear(attn, self.qkvo_proj[self.qkv_size :])
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        n_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
        )
        self.feed_forward = SquaredReLUFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x
