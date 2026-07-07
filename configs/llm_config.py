from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LLMConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 22
    d_ff: int = 2048
    n_kv_heads: int = 4

    max_seq_len: int = 2048
    vocab_size: int = 49152

    device: str = "auto"
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 8_000_000

    muon_lr: float = 0.024
    muon_momentum: float = 0.95
    adamw_lr: float = 0.006
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"

    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None

    weight_decay: float = 0.2
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True

    log_every: int = 100

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_model > 0, "d_model must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        self.d_k = self.d_model // self.n_heads


@dataclass
class FiveMillionConfig(LLMConfig):
    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 512
    n_kv_heads: int = 1
    train_tokens: int = 134_000_000


@dataclass
class TwentyFiveMillionConfig(LLMConfig):
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1536
    n_kv_heads: int = 4
    train_tokens: int = 507_000_000


@dataclass
class FiftyMillionConfig(LLMConfig):
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    n_kv_heads: int = 4
    train_tokens: int = 965_000_000


@dataclass
class HundredMillionConfig(LLMConfig):
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 26
    d_ff: int = 2048
    n_kv_heads: int = 4
    train_tokens: int = 2_000_000_000
