from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_latest_dataset(base_dir: str = "./processed_data") -> str:
    base = Path(base_dir)
    if not base.exists():
        return str(base / "pretrain_mix_latest")

    mix_dirs = [p for p in base.glob("pretrain_mix_*") if p.is_dir()]
    candidates = mix_dirs or [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return str(base / "pretrain_mix_latest")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


@dataclass
class DataConfig:
    dataset_path: str = "auto"
    dataset_name: str = "cosmopedia-v2"
    split: str = "train"
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M"
    use_fast: bool = True
    trust_remote_code: bool = False
    seq_length: int = 512
    num_samples: Optional[int] = None
    text_column: str = "text"
    cache_dir: str = "./hf_cache"
    num_proc: Optional[int] = None
    streaming: bool = True

    def __post_init__(self) -> None:
        if self.dataset_path == "auto":
            self.dataset_path = get_latest_dataset()
        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("num_samples must be positive when provided")
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")
        if not self.split:
            raise ValueError("split must be non-empty")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must be non-empty")
        if not self.text_column:
            raise ValueError("text_column must be non-empty")
