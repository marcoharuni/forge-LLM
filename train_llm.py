import argparse
import hashlib
import importlib
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs import (
    DataConfig,
    FiftyMillionConfig,
    FiveMillionConfig,
    HundredMillionConfig,
    LLMConfig,
    TwentyFiveMillionConfig,
)
from data import prepare_datasets, setup_tokenizer
from training import DEVICE_CHOICES, train_minimal_llm
from utils import set_seed


CONFIGS = {
    "default": LLMConfig,
    "5m": FiveMillionConfig,
    "25m": TwentyFiveMillionConfig,
    "50m": FiftyMillionConfig,
    "100m": HundredMillionConfig,
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def import_config_class(dotted_path):
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_eval_milestones(train_tokens):
    if train_tokens <= 8_000_000:
        return (0, 25, 50, 75, 100, 150, 200, 300, 400)
    if train_tokens <= 25_000_000:
        return (0, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500)
    if train_tokens <= 50_000_000:
        return (0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000)
    if train_tokens <= 100_000_000:
        return (0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000)
    return (0, 500, 1000, 2000, 4000, 8000, 12000, 20000, 30000, 40000, 50000)


def auto_log_every(train_tokens):
    if train_tokens <= 8_000_000:
        return 25
    if train_tokens <= 25_000_000:
        return 50
    if train_tokens <= 50_000_000:
        return 100
    if train_tokens <= 100_000_000:
        return 250
    return 1000


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)


def _validate_local_metadata(dataset_path, max_seq_len):
    path = Path(dataset_path)
    for metadata_path in (path / "prep_metadata.json", path / "cache_config.json"):
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("max_seq_len") != max_seq_len:
            raise ValueError(
                f"Dataset {dataset_path} was prepared with max_seq_len="
                f"{metadata.get('max_seq_len')}, but config.max_seq_len={max_seq_len}. "
                "Reprepare the dataset with the matching sequence length."
            )


def _cache_key(config, data_cfg):
    payload = {
        "dataset_name": data_cfg.dataset_name,
        "split": data_cfg.split,
        "tokenizer_name": data_cfg.tokenizer_name,
        "seq_length": data_cfg.seq_length,
        "num_samples": data_cfg.num_samples,
        "text_column": data_cfg.text_column,
        "vocab_size": config.vocab_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _cache_streamed_datasets(cache_dir, train_dataset, val_dataset, config, data_cfg):
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(cache_dir / "train"))
    val_dataset.save_to_disk(str(cache_dir / "val"))
    metadata = {
        "max_seq_len": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "dataset_name": data_cfg.dataset_name,
        "tokenizer_name": data_cfg.tokenizer_name,
        "num_samples": data_cfg.num_samples,
        "text_column": data_cfg.text_column,
    }
    (cache_dir / "cache_config.json").write_text(json.dumps(metadata, indent=2))


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["default", "5m", "25m", "50m", "100m"], default="default")
    parser.add_argument("--config_class", default=None)
    parser.add_argument("--muon_lr", type=float, default=None)
    parser.add_argument("--adamw_lr", type=float, default=None)
    parser.add_argument("--train_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--compile", type=str_to_bool, default=None)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--warmup", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--load_checkpoint", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    config_cls = import_config_class(args.config_class) if args.config_class else CONFIGS[args.config]
    config = config_cls()

    for arg_name, field_name in [
        ("muon_lr", "muon_lr"),
        ("adamw_lr", "adamw_lr"),
        ("train_tokens", "train_tokens"),
        ("batch_size", "batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("eval_every", "eval_every"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, field_name, value)
    if args.compile is not None:
        config.compile_model = args.compile
    config.compile_warmup = args.warmup
    config.device = args.device
    config.eval_milestones = build_eval_milestones(config.train_tokens)
    config.eval_every = None
    config.log_every = args.log_every if args.log_every != 100 else auto_log_every(config.train_tokens)

    docs = max(100, int(config.train_tokens / 1000 * 2.0))
    data_cfg = DataConfig(seq_length=config.max_seq_len, num_samples=docs)
    if args.dataset_path is not None:
        data_cfg.dataset_path = args.dataset_path

    try:
        tokenizer = setup_tokenizer(
            data_cfg.tokenizer_name,
            use_fast=data_cfg.use_fast,
            trust_remote_code=data_cfg.trust_remote_code,
        )
    except Exception as exc:
        raise SystemExit(
            "Tokenizer loading failed. Make sure the tokenizer is available locally "
            "or that Hugging Face network access is enabled, then prepare a local "
            "dataset with `python data/prepare_mix_data.py --target_tokens 22000000` "
            "or pass `--dataset_path` to a saved dataset directory. "
            f"Original error: {exc}"
        ) from exc
    config.vocab_size = tokenizer.vocab_size

    dataset_path = Path(data_cfg.dataset_path)
    cache_dir = None
    should_cache_stream = False
    if dataset_path.exists() and dataset_path.is_dir():
        _validate_local_metadata(dataset_path, config.max_seq_len)
    else:
        cache_dir = Path("./processed_data") / _cache_key(config, data_cfg)
        if cache_dir.exists():
            data_cfg.dataset_path = str(cache_dir)
        else:
            should_cache_stream = True

    try:
        train_dataset, val_dataset = prepare_datasets(data_cfg, tokenizer)
    except Exception as exc:
        raise SystemExit(
            "Dataset preparation failed. Prepare a local dataset with "
            "`python data/prepare_mix_data.py --target_tokens 22000000` or pass "
            "`--dataset_path` to a saved Hugging Face dataset directory. "
            f"Original error: {exc}"
        ) from exc
    if should_cache_stream and cache_dir is not None:
        _cache_streamed_datasets(cache_dir, train_dataset, val_dataset, config, data_cfg)

    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    train_minimal_llm(
        config,
        train_loader,
        val_loader,
        output_dir=args.output_dir,
        load_weights_path=args.load_checkpoint,
    )


if __name__ == "__main__":
    main()
