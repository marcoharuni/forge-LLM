from pathlib import Path

from datasets import DatasetDict, IterableDataset, load_dataset, load_from_disk
from transformers import AutoTokenizer


def setup_tokenizer(tokenizer_name, use_fast=True, trust_remote_code=False):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_smollm_corpus(config):
    return load_dataset(
        "HuggingFaceTB/smollm-corpus",
        config.dataset_name,
        split=config.split,
        streaming=config.streaming,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
    )


def tokenize_and_chunk(dataset, tokenizer, config):
    text_column = config.text_column
    seq_length = config.seq_length

    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples[text_column],
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        all_ids = []
        for ids in tokenized["input_ids"]:
            all_ids.extend(ids)
        total_length = (len(all_ids) // seq_length) * seq_length
        input_ids = [
            all_ids[i : i + seq_length]
            for i in range(0, total_length, seq_length)
        ]
        return {"input_ids": input_ids}

    batched = True
    remove_columns = None
    if not isinstance(dataset, IterableDataset):
        remove_columns = dataset.column_names
    return dataset.map(
        tokenize_batch,
        batched=batched,
        remove_columns=remove_columns,
        num_proc=None if config.streaming else config.num_proc,
    )


def finalize_dataset(dataset, streaming=False):
    def add_labels(examples):
        examples["labels"] = list(examples["input_ids"])
        return examples

    dataset = dataset.map(add_labels, batched=True)
    if streaming:
        dataset = list(dataset)
    if hasattr(dataset, "set_format"):
        columns = ["input_ids", "labels"]
        if "attention_mask" in getattr(dataset, "column_names", []):
            columns.append("attention_mask")
        dataset.set_format(type="torch", columns=columns)
    return dataset


def prepare_datasets(config, tokenizer):
    path = Path(config.dataset_path)
    if path.exists() and path.is_dir():
        if (path / "train").exists() and (path / "val").exists():
            train = load_from_disk(str(path / "train"))
            val = load_from_disk(str(path / "val"))
            train.set_format(type="torch", columns=["input_ids", "labels"])
            val.set_format(type="torch", columns=["input_ids", "labels"])
            return train, val
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            train = dataset["train"]
            val = dataset["validation"] if "validation" in dataset else dataset.get("val")
            if val is None:
                split = train.train_test_split(test_size=0.1, seed=42)
                train, val = split["train"], split["test"]
        else:
            split = dataset.train_test_split(test_size=0.1, seed=42)
            train, val = split["train"], split["test"]
        if "labels" not in train.column_names:
            train = train.map(lambda examples: {"labels": list(examples["input_ids"])}, batched=True)
        if "labels" not in val.column_names:
            val = val.map(lambda examples: {"labels": list(examples["input_ids"])}, batched=True)
        train.set_format(type="torch", columns=["input_ids", "labels"])
        val.set_format(type="torch", columns=["input_ids", "labels"])
        return train, val

    dataset = load_smollm_corpus(config)
    if config.num_samples is not None:
        dataset = dataset.take(config.num_samples) if config.streaming else dataset.select(range(config.num_samples))
    tokenized = tokenize_and_chunk(dataset, tokenizer, config)
    finalized = finalize_dataset(tokenized, streaming=config.streaming)
    if isinstance(finalized, list):
        from datasets import Dataset

        finalized = Dataset.from_list(finalized)
        finalized.set_format(type="torch", columns=["input_ids", "labels"])
    split = finalized.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]
