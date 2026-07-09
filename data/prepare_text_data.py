import argparse
import csv
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import DatasetDict, load_dataset

from data.loader import setup_tokenizer


SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".csv"}


def discover_input_files(input_dir=None, input_file=None, recursive=False):
    paths = []
    if input_file is not None:
        paths.append(Path(input_file))
    if input_dir is not None:
        directory = Path(input_dir)
        pattern = "**/*" if recursive else "*"
        paths.extend(path for path in directory.glob(pattern) if path.is_file())

    files = sorted(
        path
        for path in paths
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            "No supported input files found. Expected .txt, .jsonl, or .csv files."
        )
    return files


def iter_texts(path, text_column="text", min_chars=1):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) >= min_chars:
            yield text
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
                value = row.get(text_column)
                text = "" if value is None else str(value).strip()
                if len(text) >= min_chars:
                    yield text
        return

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or text_column not in reader.fieldnames:
                raise ValueError(f"{path} must contain a '{text_column}' column.")
            for row in reader:
                value = row.get(text_column)
                text = "" if value is None else str(value).strip()
                if len(text) >= min_chars:
                    yield text
        return

    raise ValueError(f"Unsupported file type: {path.suffix}")


def write_token_chunks(
    files,
    tokenizer,
    jsonl_path,
    max_seq_len,
    text_column="text",
    min_chars=1,
    target_tokens=None,
):
    buffer = []
    documents = 0
    chunks = 0
    tokens = 0

    with Path(jsonl_path).open("w", encoding="utf-8") as output:
        for path in files:
            for text in iter_texts(path, text_column=text_column, min_chars=min_chars):
                documents += 1
                ids = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
                buffer.extend(ids)

                while len(buffer) >= max_seq_len:
                    chunk = buffer[:max_seq_len]
                    del buffer[:max_seq_len]
                    output.write(json.dumps({"input_ids": chunk, "labels": list(chunk)}) + "\n")
                    chunks += 1
                    tokens += len(chunk)
                    if target_tokens is not None and tokens >= target_tokens:
                        return {
                            "documents": documents,
                            "chunks": chunks,
                            "tokens": tokens,
                            "dropped_tail_tokens": len(buffer),
                        }

    return {
        "documents": documents,
        "chunks": chunks,
        "tokens": tokens,
        "dropped_tail_tokens": len(buffer),
    }


def save_domain_dataset(jsonl_path, output_dir, validation_ratio, seed):
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    dataset = dataset.shuffle(seed=seed)
    if validation_ratio > 0.0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=validation_ratio, seed=seed)
        dataset_dict = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )
    else:
        dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(str(output_dir))
    return dataset_dict


def prepare_text_dataset(
    input_dir=None,
    input_file=None,
    output_dir="processed_data/domain_text",
    tokenizer_name="HuggingFaceTB/SmolLM2-135M",
    max_seq_len=512,
    text_column="text",
    validation_ratio=0.1,
    seed=42,
    min_chars=1,
    recursive=False,
    target_tokens=None,
    use_fast=True,
    trust_remote_code=False,
    overwrite=False,
):
    if input_dir is None and input_file is None:
        raise ValueError("Pass --input_dir or --input_file.")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("--validation_ratio must be in [0.0, 1.0).")
    if max_seq_len <= 0:
        raise ValueError("--max_seq_len must be positive.")
    if min_chars < 0:
        raise ValueError("--min_chars must be non-negative.")
    if target_tokens is not None and target_tokens <= 0:
        raise ValueError("--target_tokens must be positive when provided.")

    files = discover_input_files(input_dir, input_file, recursive)
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"{output_path} exists and is not a directory.")
    output_exists = output_path.exists() and any(output_path.iterdir())
    if output_exists and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists and is not empty. "
            "Pass --overwrite to replace it."
        )

    tokenizer = setup_tokenizer(
        tokenizer_name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )

    if output_exists:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        stats = write_token_chunks(
            files=files,
            tokenizer=tokenizer,
            jsonl_path=tmp_path,
            max_seq_len=max_seq_len,
            text_column=text_column,
            min_chars=min_chars,
            target_tokens=target_tokens,
        )
        if stats["chunks"] == 0:
            raise ValueError(
                "No full token chunks were produced. Add more text or lower --max_seq_len."
            )

        dataset = save_domain_dataset(tmp_path, output_path, validation_ratio, seed)
    finally:
        tmp_path.unlink(missing_ok=True)

    metadata = {
        "source": "local_text",
        "input_files": [str(path) for path in files],
        "output_dir": str(output_path),
        "max_seq_len": max_seq_len,
        "tokenizer_name": tokenizer_name,
        "text_column": text_column,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "min_chars": min_chars,
        "recursive": recursive,
        "target_tokens": target_tokens,
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "tokens": stats["tokens"],
        "dropped_tail_tokens": stats["dropped_tail_tokens"],
        "splits": {name: len(split) for name, split in dataset.items()},
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_path / "prep_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_dir", default="processed_data/domain_text")
    parser.add_argument("--tokenizer_name", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_chars", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--target_tokens", type=int, default=None)
    parser.add_argument("--use_fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = prepare_text_dataset(
        input_dir=args.input_dir,
        input_file=args.input_file,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        max_seq_len=args.max_seq_len,
        text_column=args.text_column,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        min_chars=args.min_chars,
        recursive=args.recursive,
        target_tokens=args.target_tokens,
        use_fast=args.use_fast,
        trust_remote_code=args.trust_remote_code,
        overwrite=args.overwrite,
    )
    print(
        f"Saved {metadata['chunks']:,} chunks "
        f"({metadata['tokens']:,} tokens) to {metadata['output_dir']}"
    )


if __name__ == "__main__":
    main()
