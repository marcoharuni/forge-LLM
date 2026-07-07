import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from configs import DataConfig
from data.loader import setup_tokenizer


def stream_subset(name, limit):
    return load_dataset(
        "HuggingFaceTB/smollm-corpus",
        name,
        split="train",
        streaming=True,
    ).take(limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens", type=int, default=22_000_000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--tokenizer_name", default="HuggingFaceTB/SmolLM2-135M")
    args = parser.parse_args()

    cfg = DataConfig(seq_length=args.max_seq_len, tokenizer_name=args.tokenizer_name)
    tokenizer = setup_tokenizer(cfg.tokenizer_name, cfg.use_fast, cfg.trust_remote_code)
    target_docs = max(100, int(args.target_tokens / 1000 * 2.0))
    fineweb_docs = int(target_docs * 0.7)
    cosmopedia_docs = target_docs - fineweb_docs
    output_dir = Path("processed_data") / f"pretrain_mix_{args.target_tokens}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_written = 0
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
        for subset, limit in [("fineweb-edu-dedup", fineweb_docs), ("cosmopedia-v2", cosmopedia_docs)]:
            for row in stream_subset(subset, limit):
                ids = tokenizer(row.get("text", ""), add_special_tokens=False)["input_ids"]
                for i in range(0, len(ids) - args.max_seq_len + 1, args.max_seq_len):
                    chunk = ids[i : i + args.max_seq_len]
                    record = {"input_ids": chunk, "labels": list(chunk)}
                    tmp.write(json.dumps(record) + "\n")
                    tokens_written += len(chunk)
                    if tokens_written >= args.target_tokens:
                        break
                if tokens_written >= args.target_tokens:
                    break
            if tokens_written >= args.target_tokens:
                break

    dataset = load_dataset("json", data_files=tmp_path, split="train").shuffle(seed=42)
    dataset.save_to_disk(str(output_dir))
    metadata = {
        "target_tokens": args.target_tokens,
        "max_seq_len": args.max_seq_len,
        "tokenizer_name": args.tokenizer_name,
        "mix_ratios": {"fineweb-edu-dedup": 0.7, "cosmopedia-v2": 0.3},
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "prep_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(dataset)} chunks to {output_dir}")


if __name__ == "__main__":
    main()
