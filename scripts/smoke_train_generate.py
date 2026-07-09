import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import DataConfig, FiveMillionConfig  # noqa: E402
from data import prepare_datasets, prepare_text_dataset, setup_tokenizer  # noqa: E402
from generation.generate import generate_text  # noqa: E402
from training import train_minimal_llm  # noqa: E402
from utils import set_seed  # noqa: E402


SAMPLE_TEXT = """
Language models learn by predicting the next token from context.
Small models are useful because they make architecture, data, optimization,
and debugging visible to a single reader on a single machine.
This smoke run is intentionally tiny. Its generated text may be messy, but the
important result is that data preparation, training, checkpointing, reporting,
and generation all work in one reproducible path.
"""


def build_sample_texts(text_dir: Path) -> None:
    text_dir.mkdir(parents=True, exist_ok=True)
    for index in range(8):
        (text_dir / f"sample_{index}.txt").write_text((SAMPLE_TEXT + "\n") * 32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--output_dir", default="smoke_runs/latest")
    parser.add_argument("--tokenizer_name", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--train_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--prompt", default="The future of language models")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    run_dir = Path(args.output_dir)
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    text_dir = run_dir / "texts"
    data_dir = run_dir / "processed_data"
    checkpoint_dir = run_dir / "checkpoints"

    build_sample_texts(text_dir)
    prepare_text_dataset(
        input_dir=text_dir,
        output_dir=data_dir,
        tokenizer_name=args.tokenizer_name,
        max_seq_len=args.max_seq_len,
        validation_ratio=0.1,
        overwrite=args.overwrite,
    )

    tokenizer = setup_tokenizer(args.tokenizer_name)
    data_cfg = DataConfig(
        dataset_path=str(data_dir),
        tokenizer_name=args.tokenizer_name,
        seq_length=args.max_seq_len,
        streaming=False,
    )
    train_dataset, val_dataset = prepare_datasets(data_cfg, tokenizer)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    config = FiveMillionConfig(
        max_seq_len=args.max_seq_len,
        vocab_size=tokenizer.vocab_size,
        train_tokens=args.train_tokens,
        batch_size=args.batch_size,
        device=args.device,
        compile_model=False,
        log_every=1,
        eval_steps=5,
        eval_milestones=(1,),
    )

    train_minimal_llm(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=checkpoint_dir,
        report_tokenizer_name=args.tokenizer_name,
        report_prompt=args.prompt,
        report_max_new_tokens=args.max_new_tokens,
    )

    print("\nGenerated sample:")
    generate_text(
        checkpoint=str(checkpoint_dir / "model.pt"),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        device=args.device,
        tokenizer_name=args.tokenizer_name,
    )
    print(f"\nSmoke run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
