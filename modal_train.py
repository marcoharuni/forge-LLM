from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import modal


APP_NAME = "forge-llm-train"
REMOTE_REPO = "/root/forge-LLM"
VOLUME_PATH = Path("/vol/forge-llm")

volume = modal.Volume.from_name("forge-llm-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "datasets",
        "transformers",
        "torchtune",
        "torchao",
        "matplotlib",
        "numpy",
        "tqdm",
    )
    .env(
        {
            "HF_HOME": str(VOLUME_PATH / "hf_cache"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "processed_data",
            "checkpoints",
            "plots",
            "logs",
            "*.pt",
            "*.pth",
        ],
    )
)

app = modal.App(APP_NAME)


def _run(args: list[str]) -> None:
    VOLUME_PATH.mkdir(parents=True, exist_ok=True)
    subprocess.run(args, cwd=VOLUME_PATH, check=True)
    volume.commit()


@app.function(image=image, volumes={VOLUME_PATH: volume}, gpu="L4", timeout=60 * 30)
def check_gpu() -> None:
    _run(
        [
            "python",
            "-c",
            (
                "import torch; "
                "print(torch.__version__); "
                "print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
            ),
        ]
    )


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 4)
def prepare_mix(target_tokens: int = 1_000_000, max_seq_len: int = 512) -> str:
    _run(
        [
            "python",
            f"{REMOTE_REPO}/data/prepare_mix_data.py",
            "--target_tokens",
            str(target_tokens),
            "--max_seq_len",
            str(max_seq_len),
        ]
    )
    return str(VOLUME_PATH / "processed_data" / f"pretrain_mix_{target_tokens}")


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="L4",
    timeout=60 * 60 * 12,
)
def train(
    run_name: str = "smoke-l4",
    config: str = "5m",
    train_tokens: int = 50_000,
    dataset_tokens: int = 1_000_000,
    max_seq_len: int = 512,
    batch_size: int = 2,
    compile_model: bool = False,
    resume: bool = True,
) -> str:
    dataset_path = VOLUME_PATH / "processed_data" / f"pretrain_mix_{dataset_tokens}"
    output_dir = VOLUME_PATH / "checkpoints" / run_name
    checkpoint_path = output_dir / "training_state.pt"

    args = [
        "python",
        f"{REMOTE_REPO}/train_llm.py",
        "--config",
        config,
        "--train_tokens",
        str(train_tokens),
        "--batch_size",
        str(batch_size),
        "--max_seq_len",
        str(max_seq_len),
        "--device",
        "cuda",
        "--compile",
        str(compile_model).lower(),
        "--dataset_path",
        str(dataset_path),
        "--output_dir",
        str(output_dir),
    ]
    if resume and checkpoint_path.exists():
        args.extend(["--resume_checkpoint", str(checkpoint_path)])

    _run(args)
    return str(output_dir)


@app.function(image=image, volumes={VOLUME_PATH: volume}, gpu="L4", timeout=60 * 15)
def generate(
    run_name: str = "smoke-l4",
    prompt: str = "The future of language models",
    max_new_tokens: int = 50,
) -> None:
    _run(
        [
            "python",
            f"{REMOTE_REPO}/generation/generate.py",
            "--checkpoint",
            str(VOLUME_PATH / "checkpoints" / run_name / "model.pt"),
            "--prompt",
            prompt,
            "--max_new_tokens",
            str(max_new_tokens),
            "--temperature",
            "0.7",
            "--top_k",
            "50",
            "--top_p",
            "0.9",
            "--device",
            "cuda",
        ]
    )


@app.local_entrypoint()
def main(
    action: str = "smoke",
    run_name: str = "smoke-l4",
    config: str = "5m",
    train_tokens: int = 50_000,
    dataset_tokens: int = 1_000_000,
    max_seq_len: int = 512,
    batch_size: int = 2,
    compile_model: bool = False,
    resume: bool = True,
    prompt: Optional[str] = None,
) -> None:
    if action == "check":
        check_gpu.remote()
    elif action == "prepare":
        print(prepare_mix.remote(dataset_tokens, max_seq_len))
    elif action == "train":
        print(
            train.remote(
                run_name,
                config,
                train_tokens,
                dataset_tokens,
                max_seq_len,
                batch_size,
                compile_model,
                resume,
            )
        )
    elif action == "generate":
        generate.remote(run_name, prompt or "The future of language models")
    elif action == "smoke":
        check_gpu.remote()
        print(prepare_mix.remote(dataset_tokens, max_seq_len))
        print(
            train.remote(
                run_name,
                config,
                train_tokens,
                dataset_tokens,
                max_seq_len,
                batch_size,
                compile_model,
                resume,
            )
        )
        generate.remote(run_name, prompt or "The future of language models")
    else:
        raise ValueError("action must be one of: check, prepare, train, generate, smoke")
