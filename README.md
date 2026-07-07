# forge-LLM

Companion repository for Marco Haruni's technical book, **Build a Small LLM in PyTorch From Scratch: A Research-Grade Guide to Architecture, Data, Training, Optimization, and Debugging**.

forge-LLM is a clean PyTorch reference implementation for small decoder-only LLM pretraining research. It is meant to be readable, runnable, and easy to modify: swap model sizes, optimizers, datasets, and devices, then run a token-budgeted from-scratch experiment on CPU, Apple MPS, or CUDA.

It is not a notebook, not a trainer wrapper, not a distributed training framework, and not a benchmark claim. Any benchmark result you record with this repo is a local measurement on your own hardware and data.

## Architecture

The default model is an approximately 88.6M parameter decoder-only transformer:

- d_model=512, n_heads=8, n_layers=22, d_ff=2048
- grouped-query attention with n_kv_heads=4
- RoPE through `torchtune.modules.RotaryPositionalEmbeddings`
- QK-Norm with `torch.nn.RMSNorm`
- fused QKVO projection stored as one `nn.Parameter`
- `torch.nn.functional.scaled_dot_product_attention`
- Squared-ReLU feed-forward network
- tied token embedding and LM head weights
- token-budget training rather than epoch-based training

## Installation

```bash
pip install -r requirements.txt
```

The project targets Python 3.10+ and PyTorch 2.x.

## Quick Smoke Test

```bash
python tests/test_device_selection.py
python -c "import models, optimizers, training, data, configs, utils"
```

## Dataset Preparation

To build the 70/30 FineWeb-Edu/Cosmopedia mix:

```bash
python data/prepare_mix_data.py --target_tokens 22000000 --max_seq_len 2048
```

This writes `processed_data/pretrain_mix_{target_tokens}` with a `prep_metadata.json` file. The training CLI validates that the dataset sequence length matches the model config.

To download the Blueberry pretraining dataset:

```bash
python data/download_hf_data.py
```

## Training

CPU smoke run:

```bash
python train_llm.py --config 5m --train_tokens 50000 --batch_size 2 --device cpu --compile false
```

Common runs:

```bash
python train_llm.py --config default --device cuda
python train_llm.py --config 25m --dataset_path processed_data/pretrain_mix_22000000 --device mps --compile false
```

Config presets:

- `default`: about 88.6M parameters, short default token budget for smoke research runs
- `5m`: about 6.65M parameters
- `25m`: about 25.37M parameters
- `50m`: about 48.24M parameters
- `100m`: about 100.17M parameters

Expected parameter counts:

| Config | Parameters |
| --- | ---: |
| default | 88,630,528 |
| 5m | 6,652,800 |
| 25m | 25,366,272 |
| 50m | 48,244,224 |
| 100m | 100,169,472 |

## Optimizers

Muon is used for 2D trainable matrices except token embeddings and normalization weights. AdamW is used for embeddings, RMSNorm weights, and non-2D parameters. AdamW uses `fused=True` only on CUDA.

## Compile, Warmup, Reset

When `compile_model=True`, the trainer calls `torch.compile`, runs a few untimed warmup forward/backward steps to trigger kernel compilation, then restores the initial CPU weight snapshot before real training. The `--warmup` CLI flag only controls this untimed compile warmup path; it does not change LR warmup or the scheduler.

## Train A Tiny Model, Then Generate Text

After a small training run writes `checkpoints/model.pt`, generate text with:

```bash
python generation/generate.py --checkpoint checkpoints/model.pt --prompt "The future of language models" --max_new_tokens 50 --device cpu
```

Sampling supports greedy decoding, temperature, top-k, top-p, and optional repetition penalty.

## Device Support

`--device auto` selects CUDA when available, then Apple MPS, then CPU. CUDA training uses bfloat16 model weights and `torch.amp`; CPU and MPS keep default dtype.

## Tests

```bash
python tests/test_device_selection.py
```

Expected result: 4 tests pass.

## Notes On Reproducibility

The CLI seeds Python, NumPy, and PyTorch. Deterministic first-step loss assumes a deterministic backend and identical dependency versions, hardware, config, and data order.
