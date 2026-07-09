# Debugging and Expectations

This guide is for the first hour of running forge-LLM. It explains what is normal, what is a problem, and what the first generated text means.

## Normal Warnings

### Hugging Face Unauthenticated Requests

Message:

```text
You are sending unauthenticated requests to the HF Hub.
```

This is normal. Downloads can still work. Add `HF_TOKEN` for higher rate limits.

### PyTorch / Torchtune KernelPreference Warning

Message:

```text
KernelPreference is an Enum subclass...
```

This comes from PyTorch/torchtune internals. It is not a forge-LLM failure.

### Tokenizer Long Sequence Warning

Long documents can be longer than the tokenizer model's nominal sequence length. For data preparation, forge-LLM tokenizes without truncation and then chunks the stream into fixed training blocks. That is expected.

## First Smoke Run Expectations

A T4 or small Colab run is useful for proving the full loop:

```text
data -> tokens -> train -> checkpoint -> generate
```

It is not expected to produce beautiful text. A tiny run may output:

- repeated words
- strange punctuation
- broken phrases
- mixed topics
- fragments that look almost grammatical

That is still a successful smoke test if the model trained, saved, loaded, and generated tokens.

## Signs The Pipeline Works

- `tests/test_device_selection.py` passes.
- `train_llm.py --help` shows config choices and device choices.
- dataset preparation writes `processed_data/pretrain_mix_*`.
- training creates `checkpoints/model.pt`.
- generation prints text beginning with the prompt.
- `plots/metrics_*.json` and `plots/val_loss_*.png` are created after a run with validation.

## Loss Expectations

At initialization, loss should be close to:

```text
log(vocab_size)
```

For a vocabulary around 49k, that is roughly 10.8. A short smoke run should usually lower the loss, but do not expect a tiny model trained for a tiny token budget to become coherent.

## Common Problems

### `ModuleNotFoundError`

Run commands from the repo root:

```bash
cd forge-LLM
uv run python train_llm.py --help
```

For Colab, `%cd forge-LLM` before running repo commands.

### Dataset Sequence Mismatch

If data was prepared with `--max_seq_len 512`, train with:

```bash
--max_seq_len 512
```

If data was prepared with `--max_seq_len 2048`, use the config default or pass:

```bash
--max_seq_len 2048
```

### CUDA Out Of Memory

Try:

- lower `--batch_size`
- lower `--max_seq_len`
- increase `--gradient_accumulation_steps`
- use the `5m` config first
- use a GPU with more VRAM

### Poor Generation

First try calmer sampling:

```bash
uv run python generation/generate.py \
  --checkpoint checkpoints/model.pt \
  --prompt "The future of language models" \
  --max_new_tokens 80 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.9 \
  --device cuda
```

If text is still poor, train longer or use more data. Sampling can reduce chaos, but it cannot create knowledge the model did not learn.

## T4 Versus Modern CUDA GPUs

T4 is a smoke-test target. It is useful for checking that the code runs on accessible cloud hardware.

For serious training, prefer modern CUDA GPUs with better bf16 and attention-kernel support:

- A100
- H100
- RTX 3090
- RTX 4090
- L4
- L40S

The code uses PyTorch `scaled_dot_product_attention`. On supported GPUs and shapes, PyTorch can use fused attention kernels. On older GPUs, it may fall back to a different backend. Correctness still matters more than the exact backend for beginner smoke tests.

## What Counts As Success

For a smoke test:

```text
The model trains without crashing and generates any text from a checkpoint.
```

For a real experiment:

```text
Validation loss decreases over a meaningful token budget, generation improves over checkpoints, and the setup is reproducible.
```

