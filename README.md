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
uv sync
```

The project targets Python 3.10+ and PyTorch 2.x. `uv` reads `pyproject.toml` and manages the environment for you. `requirements.txt` is kept with the exact dependency list for compatibility with plain `pip` workflows.

## Quick Smoke Test

```bash
uv run python tests/test_device_selection.py
uv run python -c "import models, optimizers, training, data, configs, utils"
```

## Dataset Preparation

To build the 70/30 FineWeb-Edu/Cosmopedia mix:

```bash
uv run python data/prepare_mix_data.py --target_tokens 22000000 --max_seq_len 2048
```

This writes `processed_data/pretrain_mix_{target_tokens}` with a `prep_metadata.json` file. The training CLI validates that the dataset sequence length matches the model config.

To download the Blueberry pretraining dataset:

```bash
uv run python data/download_hf_data.py
```

## Training

CPU smoke run:

```bash
uv run python train_llm.py --config 5m --train_tokens 50000 --batch_size 2 --device cpu --compile false
```

Common runs:

```bash
uv run python train_llm.py --config default --device cuda
uv run python train_llm.py --config 25m --dataset_path processed_data/pretrain_mix_22000000 --device mps --compile false
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
uv run python generation/generate.py --checkpoint checkpoints/model.pt --prompt "The future of language models" --max_new_tokens 50 --temperature 0.7 --top_k 50 --top_p 0.9 --device cpu
```

Sampling supports greedy decoding, temperature, top-k, top-p, and optional repetition penalty.

## Device Support

`--device auto` selects CUDA when available, then Apple MPS, then CPU. CUDA training uses bfloat16 model weights and `torch.amp`; CPU and MPS keep default dtype.

## Training On A Cloud GPU

You can train forge-LLM on any rented NVIDIA GPU machine that gives you a normal Linux shell: RunPod, Lambda, Vast, Paperspace, CoreWeave, AWS, GCP, Azure, a university cluster, or a lab workstation. The provider does not matter much. What matters is that the instance has:

- an NVIDIA GPU visible to PyTorch through CUDA
- recent NVIDIA drivers
- enough VRAM for the config you choose
- persistent storage for datasets and checkpoints
- SSH, Jupyter, or a web terminal

Recommended starting points:

- `5m`: 8-12 GB VRAM is enough for small smoke runs
- `25m`: 16 GB VRAM is a good practical target
- `50m`: 24 GB VRAM is more comfortable
- `default` / `100m`: 24 GB minimum, 48 GB preferred

Older GPUs such as T4 can run small tests, but they do not have native bf16. forge-LLM falls back to fp16 on CUDA GPUs without bf16 support. For smoother longer training, prefer Ampere/Ada/Hopper GPUs such as A100, RTX 3090, RTX 4090, L4, L40S, or H100.

On any vendor, start from a PyTorch/CUDA image if one is offered. Avoid bare Ubuntu images unless you are comfortable installing NVIDIA drivers and CUDA tooling yourself.

### Generic SSH Workflow

On your local machine, generate an SSH key if you do not already have one:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub
```

Add the printed public key to your cloud GPU provider. Create a GPU instance, then copy the SSH command from the provider dashboard. It usually looks like one of these:

```bash
ssh root@PUBLIC_IP -p SSH_PORT -i ~/.ssh/id_ed25519
ssh ubuntu@PUBLIC_IP -i ~/.ssh/id_ed25519
ssh USERNAME@HOSTNAME -i ~/.ssh/id_ed25519
```

After connecting to the GPU machine:

```bash
nvidia-smi
git clone https://github.com/marcoharuni/forge-LLM.git
cd forge-LLM
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync
uv run python tests/test_device_selection.py
uv run python train_llm.py --help
```

If `nvidia-smi` does not show a GPU, fix the cloud image/driver/runtime before training.

Prepare data on the GPU machine, or upload a prepared dataset into `processed_data/`:

```bash
uv run python data/prepare_mix_data.py --target_tokens 22000000 --max_seq_len 2048
```

Start with a small CUDA run:

```bash
uv run python train_llm.py \
  --config 5m \
  --train_tokens 50000 \
  --batch_size 2 \
  --device cuda \
  --dataset_path processed_data/pretrain_mix_22000000
```

For longer SSH runs, use `tmux` so training continues if your SSH session drops:

```bash
tmux new -s forge
uv run python train_llm.py --config 25m --device cuda --dataset_path processed_data/pretrain_mix_22000000
```

Detach with `Ctrl-b` then `d`, and reconnect later with:

```bash
tmux attach -t forge
```

Keep important outputs on persistent storage. On rented GPUs, local container storage may disappear when the machine is deleted. Save or download:

- `checkpoints/`
- `plots/`
- `processed_data/`

When training is finished, stop or terminate the rented GPU so you are not billed for idle time.

### Provider Notes

- RunPod: use a Pod for training, preferably with a PyTorch template. RunPod documents Pod setup and SSH access in their [Pods overview](https://docs.runpod.io/pods/overview) and [SSH guide](https://docs.runpod.io/pods/configuration/use-ssh).
- Lambda, Paperspace, Vast, CoreWeave, AWS, GCP, Azure: choose a PyTorch/CUDA image, connect by SSH or Jupyter, verify `nvidia-smi`, clone the repo, then run the same commands above.
- Managed notebooks: use the Colab-style commands below, but remember that notebook runtimes are usually less persistent than a normal VM.

### Colab A100 Notebook Workflow

Google Colab can run this repo in a notebook. GPU type, runtime length, and availability vary over time; Colab's own FAQ says GPU types and limits are not guaranteed and may change. If you specifically want an A100, use a paid Colab option when available and select an A100 runtime from the Colab UI.

In Colab:

1. Open a new notebook.
2. Choose **Runtime > Change runtime type**.
3. Select **GPU**.
4. If the UI offers GPU class selection, choose **A100**.
5. Verify the GPU:

Colab shell commands start with `!`. In a normal Ubuntu terminal, run the same commands without `!`.

```python
!nvidia-smi
```

Clone the repo:

```python
!git clone https://github.com/marcoharuni/forge-LLM.git
```

Enter the repo:

```python
%cd forge-LLM
```

Install `uv`:

```python
!pip install uv
```

Install dependencies:

```python
!uv sync
```

Run the device tests:

```python
!uv run python tests/test_device_selection.py
```

Check the training CLI:

```python
!uv run python train_llm.py --help
```

If Hugging Face prints an unauthenticated-request warning, the command can still work. For higher rate limits, add an `HF_TOKEN` secret in Colab and run:

```python
from google.colab import userdata
import os

hf_token = userdata.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
```

Prepare a small dataset. This is a smoke dataset for testing the full loop on a T4 or A100, not a quality dataset:

```python
!uv run python data/prepare_mix_data.py --target_tokens 1000000 --max_seq_len 512
```

Train the 5m model on CUDA:

```python
!uv run python train_llm.py \
  --config 5m \
  --train_tokens 50000 \
  --batch_size 2 \
  --max_seq_len 512 \
  --device cuda \
  --compile false \
  --dataset_path processed_data/pretrain_mix_1000000
```

Generate text from the checkpoint:

```python
!uv run python generation/generate.py \
  --checkpoint checkpoints/model.pt \
  --prompt "The future of language models" \
  --max_new_tokens 50 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.9 \
  --device cuda
```

The generated text from a tiny smoke run may be messy. That is expected. The purpose of this path is to prove that data preparation, training, checkpointing, and generation all work. Better text requires more tokens and a longer run.

Colab VMs are temporary. Download your outputs before disconnecting, or copy them to Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then copy `checkpoints/`, `plots/`, and any prepared datasets you want to keep into Drive. For serious long-running pretraining, a normal cloud GPU VM with persistent storage is usually better than Colab.

## Tests

```bash
uv run python tests/test_device_selection.py
```

Expected result: 4 tests pass.

## Notes On Reproducibility

The CLI seeds Python, NumPy, and PyTorch. Deterministic first-step loss assumes a deterministic backend and identical dependency versions, hardware, config, and data order.
