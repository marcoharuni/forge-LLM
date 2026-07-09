# Book Blueprint

Working title:

```text
Build a Small LLM in PyTorch From Scratch
```

Subtitle:

```text
A Research-Grade Guide to Architecture, Data, Training, Optimization, and Debugging
```

This book should be clear like a beginner book, but technically sharper than a minimal GPT clone. The reader should finish with a working repo, a mental model of every major system, and enough debugging instinct to run small experiments without treating the model as magic.

## What Similar Beginner Books Do Well

Some short pretraining books are excellent at narrative flow. They usually win on:

- starting with the simple next-token objective
- introducing one concept at a time
- using small code snippets before full systems
- explaining compute and memory in plain language
- showing loss curves, checkpointing, and generation progress
- giving an appendix with debugging tips and hyperparameters

forge-LLM should keep those strengths.

## Where This Book Should Go Further

This repo is not a one-file GPT-2 clone. The book should emphasize the more modern research-grade choices:

- grouped-query attention
- RoPE through `torchtune`, not hand-rolled code
- QK-Norm
- RMSNorm
- Squared-ReLU feed-forward networks
- fused QKVO parameter slicing
- PyTorch `scaled_dot_product_attention`
- Muon plus AdamW optimizer routing
- token-budget training
- `torch.compile` warmup and weight reset
- CPU, MPS, CUDA, bf16, and fp16 fallback
- cloud GPU and Colab smoke paths

The promise is not "train a frontier model." The promise is "understand and run a real small-LLM pretraining stack."

## Contents

### Prologue: The First Token

- what pretraining is
- why next-token prediction is the core objective
- why small LLMs are worth building
- what the repo builds
- what tiny smoke runs prove
- what they do not prove

### Chapter 1: From Text to Tokens

- why models need token IDs
- vocabulary size and embedding cost
- sequence length and attention cost
- Hugging Face tokenizers
- EOS and padding behavior
- tokenizer warnings and long documents

### Chapter 2: Building the Data Pipeline

- streaming data from Hugging Face
- prepared local datasets
- FineWeb-Edu and Cosmopedia-style data
- tokenizing without truncation
- fixed-length chunks
- dropping partial tails
- `input_ids` and copied `labels`
- train/validation split
- `prep_metadata.json`
- why dataset sequence length must match model sequence length

### Chapter 3: The Decoder-Only Transformer

- decoder-only versus encoder-decoder
- causal language modeling
- logits over vocabulary
- why training and generation use the same model
- full architecture overview

### Chapter 4: Embeddings and Weight Tying

- token embedding table
- embedding scale by `sqrt(d_model)`
- LM head
- tying `lm_head.weight` to `token_embedding.weight`
- parameter-count impact

### Chapter 5: Attention and Causality

- Q, K, V
- causal masking
- scaled dot-product attention
- PyTorch SDPA
- how PyTorch may choose fused kernels
- why the repo does not depend on `flash-attn`
- why older T4-style GPUs are smoke-test hardware

### Chapter 6: Grouped-Query Attention

- multi-head attention recap
- key/value heads versus query heads
- `n_heads`, `n_kv_heads`, and key-value groups
- `repeat_interleave`
- memory and speed motivation

### Chapter 7: RoPE and QK-Norm

- why positions matter
- absolute embeddings versus RoPE
- using `torchtune.modules.RotaryPositionalEmbeddings`
- applying RoPE to Q and K
- RMSNorm on Q and K
- stability and scale

### Chapter 8: The Transformer Block

- pre-norm residual path
- RMSNorm
- attention branch
- Squared-ReLU feed-forward branch
- dropout
- residual updates
- stacking layers

### Chapter 9: Assembling `MinimalLLM`

- `LLMConfig`
- model presets
- initialization
- forward pass
- sequence length checks
- parameter counts
- shape checks

### Chapter 10: Training by Token Budget

- why tokens beat epochs for pretraining
- batch size
- gradient accumulation
- shift-labels-not-logits
- `ignore_index=-100`
- progress bar by tokens
- logging without excessive GPU sync

### Chapter 11: Evaluation and Generation

- validation loss
- next-token accuracy
- perplexity
- greedy decoding
- temperature
- top-k
- top-p
- repetition penalty
- why early generated text is broken

### Chapter 12: Optimizers: AdamW and Muon

- why optimizer routing matters
- Muon for 2D matrix parameters
- AdamW for embeddings, norms, and non-2D parameters
- weight decay
- LR schedules
- sanity checks for routed parameters

### Chapter 13: Devices, Precision, and Compile

- CPU
- Apple MPS
- CUDA
- bf16 on modern CUDA GPUs
- fp16 fallback on T4/V100-style GPUs
- `torch.amp`
- `torch.compile`
- compile warmup
- restoring weights after warmup

### Chapter 14: Running Experiments

- CPU smoke run
- Colab/T4 smoke run
- modern CUDA run
- cloud GPU setup
- checkpoint outputs
- plots
- metrics JSON
- what counts as a real experiment

### Chapter 15: Debugging the Run

- import errors
- tokenizer access errors
- dataset sequence mismatch
- loss does not move
- loss explodes
- NaNs
- CUDA OOM
- slow training
- poor generation

### Chapter 16: Scaling Carefully

- 5m, 25m, 50m, default, 100m
- data requirements
- token budgets
- sequence length tradeoffs
- batch size and gradient accumulation
- when more parameters help
- when better data helps more

### Chapter 17: Domain Small LLMs

- pharmacy, legal, education, code, and private corpora
- pretraining versus fine-tuning
- privacy and anonymization
- domain data cleaning
- expected quality
- evaluation before deployment

### Epilogue: What You Now Own

- the full path from text to generation
- the architecture
- the training loop
- debugging instincts
- next steps: instruction tuning, evaluation, quantization, and deployment

## Appendix

- config reference
- CLI reference
- parameter-count table
- device support table
- cloud GPU commands
- Colab commands
- common warnings
- troubleshooting checklist
- papers and repositories

