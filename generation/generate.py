import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=".*KernelPreference.*",
    category=FutureWarning,
)

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import LLMConfig
from generation.sampler import sample_next_token
from models import MinimalLLM
from training import resolve_device


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _config_from_checkpoint(checkpoint):
    cfg_dict = checkpoint.get("config", {})
    valid = LLMConfig.__dataclass_fields__.keys()
    kwargs = {key: value for key, value in cfg_dict.items() if key in valid}
    return LLMConfig(**kwargs)


def load_model(checkpoint_path, device_name="auto"):
    checkpoint = _load_checkpoint(checkpoint_path)
    config = _config_from_checkpoint(checkpoint) if isinstance(checkpoint, dict) else LLMConfig()
    config.device = device_name
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model = MinimalLLM(config)
    model.load_state_dict(state_dict, strict=False)
    device = resolve_device(device_name)
    model.to(device)
    model.eval()
    return model, config, device


@torch.no_grad()
def generate_text(
    checkpoint,
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    device="auto",
    tokenizer_name="HuggingFaceTB/SmolLM2-135M",
    repetition_penalty=None,
):
    model, config, resolved_device = load_model(checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(resolved_device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -config.max_seq_len :]
        logits = model(context)[:, -1, :]
        next_token = sample_next_token(
            logits,
            input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            greedy=temperature == 0,
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    generate_text(**vars(args))


if __name__ == "__main__":
    main()
