import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer

from configs import LLMConfig
from generation.sampler import sample_next_token
from models import MinimalLLM
from optimizers import Muon
from utils.helpers import count_parameters, format_time, set_seed
from utils.plot_loss import plot_loss
from .device import cuda_training_dtype, describe_device, resolve_device
from .evaluation import evaluate_model


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_step = 0
        self.bad_checks = 0

    def __call__(self, val_loss: float, step: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = step
            self.bad_checks = 0
            return False
        self.bad_checks += 1
        return self.bad_checks >= self.patience


def setup_muon_optimizer(model, config: LLMConfig):
    muon_params = []
    adamw_params = []
    muon_count = 0
    adamw_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        route_to_muon = (
            param.ndim == 2
            and "token_embedding" not in name
            and "norm" not in name
        )
        if route_to_muon:
            muon_params.append(param)
            muon_count += param.numel()
        else:
            adamw_params.append(param)
            adamw_count += param.numel()

    print(f"Muon parameters: {muon_count:,}")
    print(f"AdamW parameters: {adamw_count:,}")

    optimizers = []
    if muon_params:
        optimizers.append(
            Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum)
        )
    if adamw_params:
        fused = torch.cuda.is_available() and config.device in {"auto", "cuda"}
        optimizers.append(
            AdamW(
                adamw_params,
                lr=config.adamw_lr,
                weight_decay=config.weight_decay,
                fused=fused,
            )
        )
    return optimizers


def _batch_to_tensors(batch):
    if isinstance(batch, dict):
        x = batch["input_ids"]
        y = batch.get("labels", x)
    else:
        x = batch[0]
        y = batch[1] if len(batch) > 1 else x
    return x, y


def _should_evaluate(config: LLMConfig, steps: int, eval_seen: set[int]) -> bool:
    if config.eval_milestones is not None:
        due = [m for m in config.eval_milestones if steps >= m and m not in eval_seen]
        if due:
            eval_seen.update(due)
            return True
    if config.eval_every is not None and steps > 0 and steps % config.eval_every == 0:
        return True
    return False


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def _strip_compile_prefix(state_dict):
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def _build_grad_scaler(device, amp_dtype, use_amp: bool):
    enabled = bool(use_amp and device.type == "cuda" and amp_dtype == torch.float16)
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _current_lrs(optimizers):
    lrs = {}
    for optimizer_index, optimizer in enumerate(optimizers):
        for group_index, group in enumerate(optimizer.param_groups):
            lrs[f"lr_{optimizer_index}_{group_index}"] = group["lr"]
    return lrs


def _cuda_memory_gb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return str(value)


def _append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def _write_metrics_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    preferred = [
        "type",
        "step",
        "tokens_seen",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_perplexity",
        "grad_norm",
        "tokens_per_sec",
        "total_tokens_per_sec",
        "step_time_seconds",
        "gpu_memory_gb",
        "elapsed_seconds",
    ]
    fieldnames = [field for field in preferred if any(field in record for record in records)]
    extras = sorted({key for record in records for key in record} - set(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extras)
        writer.writeheader()
        writer.writerows(records)


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _best_val_record(records):
    val_records = [record for record in records if "val_loss" in record]
    if not val_records:
        return {}
    return min(val_records, key=lambda record: record["val_loss"])


def _average_metric(records, key):
    values = [record[key] for record in records if key in record and record[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _training_state_dict(
    model,
    config,
    optimizers,
    schedulers,
    scaler,
    steps,
    tokens_seen,
    micro_steps,
    metrics_history,
    eval_seen,
    training_time,
    extra_config,
):
    return {
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
        "scheduler_state_dicts": [scheduler.state_dict() for scheduler in schedulers],
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "steps": steps,
        "tokens_seen": tokens_seen,
        "micro_steps": micro_steps,
        "metrics_history": metrics_history,
        "eval_seen": sorted(eval_seen),
        "training_time": training_time,
        "config": config.__dict__,
        "extra_config": extra_config or {},
        "rng_state": _rng_state(),
        "best_val": _best_val_record(metrics_history),
    }


def _save_training_checkpoint(
    path,
    model,
    config,
    optimizers,
    schedulers,
    scaler,
    steps,
    tokens_seen,
    micro_steps,
    metrics_history,
    eval_seen,
    training_time,
    extra_config,
):
    torch.save(
        _training_state_dict(
            model=model,
            config=config,
            optimizers=optimizers,
            schedulers=schedulers,
            scaler=scaler,
            steps=steps,
            tokens_seen=tokens_seen,
            micro_steps=micro_steps,
            metrics_history=metrics_history,
            eval_seen=eval_seen,
            training_time=training_time,
            extra_config=extra_config,
        ),
        path,
    )


def _load_training_state(path):
    if path is None:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def _generate_report_sample(
    model,
    config,
    device,
    tokenizer_name,
    prompt,
    max_new_tokens,
):
    if not prompt or max_new_tokens <= 0:
        return ""

    was_training = model.training
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    for _ in range(max_new_tokens):
        context = input_ids[:, -config.max_seq_len :]
        logits = model(context)[:, -1, :]
        next_token = sample_next_token(
            logits,
            input_ids,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    if was_training:
        model.train()
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def _write_training_report(path, result, config, model, device_description, sample_text=None):
    records = result.get("metrics_history", [])
    best_val = _best_val_record(records)
    final_metrics = result.get("final_metrics", {})
    avg_tokens_per_sec = _average_metric(records, "tokens_per_sec")
    total_params = count_parameters(_unwrap_model(model))

    lines = [
        "# Training Report",
        "",
        "## Run",
        "",
        f"- device: {device_description}",
        f"- total_parameters: {total_params:,}",
        f"- train_tokens_target: {config.train_tokens:,}",
        f"- tokens_seen: {result.get('tokens_seen', 0):,}",
        f"- steps: {result.get('steps', 0):,}",
        f"- training_time: {format_time(result.get('training_time', 0))}",
    ]
    if avg_tokens_per_sec is not None:
        lines.append(f"- average_logged_tokens_per_sec: {avg_tokens_per_sec:.2f}")

    lines.extend(
        [
            "",
            "## Config",
            "",
            f"- d_model: {config.d_model}",
            f"- n_heads: {config.n_heads}",
            f"- n_kv_heads: {config.n_kv_heads}",
            f"- n_layers: {config.n_layers}",
            f"- d_ff: {config.d_ff}",
            f"- max_seq_len: {config.max_seq_len}",
            f"- vocab_size: {config.vocab_size}",
            f"- batch_size: {config.batch_size}",
            f"- gradient_accumulation_steps: {config.gradient_accumulation_steps}",
            f"- schedule_type: {config.schedule_type}",
        ]
    )

    lines.extend(["", "## Evaluation", ""])
    if best_val:
        lines.append(
            f"- best_val_loss: {best_val['val_loss']:.4f} "
            f"at step {best_val.get('step', 'unknown')}"
        )
    if "val_loss" in final_metrics:
        lines.append(f"- final_val_loss: {final_metrics['val_loss']:.4f}")
    if "val_perplexity" in final_metrics:
        lines.append(f"- final_val_perplexity: {final_metrics['val_perplexity']:.4f}")
    if "val_accuracy" in final_metrics:
        lines.append(f"- final_val_accuracy: {final_metrics['val_accuracy']:.4f}")
    if not best_val and not final_metrics:
        lines.append("- no validation loader was provided")

    lines.extend(["", "## Sample", ""])
    if sample_text:
        lines.extend(["```text", sample_text, "```"])
    else:
        lines.append("No sample was generated for this run.")

    path.write_text("\n".join(lines) + "\n")


def train_model(
    model,
    config,
    train_loader,
    val_loader,
    optimizers,
    schedulers=None,
    early_stopper=None,
    output_dir=None,
    extra_config=None,
    log_every=100,
    resume_state=None,
    report_tokenizer_name=None,
    report_prompt=None,
    report_max_new_tokens=50,
):
    device = resolve_device(config.device)
    if device.type == "cuda":
        amp_dtype = cuda_training_dtype()
        model.to(device, dtype=amp_dtype)
    else:
        amp_dtype = torch.float32
        model.to(device)
    print(describe_device(device))

    schedulers = schedulers or []
    scaler = _build_grad_scaler(device, amp_dtype, config.use_amp)
    if resume_state and resume_state.get("scaler_state_dict") and scaler.is_enabled():
        scaler.load_state_dict(resume_state["scaler_state_dict"])
    if scaler.is_enabled():
        print("AMP: fp16 autocast with GradScaler enabled.")
    elif config.use_amp and device.type == "cuda":
        print(f"AMP: {amp_dtype} autocast without GradScaler.")

    output_path = Path(output_dir) if output_dir else None
    metrics_jsonl_path = None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path = output_path / "metrics.jsonl"
        if not resume_state:
            metrics_jsonl_path.write_text("")

    model.train()
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = int(resume_state.get("tokens_seen", 0)) if resume_state else 0
    last_log_step = int(resume_state.get("steps", 0)) if resume_state else 0
    tokens_seen = int(resume_state.get("tokens_seen", 0)) if resume_state else 0
    steps = int(resume_state.get("steps", 0)) if resume_state else 0
    micro_steps = int(resume_state.get("micro_steps", 0)) if resume_state else 0
    last_loss = None
    metrics_history = list(resume_state.get("metrics_history", [])) if resume_state else []
    eval_seen = set(resume_state.get("eval_seen", [])) if resume_state else set()
    previous_training_time = float(resume_state.get("training_time", 0.0)) if resume_state else 0.0
    progress = tqdm(total=config.train_tokens, initial=min(tokens_seen, config.train_tokens), unit="tok")
    if resume_state:
        print(f"Resuming from step={steps:,}, tokens_seen={tokens_seen:,}.")

    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens:
                break
            x, y = _batch_to_tensors(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            autocast_enabled = config.use_amp and device.type == "cuda"
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / config.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_steps += 1
            batch_tokens = int(x.numel())
            tokens_seen += batch_tokens
            progress.update(batch_tokens)

            if micro_steps % config.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
                if scaler.is_enabled():
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                    scaler.update()
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                else:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
                steps += 1

                should_log = steps % log_every == 0
                if steps % 100 == 0 or last_loss is None or should_log:
                    last_loss = loss.detach().float().item() * config.gradient_accumulation_steps

                if should_log:
                    now = time.time()
                    elapsed_since_log = max(1e-9, now - last_log_time)
                    elapsed_total = previous_training_time + max(1e-9, now - start_time)
                    log_step_delta = max(1, steps - last_log_step)
                    record = {
                        "type": "train",
                        "step": steps,
                        "tokens_seen": tokens_seen,
                        "train_loss": last_loss,
                        "grad_norm": float(total_grad_norm.detach().cpu()),
                        "tokens_per_sec": (tokens_seen - last_log_tokens) / elapsed_since_log,
                        "total_tokens_per_sec": tokens_seen / elapsed_total,
                        "step_time_seconds": elapsed_since_log / log_step_delta,
                        "gpu_memory_gb": _cuda_memory_gb(device),
                        "elapsed_seconds": elapsed_total,
                    }
                    record.update(_current_lrs(optimizers))
                    metrics_history.append(record)
                    _append_jsonl(metrics_jsonl_path, record)
                    if output_path is not None:
                        _save_training_checkpoint(
                            output_path / "training_state.pt",
                            model,
                            config,
                            optimizers,
                            schedulers,
                            scaler,
                            steps,
                            tokens_seen,
                            micro_steps,
                            metrics_history,
                            eval_seen,
                            elapsed_total,
                            extra_config,
                        )
                    progress.set_postfix(
                        {
                            "loss": f"{last_loss:.4f}",
                            "tok/s": f"{record['tokens_per_sec']:.0f}",
                            "grad": f"{record['grad_norm']:.2f}",
                        }
                    )
                    last_log_time = now
                    last_log_tokens = tokens_seen
                    last_log_step = steps

                if val_loader is not None and _should_evaluate(config, steps, eval_seen):
                    metrics = evaluate_model(model, val_loader, config)
                    metrics.update(
                        {
                            "type": "eval",
                            "step": steps,
                            "tokens_seen": tokens_seen,
                            "elapsed_seconds": previous_training_time + (time.time() - start_time),
                        }
                    )
                    metrics.update(_current_lrs(optimizers))
                    metrics_history.append(metrics)
                    _append_jsonl(metrics_jsonl_path, metrics)
                    if output_path is not None:
                        _save_training_checkpoint(
                            output_path / "training_state.pt",
                            model,
                            config,
                            optimizers,
                            schedulers,
                            scaler,
                            steps,
                            tokens_seen,
                            micro_steps,
                            metrics_history,
                            eval_seen,
                            metrics["elapsed_seconds"],
                            extra_config,
                        )
                    print(f"eval step={steps} tokens={tokens_seen:,} loss={metrics['val_loss']:.4f}")
                    if early_stopper is not None and early_stopper(metrics["val_loss"], steps):
                        tokens_seen = config.train_tokens
                        break

    progress.close()
    final_metrics = {}
    if val_loader is not None:
        final_metrics = evaluate_model(model, val_loader, config)
        final_metrics.update(
            {
                "type": "final_eval",
                "step": steps,
                "tokens_seen": tokens_seen,
                "elapsed_seconds": previous_training_time + (time.time() - start_time),
            }
        )
        final_metrics.update(_current_lrs(optimizers))
        metrics_history.append(final_metrics)
        _append_jsonl(metrics_jsonl_path, final_metrics)

    training_time = previous_training_time + (time.time() - start_time)
    result = {
        "model": model,
        "final_metrics": final_metrics,
        "metrics_history": metrics_history,
        "training_time": training_time,
        "steps": steps,
        "tokens_seen": tokens_seen,
    }

    if output_path is not None:
        serializable = {
            "final_metrics": final_metrics,
            "metrics_history": metrics_history,
            "training_time": training_time,
            "steps": steps,
            "tokens_seen": tokens_seen,
            "extra_config": extra_config or {},
        }
        (output_path / "metrics.json").write_text(json.dumps(serializable, indent=2, default=_json_default))
        _write_metrics_csv(output_path / "metrics.csv", metrics_history)
        _save_training_checkpoint(
            output_path / "training_state.pt",
            model,
            config,
            optimizers,
            schedulers,
            scaler,
            steps,
            tokens_seen,
            micro_steps,
            metrics_history,
            eval_seen,
            training_time,
            extra_config,
        )
        torch.save(
            {
                "model_state_dict": _unwrap_model(model).state_dict(),
                "config": config.__dict__,
            },
            output_path / "model.pt",
        )
        sample_text = None
        if report_prompt and report_tokenizer_name:
            try:
                sample_text = _generate_report_sample(
                    model=model,
                    config=config,
                    device=device,
                    tokenizer_name=report_tokenizer_name,
                    prompt=report_prompt,
                    max_new_tokens=report_max_new_tokens,
                )
            except Exception as exc:
                sample_text = f"Sample generation failed: {exc}"
        _write_training_report(
            output_path / "training_report.md",
            result,
            config,
            model,
            describe_device(device),
            sample_text=sample_text,
        )

    return result


def warmup_compiled_kernels(model, config, train_loader, device, num_steps: int = 3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    amp_dtype = cuda_training_dtype() if device.type == "cuda" else torch.float32
    scaler = _build_grad_scaler(device, amp_dtype, config.use_amp)
    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break
        x, y = _batch_to_tensors(batch)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = config.use_amp and device.type == "cuda"
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
            logits = model(x)
            shift_labels = torch.full_like(y, -100)
            shift_labels[:, :-1] = y[:, 1:]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


def _build_schedulers(optimizers, config):
    total_steps = max(
        1,
        config.train_tokens
        // (config.batch_size * config.max_seq_len * config.gradient_accumulation_steps),
    )
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        if config.schedule_type == "cosine":
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        if config.schedule_type == "linear":
            return max(0.1, 1.0 - 0.9 * progress)
        if config.schedule_type == "constant":
            return 1.0
        raise ValueError("schedule_type must be one of {'cosine', 'linear', 'constant'}")

    return [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]


def train_minimal_llm(
    config,
    train_loader,
    val_loader,
    output_dir=None,
    load_weights_path=None,
    resume_checkpoint=None,
    compare_baseline=False,
    report_tokenizer_name=None,
    report_prompt=None,
    report_max_new_tokens=50,
):
    set_seed(42)
    resume_state = _load_training_state(resume_checkpoint)
    model = MinimalLLM(config)
    device = resolve_device(config.device)
    if device.type == "cuda":
        model.to(device, dtype=cuda_training_dtype())
    else:
        model.to(device)

    if resume_state is not None:
        state_dict = _strip_compile_prefix(resume_state["model_state_dict"])
        model.load_state_dict(state_dict, strict=False)
        _restore_rng_state(resume_state.get("rng_state"))
    elif load_weights_path:
        try:
            checkpoint = torch.load(load_weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(load_weights_path, map_location="cpu")
        state_dict = _strip_compile_prefix(checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)

    snapshot = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    print(f"Total parameters: {count_parameters(model):,}")

    if config.compile_model:
        try:
            base_model = model
            model = torch.compile(base_model)
            if getattr(config, "compile_warmup", True):
                warmup_compiled_kernels(model, config, train_loader, device, 3)
                base_model.load_state_dict(snapshot, strict=True)
                print("torch.compile warmup complete; weights restored.")
            else:
                print("torch.compile enabled; untimed kernel warmup skipped.")
        except Exception as exc:
            print(f"torch.compile failed; falling back to eager mode: {exc}")
            model = MinimalLLM(config)
            if device.type == "cuda":
                model.to(device, dtype=cuda_training_dtype())
            else:
                model.to(device)
            model.load_state_dict(snapshot, strict=True)

    optimizers = setup_muon_optimizer(model, config)
    schedulers = _build_schedulers(optimizers, config)
    if resume_state is not None:
        for optimizer, state_dict in zip(optimizers, resume_state.get("optimizer_state_dicts", [])):
            optimizer.load_state_dict(state_dict)
        for scheduler, state_dict in zip(schedulers, resume_state.get("scheduler_state_dicts", [])):
            scheduler.load_state_dict(state_dict)
        _restore_rng_state(resume_state.get("rng_state"))
    else:
        set_seed(42)
    result = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        output_dir=output_dir,
        extra_config={"compare_baseline": compare_baseline},
        log_every=config.log_every,
        resume_state=resume_state,
        report_tokenizer_name=report_tokenizer_name,
        report_prompt=report_prompt,
        report_max_new_tokens=report_max_new_tokens,
    )

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_file = plots_dir / f"metrics_{timestamp}.json"
    plot_file = plots_dir / f"val_loss_{timestamp}.png"
    metrics_file.write_text(json.dumps(result["metrics_history"], indent=2))
    baseline_file = Path("plots/baseline.json") if compare_baseline else None
    plot_loss(
        str(metrics_file),
        str(plot_file),
        "Validation Loss",
        str(baseline_file) if baseline_file and baseline_file.exists() else None,
    )
    print(
        f"Training finished in {format_time(result['training_time'])}; "
        f"tokens_seen={result['tokens_seen']:,}, steps={result['steps']:,}"
    )
    return result
