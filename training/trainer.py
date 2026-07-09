import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from configs import LLMConfig
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
    model.train()
    start_time = time.time()
    tokens_seen = 0
    steps = 0
    micro_steps = 0
    last_loss = None
    metrics_history = []
    eval_seen = set()
    progress = tqdm(total=config.train_tokens, unit="tok")

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

            loss.backward()
            micro_steps += 1
            batch_tokens = int(x.numel())
            tokens_seen += batch_tokens
            progress.update(batch_tokens)

            if micro_steps % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
                steps += 1

                if steps % 100 == 0 or last_loss is None:
                    last_loss = loss.detach().float().item() * config.gradient_accumulation_steps

                if steps % log_every == 0:
                    record = {
                        "step": steps,
                        "tokens_seen": tokens_seen,
                        "train_loss": last_loss,
                    }
                    metrics_history.append(record)
                    progress.set_postfix(record)

                if val_loader is not None and _should_evaluate(config, steps, eval_seen):
                    metrics = evaluate_model(model, val_loader, config)
                    metrics.update({"step": steps, "tokens_seen": tokens_seen})
                    metrics_history.append(metrics)
                    print(f"eval step={steps} tokens={tokens_seen:,} loss={metrics['val_loss']:.4f}")
                    if early_stopper is not None and early_stopper(metrics["val_loss"], steps):
                        tokens_seen = config.train_tokens
                        break

    progress.close()
    final_metrics = {}
    if val_loader is not None:
        final_metrics = evaluate_model(model, val_loader, config)
        final_metrics.update({"step": steps, "tokens_seen": tokens_seen})
        metrics_history.append(final_metrics)

    training_time = time.time() - start_time
    result = {
        "model": model,
        "final_metrics": final_metrics,
        "metrics_history": metrics_history,
        "training_time": training_time,
        "steps": steps,
        "tokens_seen": tokens_seen,
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        serializable = {
            "final_metrics": final_metrics,
            "metrics_history": metrics_history,
            "training_time": training_time,
            "steps": steps,
            "tokens_seen": tokens_seen,
            "extra_config": extra_config or {},
        }
        (output_path / "metrics.json").write_text(json.dumps(serializable, indent=2))
        torch.save(
            {
                "model_state_dict": _unwrap_model(model).state_dict(),
                "config": config.__dict__,
            },
            output_path / "model.pt",
        )

    return result


def warmup_compiled_kernels(model, config, train_loader, device, num_steps: int = 3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    amp_dtype = cuda_training_dtype() if device.type == "cuda" else torch.float32
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
    compare_baseline=False,
):
    set_seed(42)
    model = MinimalLLM(config)
    device = resolve_device(config.device)
    if device.type == "cuda":
        model.to(device, dtype=cuda_training_dtype())
    else:
        model.to(device)

    if load_weights_path:
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
