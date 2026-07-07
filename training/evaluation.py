import math

import torch
import torch.nn.functional as F


def _batch_to_tensors(batch):
    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
    else:
        input_ids = batch[0]
        labels = batch[1] if len(batch) > 1 else input_ids
    return input_ids, labels


@torch.no_grad()
def evaluate_model(model, val_loader, config):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for step, batch in enumerate(val_loader):
        if step >= config.eval_steps:
            break
        input_ids, labels = _batch_to_tensors(batch)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        predictions = shift_logits.argmax(dim=-1)
        total_correct += (predictions == shift_labels).sum().item()
        token_count = shift_labels.numel()
        total_loss += loss.item()
        total_tokens += token_count

    if was_training:
        model.train()

    avg_loss = total_loss / max(1, total_tokens)
    accuracy = total_correct / max(1, total_tokens)
    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_perplexity": math.exp(min(avg_loss, 20)),
    }
