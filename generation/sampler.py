import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits, input_ids, penalty=None):
    if penalty is None or penalty == 1.0:
        return logits
    logits = logits.clone()
    for batch_idx in range(logits.size(0)):
        seen = torch.unique(input_ids[batch_idx])
        logits[batch_idx, seen] = torch.where(
            logits[batch_idx, seen] < 0,
            logits[batch_idx, seen] * penalty,
            logits[batch_idx, seen] / penalty,
        )
    return logits


def top_k_filter(logits, top_k=None):
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    cutoff = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < cutoff, float("-inf"))


def top_p_filter(logits, top_p=None):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    mask = cumulative > top_p
    mask[:, 1:] = mask[:, :-1].clone()
    mask[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(1, sorted_indices, sorted_logits)


def sample_next_token(
    logits,
    input_ids,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=None,
    greedy=False,
):
    logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
    if greedy or temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / max(temperature, 1e-8)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
