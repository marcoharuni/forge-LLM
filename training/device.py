import torch


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{requested}'. Expected one of {DEVICE_CHOICES}.")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            memory_gb = props.total_memory / (1024**3)
            return f"CUDA device: {name} ({memory_gb:.1f} GB)"
        except Exception:
            return "CUDA device"
    if device.type == "mps":
        return "Apple Metal Performance Shaders (MPS)"
    return "CPU"


def cuda_training_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
