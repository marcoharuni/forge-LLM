from .device import DEVICE_CHOICES, cuda_training_dtype, describe_device, resolve_device
from .evaluation import evaluate_model
from .trainer import (
    EarlyStopping,
    setup_muon_optimizer,
    train_minimal_llm,
    train_model,
    warmup_compiled_kernels,
)

__all__ = [
    "DEVICE_CHOICES",
    "resolve_device",
    "describe_device",
    "cuda_training_dtype",
    "evaluate_model",
    "EarlyStopping",
    "setup_muon_optimizer",
    "train_model",
    "warmup_compiled_kernels",
    "train_minimal_llm",
]
