import torch.nn as nn
from floodsegment.utils.builder import build_object

from typing import Dict, Any


def compute_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2


def build_normalization(name: str, params: Dict[str, Any] = {}, overrides: Dict[str, Any] = {}) -> nn.Module:
    if "torch.nn.BatchNorm" in name:
        assert "num_features" in params or "num_features" in overrides

    _normalization = build_object(name=name, params=params, overrides=overrides)
    return _normalization
