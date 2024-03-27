import torch.nn as nn
from floodsegment.utils.builder import build_object

from typing import Any, Dict


class _Buildable(nn.Module):
    def __init__(self, **kwargs):
        super(__class__, self).__init__()
        self.block = self._build(**kwargs)

    def _build(self, **kwargs) -> nn.ModuleList:
        raise NotImplementedError(f"Please implement this fucntion for your class ({self.__class__.__name__})")

    def forward(self, x):
        return self.block(x)

    def __len__(self):
        return len(self.block)

    def __getitem__(self, i):
        return self.block[i]


class BaseModule(_Buildable):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: Dict[str, Any],
        normalization: Dict[str, Any],
        **kwargs,
    ):
        _activation = build_object(**activation)
        _normalization = build_object(**normalization, params={"num_features": in_channels})
        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=_activation,
            normalization=_normalization,
            **kwargs,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = _activation
        self.normalization = _normalization
