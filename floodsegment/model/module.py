import torch.nn as nn
from floodsegment.utils.misc import build_object

from typing import Any, Dict


class BaseModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscale_factor: int,
        activation: Dict[str, Any] | nn.Module,
        # normalization: Dict[str, Any] | nn.module,
    ):
        super(__class__, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downscale_factor = downscale_factor
        self.activation = activation if isinstance(activation, nn.Module) else build_object(**activation)
        # self.normalization = normalization if isinstance(normalization, nn.Module) else get_object(**normalization)

        self.block = None

    def forward(self, x):
        return self.block(x)


class SimpleCNN(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscale_factor: int,
        activation: str | nn.Module,
        # normalization: str | nn.module,
        kernel_size: int = 3,
        **kwargs,
    ):
        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale_factor=downscale_factor,
            activation=activation,
            # normalization=normalization,
        )

        padding = max(0, (kernel_size - 1) // 2)
        conv_kwargs = {
            "kernel_size": kernel_size,
            "padding": padding,
            "groups": kwargs.get("groups", 1),
            "bias": kwargs.get("bias", False),
            "dilation": kwargs.get("dilation", 1),
        }
        self.block = nn.ModuleList(
            [
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, **conv_kwargs),
                # self.normalization,
                self.activation,
            ]
        )
