import torch.nn as nn
from floodsegment.net.utils import compute_padding
from floodsegment.utils.builder import build_object

from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)


class BaseModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Dict[str, Any] | nn.Module,
        normalization: Dict[str, Any] | nn.Module,
    ):
        super(__class__, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation if isinstance(activation, nn.Module) else build_object(**activation)
        self.normalization = (
            normalization
            if isinstance(normalization, nn.Module)
            else build_object(**normalization, params={"num_features": in_channels})
        )

        self.block = nn.ModuleList()

    def forward(self, x):
        return self.block(x)


class SimpleConvLayer(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Dict[str, Any] | nn.Module,
        normalization: Dict[str, Any] | nn.Module,
        kernel_size: int,
        dilation: int = 1,
        **kwargs,
    ):
        assert kernel_size > 0, f"kernel_size must be greater than 1"
        assert dilation > 0, f"dilation must be greater than 1"

        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            normalization=normalization,
        )

        padding = compute_padding(kernel_size=kernel_size, dilation=dilation)
        conv_kwargs = {"kernel_size": kernel_size, "padding": padding, "dilation": dilation, **kwargs}
        self.block = nn.ModuleList(
            [
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, **conv_kwargs),
                self.normalization,
                self.activation,
            ]
        )


class GenericBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_layers: int,
        base_config: Dict[str, Any],
    ):
        super(__class__, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.n_layers = n_layers
        self.base_config = base_config

        self.block = nn.ModuleList()
        self._build()  # populate self.block

    def _build(self) -> nn.ModuleList:
        out_ch = None
        _in_ch = [self.in_channels] * self.n_layers
        _out_ch = _in_ch[:]  # copy
        _stride = [1] * self.n_layers

        # update last layer config
        _out_ch[-1] = self.out_channels
        _stride[-1] = self.stride

        for i in range(self.n_layers):
            overrides = {
                "in_channels": _in_ch[i],
                "out_channels": _out_ch[i],
                "stride": _stride[i],
            }

            self.block.append(build_object(**self.base_config, overrides=overrides))
