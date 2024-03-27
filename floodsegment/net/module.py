import torch.nn as nn
from floodsegment.net.utils import compute_padding
from floodsegment.utils.builder import build_object

from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)


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


class SimpleConvLayer(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: Dict[str, Any],
        normalization: Dict[str, Any],
        kernel_size: int,
        dilation: int = 1,
        **kwargs,
    ):
        assert kernel_size > 0, f"kernel_size must be greater than 1"
        assert kernel_size % 2 == 1, f"kernel_Size must be odd"
        assert dilation > 0, f"dilation must be greater than 1"

        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            normalization=normalization,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs,
        )
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.kwargs = kwargs

    def _build(
        self,
        in_channels: int,
        out_channels: int,
        normalization: nn.Module,
        activation: nn.Module,
        kernel_size: int,
        dilation: int,
        **kwargs,
    ) -> nn.ModuleList:
        padding = compute_padding(kernel_size=kernel_size, dilation=dilation)
        conv_kwargs = {"kernel_size": kernel_size, "padding": padding, "dilation": dilation, **kwargs}
        return nn.ModuleList(
            [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **conv_kwargs),
                normalization,
                activation,
            ]
        )


class GenericBlock(_Buildable):
    """
    !! DO NOT USE !!
    This class creates a block by stacking modules together.
    The module to be stacked is specified via base_config

    in/out_channels: number of input/output channels (see Note below)
    stride: stride for last block, all other block will have stride 1
    n_layers: number of times to stack
    base_config: config to build repeating block

    Note: All `n_layer` layers will use `in_channels` input channels.
    The first `n_layer`-1 layers will use `in_channels` output channels.
    Only the `n_layer`th layer will have out channel `out_channels`

    """

    def __init__(
        self, *, in_channels: int, out_channels: int, stride: int, n_layers: int, base_config: Dict[str, Any], **kwargs
    ):
        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            n_layers=n_layers,
            base_config=base_config,
            **kwargs,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.n_layers = n_layers
        self.base_config = base_config

    def _build(
        self, *, in_channels: int, out_channels: int, stride: int, n_layers: int, base_config: Dict[str, Any]
    ) -> nn.ModuleList:
        _in_ch = [in_channels] * n_layers
        _out_ch = _in_ch[:]  # copy
        _stride = [1] * n_layers

        # update last layer config
        _out_ch[-1] = out_channels
        _stride[-1] = stride

        _block = nn.ModuleList()
        for i in range(n_layers):
            overrides = {
                "in_channels": _in_ch[i],
                "out_channels": _out_ch[i],
                "stride": _stride[i],
            }

            _block.append(build_object(**base_config, overrides=overrides))

        return _block
