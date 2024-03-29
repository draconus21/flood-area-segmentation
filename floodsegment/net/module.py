import torch.nn as nn
from floodsegment.net.utils import compute_padding, build_normalization
from floodsegment.net.base import BaseModule, _Buildable
from floodsegment.utils.builder import build_object

from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)


class PatchDownsampling(_Buildable):
    def __init__(
        self, in_channels: int, out_channels: int, *, normalization: Dict[str, Any] = {"name": "torch.nn.BatchNorm2d"}
    ):
        super(__class__, self).__init__(in_channels=in_channels, out_channels=out_channels, normalization=normalization)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

    def _build(self, in_channels: int, out_channels: int, *, normalization: Dict[str, Any]) -> nn.ModuleList:
        return nn.ModuleList(
            [
                build_normalization(**normalization, overrides={"num_features": in_channels}),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=False),
            ]
        )


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
        assert kernel_size > 0, f"kernel_size must be greater than 1, got {kernel_size}"
        assert kernel_size % 2 == 1, f"kernel_Size must be odd, got {kernel_size}"
        assert dilation > 0, f"dilation must be greater than 1, got {dilation}"

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
        activation: Dict[str, Any],
        normalization: Dict[str, Any],
        kernel_size: int,
        dilation: int,
        **kwargs,
    ) -> nn.ModuleList:
        padding = compute_padding(kernel_size=kernel_size, dilation=dilation)
        conv_kwargs = {"kernel_size": kernel_size, "padding": padding, "dilation": dilation, **kwargs}
        return nn.ModuleList(
            [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **conv_kwargs),
                build_normalization(**normalization, overrides={"num_features": out_channels}),
                build_object(**activation),
            ]
        )


class ConvNeXt(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: Dict[str, Any] = {"name": "torch.nn.GELU"},
        normalization: Dict[str, Any] = {"name": "torch.nn.BatchNorm2d"},
        bn_factor: int = 4,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int = 1,
    ):
        assert stride in [1, 2], f"Only strides of 1 and 2 are supported, got {stride}"
        assert kernel_size > 0, f"kernel_size must be greater than 1, got {kernel_size}"
        assert kernel_size % 2 == 1, f"kernel_Size must be odd, got {kernel_size}"
        assert dilation > 0, f"dilation must be greater than 1, got {dilation}"

        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            normalization=normalization,
            bn_factor=bn_factor,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.bn_factor = bn_factor
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.skip_connection = (stride == 1) and (in_channels == out_channels)

    def _build(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: Dict[str, Any],
        normalization: Dict[str, Any],
        bn_factor: int,
        kernel_size: int,
        stride: int,
        dilation: int,
    ) -> nn.ModuleList:
        padding = compute_padding(kernel_size=kernel_size, dilation=dilation)

        _block = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    groups=in_channels,
                    dilation=dilation,
                ),
                build_normalization(**normalization, overrides={"num_features": in_channels}),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * bn_factor,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                build_object(**activation),
                nn.Conv2d(
                    in_channels=in_channels * bn_factor,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ]
        )

        # TODO: Verify this
        if in_channels != out_channels and stride == 1:
            _block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
        elif stride == 2:
            _block.append(
                PatchDownsampling(in_channels=in_channels, out_channels=out_channels, normalization=normalization)
            )
        return _block

    def forward(self, x):
        out = self.block(x)
        if self.skip_connection:
            out = out + x
        return out


class GenericBlock(_Buildable):
    """
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
