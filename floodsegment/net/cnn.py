import torch.nn as nn

from floodsegment.utils.builder import build_object
from floodsegment.net.base import _Buildable
from floodsegment.net.module import GenericBlock, SimpleConvLayer
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)


class GenericCnn(_Buildable):
    def __init__(
        self,
        *,
        # Block params
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        # dilation_config: List[int],
        base_config: Dict[str, Any],
        **kwargs,
    ):
        assert len(layer_config) == len(channel_config), f"channel config must be the same as layer config"
        assert len(layer_config) == len(stride_config), f"stride config must be the same as layer config"
        # assert len(layer_config) == len(dilation_config), f"dilation config must be the same as layer config"

        super(__class__, self).__init__(
            layer_config=layer_config,
            channel_config=channel_config,
            stride_config=stride_config,
            base_config=base_config,
            **kwargs,
        )

        self.layer_config = layer_config
        self.channel_config = channel_config
        self.stride_config = stride_config
        self.base_config = base_config

    def _build(
        self,
        *,
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        base_config: Dict[str, Any],
    ) -> nn.ModuleList:
        out_ch = None
        _block = nn.ModuleList()

        for i in range(len(layer_config)):
            in_ch = out_ch or channel_config[0]
            out_ch = channel_config[i]
            _kwargs = {
                "in_channels": in_ch,
                "out_channels": out_ch,
                "stride": stride_config[i],
                "n_layers": layer_config[i],
                # "dilation": dilation_config[i],
                "base_config": base_config,
            }

            _block.append(GenericBlock(**_kwargs))
        return _block


class GenericDecoder(GenericCnn):
    def __init__(
        self,
        *,
        # Block params
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        # dilation_config: List[int],
        base_config: Dict[str, Any],
        net_name: str = "GenericDecoder",
    ):
        super(__class__, self).__init__(
            layer_config=layer_config,
            channel_config=channel_config,
            stride_config=stride_config,
            base_config=base_config,
        )
        self.net_name = net_name


class GenericEncoder(GenericCnn):
    def __init__(
        self,
        *,
        # base layer params
        input_ch: int,
        init_kernel_size: int,
        init_stride: int,
        init_activation: Dict[str, Any],
        init_normalization: Dict[str, Any],
        # Block params
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        # dilation_config: List[int],
        base_config: Dict[str, Any],
        net_name: str = "GenericEncoder",
        **kwargs,
    ):
        assert len(layer_config) == len(channel_config), f"channel config must be the same as layer config"
        assert len(layer_config) == len(stride_config), f"stride config must be the same as layer config"
        # assert len(layer_config) == len(dilation_config), f"dilation config must be the same as layer config"

        super(__class__, self).__init__(
            input_ch=input_ch,
            init_kernel_size=init_kernel_size,
            init_stride=init_stride,
            init_activation=init_activation,
            init_normalization=init_normalization,
            layer_config=layer_config,
            channel_config=channel_config,
            stride_config=stride_config,
            base_config=base_config,
            **kwargs,
        )

        self.net_name = net_name
        self.input_ch = input_ch
        self.init_kernel_size = init_kernel_size
        self.init_stride = init_stride
        self.init_activation = build_object(**init_activation)
        self.init_normalization = build_object(**init_normalization, params={"num_features": channel_config[0]})
        self.layer_config = layer_config
        self.channel_config = channel_config
        self.stride_config = stride_config
        self.base_config = base_config

    def _build(
        self,
        *,
        input_ch: int,
        init_kernel_size: int,
        init_stride: int,
        init_activation: Dict[str, Any],
        init_normalization: Dict[str, Any],
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        base_config: Dict[str, Any],
        **kwargs,
    ) -> nn.ModuleList:
        out_ch = None
        _block = nn.ModuleList()

        # base layer
        _block.append(
            SimpleConvLayer(
                in_channels=input_ch,
                out_channels=channel_config[0],
                activation=init_activation,
                normalization=init_normalization,
                kernel_size=init_kernel_size,
                stride=init_stride,
                **kwargs,
            )
        )

        _block.extend(
            super()._build(
                layer_config=layer_config,
                channel_config=channel_config,
                stride_config=stride_config,
                base_config=base_config,
            )
        )
        return _block
