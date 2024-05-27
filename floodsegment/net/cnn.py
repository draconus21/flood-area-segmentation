import torch
import torch.nn as nn
from torch import randn as trandn

from floodsegment.utils.builder import build_object
from floodsegment.net.base import _Buildable
from floodsegment.net.module import GenericBlock, SimpleConvLayer
from typing import Any, OrderedDict, Dict, List, Tuple

import logging

logger = logging.getLogger(__name__)


class GenericCnn(_Buildable):
    def __init__(
        self,
        *,
        input_ch: int,  # number of input channels
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
            input_ch=input_ch,
            layer_config=layer_config,
            channel_config=channel_config,
            stride_config=stride_config,
            base_config=base_config,
            **kwargs,
        )

        self.input_ch = input_ch
        self.layer_config = layer_config
        self.channel_config = channel_config
        self.stride_config = stride_config
        self.base_config = base_config

    def get_out_sizes(self, input_size: Tuple[int, int, int]) -> OrderedDict[str, torch.Size]:
        assert len(input_size) == 3, f"input_size must be [ch, h, w], got {input_size}"

        if not hasattr(self, "block") or not self.block:
            raise ValueError(
                f"self.block is not populated. Ensure that self._build() has been implemented for {self.__class__.__name__}."
            )

        dummy_inp = trandn(1, *input_size)
        outs: torch.Tensor | OrderedDict[str, torch.Tensor] = self.forward(dummy_inp, keep_interim_outs=True)

        outs = outs if isinstance(outs, OrderedDict) else OrderedDict({self.out_key_at(0): outs})

        return OrderedDict({k: out.shape[1:] for k, out in outs.items()})

    def _build(
        self,
        *,
        input_ch: int,
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        base_config: Dict[str, Any],
    ) -> nn.ModuleList:
        out_ch = None
        _block = nn.ModuleList()

        for i in range(len(layer_config)):
            in_ch = input_ch if i == 0 else channel_config[i - 1]
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
            logger.debug(f"{i}: added {base_config['name']}: {in_ch}->{out_ch}")
        return _block


class GenericDecoder(GenericCnn):
    def __init__(
        self,
        *,
        input_ch: int,
        # last layer params
        output_ch: int,
        out_kernel_size: int,
        out_stride: int,
        out_activation: Dict[str, Any],
        out_normalization: Dict[str, Any],
        # Block params
        size_config: List[int],
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        # dilation_config: List[int],
        base_config: Dict[str, Any],
        upsample_config: Dict[str, Any],
        net_name: str = "GenericDecoder",
        **kwargs,
    ):
        super(__class__, self).__init__(
            input_ch=input_ch,
            output_ch=output_ch,
            out_kernel_size=out_kernel_size,
            out_stride=out_stride,
            out_activation=out_activation,
            out_normalization=out_normalization,
            size_config=size_config,
            layer_config=layer_config,
            channel_config=channel_config,
            stride_config=stride_config,
            upsample_config=upsample_config,
            base_config=base_config,
            **kwargs,
        )
        self.net_name = net_name
        self.upsample_config = upsample_config
        self.output_ch = output_ch
        self.out_kernel_size = out_kernel_size
        self.out_stride = out_stride
        self.out_activation = out_activation
        self.out_normalization = out_normalization

    def _build(
        self,
        *,
        input_ch: int,
        # last layer params
        output_ch: int,
        out_kernel_size: int,
        out_stride: int,
        out_activation: Dict[str, Any],
        out_normalization: Dict[str, Any],
        # Block params
        size_config: List[int],
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        # dilation_config: List[int],
        upsample_config: Dict[str, Any],
        base_config: Dict[str, Any],
        **kwargs,
    ) -> nn.ModuleList:
        conv_blocks = super()._build(
            input_ch=input_ch,
            layer_config=layer_config,
            channel_config=channel_config,
            stride_config=stride_config,
            base_config=base_config,
        )

        _block = nn.ModuleList()
        up_channel_config = [input_ch, *channel_config[:-1]]
        for i, c_block in enumerate(conv_blocks):
            up_module: nn.Sequential = build_object(
                **upsample_config,
                overrides={
                    "in_channels": up_channel_config[i],
                    "out_channels": up_channel_config[i],
                    "size": size_config[i],
                },
            )
            _block.append(up_module)
            _block.append(c_block)
            logger.debug(f"added {up_module}_{i} for size {size_config[0]}")

        # last layer
        _block.append(
            SimpleConvLayer(
                in_channels=channel_config[-1],
                out_channels=output_ch,
                activation=out_activation,
                normalization=out_normalization,
                kernel_size=out_kernel_size,
                stride=out_stride,
                **kwargs,
            )
        )
        logger.debug(f"added last layer with {output_ch} output channels")
        return _block


class GenericEncoder(GenericCnn):
    def __init__(
        self,
        *,
        # base layer params
        input_size: Tuple[int, int, int],  # [channel, height, width]
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
        assert len(input_size) == 3, f"input_size must be [ch, h, w], got {input_size}"
        assert len(layer_config) == len(channel_config), f"channel config must be the same as layer config"
        assert len(layer_config) == len(stride_config), f"stride config must be the same as layer config"
        # assert len(layer_config) == len(dilation_config), f"dilation config must be the same as layer config"

        super(__class__, self).__init__(
            input_ch=input_size[0],  # needed for GenericCnn
            input_size=input_size,
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
        self.input_size = input_size
        self.init_kernel_size = init_kernel_size
        self.init_stride = init_stride
        self.init_activation = init_activation
        self.init_normalization = init_normalization

    def _build(
        self,
        *,
        input_ch: int,  # needed for GenericCnn
        # base layer params
        input_size: Tuple[int, int, int],  # [channel, height, width]
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
        **kwargs,
    ) -> nn.ModuleList:
        out_ch = None
        _block = nn.ModuleList()

        # base layer
        _block.append(
            SimpleConvLayer(
                in_channels=input_size[0],
                out_channels=channel_config[0],
                activation=init_activation,
                normalization=init_normalization,
                kernel_size=init_kernel_size,
                stride=init_stride,
                **kwargs,
            )
        )
        logger.debug(f"added base layer {input_size[1:]}: {input_size[0]}->{channel_config[0]} ")

        _block.extend(
            super()._build(
                input_ch=channel_config[0],
                layer_config=layer_config,
                channel_config=channel_config,
                stride_config=stride_config,
                base_config=base_config,
            )
        )
        return _block
