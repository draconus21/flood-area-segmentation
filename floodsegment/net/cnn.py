import torch.nn as nn

from floodsegment.utils.builder import build_object
from floodsegment.net.module import GenericBlock
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)


class BaseCnn(nn.Module):
    def __init__(
        self,
        input_ch: int,
        encoder: Dict[str, Any],
        # decoder: Dict[str, Any],
        net_name: str = "",
    ):
        super(__class__, self).__init__()

        self.net_name = net_name
        self.input_ch = input_ch
        self.encoder = build_object(**encoder, overrides={"input_ch": self.input_ch})
        # self.decoder = build_object(**decoder)


class GenericCnn(nn.Module):
    def __init__(
        self,
        input_ch: int,
        layer_config: List[int],
        channel_config: List[int],
        stride_config: List[int],
        # dilation_config: List[int],
        base_config: Dict[str, Any],
        net_name: str,
    ):
        if not channel_config:
            assert len(layer_config) == len(channel_config), f"channel config must be the same as layer config"
        if not stride_config:
            assert len(layer_config) == len(stride_config), f"stride config must be the same as layer config"
        # if not dilation_config:
        #    assert len(layer_config) == len(dilation_config), f"dilation config must be the same as layer config"

        super(__class__, self).__init__()

        self.net_name = net_name
        self.input_ch = input_ch
        self.layer_config = layer_config
        self.channel_config = channel_config
        self.stride_config = stride_config
        self.base_config: nn.Module = base_config

        self.block = nn.ModuleList()
        self._build()  # populate self.block

    def _build(self) -> nn.ModuleList:
        out_ch = None
        for i in range(len(self.layer_config)):
            in_ch = out_ch or self.channel_config[0]
            out_ch = self.channel_config[i]
            kwargs = {
                "in_channels": in_ch,
                "out_channels": out_ch,
                "stride": self.stride_config[i],
                "n_layers": self.layer_config[i],
                # "dilation": self.dilation_config[i],
                "base_config": self.base_config,
            }

            self.block.append(GenericBlock(**kwargs))
