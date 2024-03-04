import torch.nn as nn

from typing import List


class BaseCNN(nn.Module):
    def __init__(
        self,
        layer_config: List[int],
        input_size: List[int] = [],
        channel_config: List[int] = [],
        stride_config: List[int] = [],
        dilation_config: List[int] = [],
        net_name: str = "",
    ):
        super(__class__, self).__init__()

        if not channel_config:
            assert len(layer_config) == len(channel_config), f"channel config must be the same as layer config"
        if not dilation_config:
            assert len(layer_config) == len(dilation_config), f"dilation config must be the same as layer config"
        if not stride_config:
            assert len(layer_config) == len(stride_config), f"stride config must be the same as layer config"

        self.name = net_name
