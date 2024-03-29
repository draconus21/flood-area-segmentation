import torch
from torch import nn
from floodsegment.utils.builder import build_object

from typing import Any, Dict


class BaseModel(nn.Module):
    def __init__(self, net: Dict[str, Any]):
        super(__class__, self).__init__()
        assert ".net." in net["name"], f"All cnns must be placed in net, got {net['name']}"
        self.net = build_object(**net)
        self.name = self.__class__.__name__

    def plot_step(self):
        raise NotImplementedError()

    @torch.no_grad
    def valid_step(self):
        raise NotImplementedError()

    def train_step(self):
        raise NotImplementedError()


class SampleModel(BaseModel):
    def __init__(self, net: Dict[str, Any]):
        super(__class__, self).__init__(net=net)
