import torch
from torch import nn
from floodsegment.utils.builder import build_object

from typing import Any, Dict


class BaseModel(nn.Module):
    def __init__(self, cnn: Dict[str, Any]):
        super(__class__, self).__init__()
        assert ".cnn." in cnn["name"], f"All cnns must be placed in cnn, got {cnn['name']}"
        self.cnn = build_object(**cnn)
        self.name = self.__clas__.__name__

    def plot_step(self):
        raise NotImplementedError()

    @torch.no_grad
    def valid_step(self):
        raise NotImplementedError()

    def train_step(self):
        raise NotImplementedError()


class SampleModel(BaseModel):
    def __init__(self, cnn):
        super(__class__, self).__init__(cnn)
