import torch
from torch import nn
from floodsegment import Mode
from floodsegment.utils.misc import recursive_wrapper
from floodsegment.utils.builder import build_object

from typing import Any, Dict, Tuple

from logging import getLogger

logger = getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, net: Dict[str, Any]):
        super(__class__, self).__init__()
        assert ".net." in net["name"], f"All cnns must be placed in net, got {net['name']}"
        self.net = build_object(**net)
        self.name = self.__class__.__name__

    def to_device(self, data: Dict, device):
        @recursive_wrapper
        def _to_dvc(_data):
            return _data.to(device) if isinstance(_data, torch.Tensor) else _data

        return _to_dvc(data)

    def compute_loss(self, sample, *, criterion, device) -> Tuple:
        raise NotImplementedError()

    @torch.no_grad()
    def plot(self, sample, outputs, *, plotter, global_step, sample_viz=None):
        raise NotImplementedError()

    def plot_step(self, sample, outputs, plotters, global_step, mode, sample_viz=None, frequency=50):
        if global_step % frequency == 0:
            self.plot(sample, outputs, plotter=plotters[mode], global_step=global_step, sample_viz=sample_viz)
            logger.debug(f"Plotted data at global_step: {global_step}")

    def inference_step(self, sample, *, criterion, plotters, device, mode, global_step, sample_viz=None):
        loss, outputs = self.compute_loss(sample, criterion=criterion, device=device)
        self.plot_step(sample, outputs, plotters=plotters, global_step=global_step, mode=mode, sample_viz=sample_viz)
        return loss, outputs

    @torch.no_grad()
    def valid_step(self, sample, *, criterion, plotters, device, global_step, sample_viz=None):
        self.eval()
        logger.debug(f"{self.name} in eval mode")

        loss, outputs = self.inference_step(
            sample,
            criterion=criterion,
            plotters=plotters,
            device=device,
            global_step=global_step,
            sample_viz=sample_viz,
            mode=Mode.VALID,
        )
        return loss, outputs

    def train_step(self, sample, *, optimizer, criterion, plotters, device, global_step, sample_viz=None):
        self.train()
        logger.debug(f"{self.name} in train mode")

        loss, outputs = self.inference_step(
            sample,
            criterion=criterion,
            plotters=plotters,
            device=device,
            global_step=global_step,
            sample_viz=sample_viz,
            mode=Mode.TRAIN,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss, outputs

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.net(data)


class FloodModel(BaseModel):
    def __init__(self, net: Dict[str, Any]):
        super(__class__, self).__init__(net=net)

    def compute_loss(self, sample: BaseModel, *, criterion, device) -> Tuple:
        _sample = self.to(sample.model_dump(mode="python"))
        outputs = self.forward(_sample["image"], device)
        loss = criterion(outputs, sample["mask"])

        return loss, outputs

    @torch.no_grad()
    def plot(self, sample, outputs, plotter, global_step, sample_viz=None):
        self.eval()

        if sample_viz:
            sample_viz(sample, plotter, global_step, global_step)
