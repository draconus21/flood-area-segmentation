import torch
from torch import nn
from floodsegment import Mode
from ..utils.tensors import tensor_to_numpy
from ..utils.viz import quickmatshow_dict
from ..utils.misc import recursive_wrapper
from ..utils.builder import build_object

from typing import Any, Dict, Tuple, Optional

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
    def plot(self, sample, outputs, plotter, *, global_step, sample_viz=None):
        raise NotImplementedError()

    def plot_step(self, sample, outputs, plotter, global_step, sample_viz=None, frequency=500):
        if global_step % frequency == 0:
            self.plot(sample, outputs, plotter=plotter, global_step=global_step, sample_viz=sample_viz)
            logger.debug(f"Plotted data at global_step: {global_step}")

    def inference_step(self, sample, *, criterion, device):
        loss, outputs = self.compute_loss(sample, criterion=criterion, device=device)
        return loss, outputs

    @torch.no_grad()
    def valid_step(self, sample, *, criterion, plotters, device, global_step, sample_viz=None, plot_freq: int = 500):
        self.eval()
        logger.debug(f"{self.name} in eval mode")

        loss, outputs = self.inference_step(sample, criterion=criterion, device=device)
        self.plot_step(
            sample,
            outputs,
            plotter=plotters[Mode.VALID.value],
            global_step=global_step,
            sample_viz=sample_viz,
            frequency=plot_freq,
        )
        return loss, outputs

    def train_step(
        self, sample, *, optimizer, criterion, plotters, device, global_step, sample_viz=None, plot_freq: int = 500
    ):
        self.train()
        logger.debug(f"{self.name} in train mode")

        loss, outputs = self.inference_step(sample, criterion=criterion, device=device)
        self.plot_step(
            sample,
            outputs,
            plotter=plotters[Mode.TRAIN.value],
            global_step=global_step,
            sample_viz=sample_viz,
            frequency=plot_freq,
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

    def compute_loss(self, sample: Dict, *, criterion, device) -> Tuple:
        _sample = self.to_device(sample, device)
        outputs = self.forward(_sample)
        loss = criterion(outputs.flood_mask, _sample["mask"])

        return loss, outputs

    @torch.no_grad()
    def plot(
        self,
        sample,
        outputs,
        plotter,
        *,
        global_step,
        sample_viz=None,
        close: bool = True,
        walltime: Optional[float] = None,
    ):
        self.eval()

        idx = 0
        v_sample = {k: tensor_to_numpy(v[idx]) for k, v in sample.items()}
        v_output = {k: tensor_to_numpy(v[idx]) for k, v in outputs._asdict().items()}

        if sample_viz:
            sample_viz(v_sample, plotter, global_step=global_step, tag="model inputs")

        plotter.add_figure(
            tag="Pred",
            figure=quickmatshow_dict(v_output, title="model outs"),
            global_step=global_step,
            close=close,
            walltime=walltime,
        )
