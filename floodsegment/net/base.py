import torch
import torch.nn as nn
from torch import randn as trandn
from floodsegment.utils.builder import build_object

from collections import namedtuple
from typing import Any, Dict, OrderedDict, Tuple, List, NamedTuple

"""
Heirarchy

Model -> Net [UNet] -> Cnn [Encoder/Decoder] -> Blocks [GenericBlock] -> Module/Layer
"""


class _Buildable(nn.Module):
    def __init__(self, **kwargs):
        super(__class__, self).__init__()
        self.block: nn.ModuleList | nn.Sequential = self._build(**kwargs)

    def _build(self, **kwargs) -> nn.ModuleList | nn.Sequential:
        raise NotImplementedError(f"Please implement this fucntion for your class ({self.__class__.__name__})")

    def out_key_at(self, idx: int):
        return f"{self}_{idx}"

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, x, keep_interim_outs=False) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        outs = OrderedDict()
        outs[self.out_key_at(0)] = self.block[0](x)
        for i in range(1, len(self)):
            outs[self.out_key_at(i)] = self.block[i](outs[self.out_key_at(i - 1)])

        return outs if keep_interim_outs else outs[self.out_key_at(len(self) - 1)]

    def fxorward(self, x):
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
        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            normalization=normalization,
            **kwargs,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.normalization = normalization


class BaseEDNet(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],  # height, widith
        input_names: List[str],
        output_names: List[str],
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
        net_name: str = "",
    ):
        assert len(img_size) == 2, f"img_size must be [h, w], got {img_size}"
        super(__class__, self).__init__()

        self.net_name = net_name
        self.input_names = input_names
        self.output_names = output_names
        self.img_size = img_size
        self.input_ch = len(self.input_names)
        self.output_ch = len(self.output_names)
        self.encoder_config = encoder
        self.decoder_config = decoder

        self.encoder = self.build_encoder(self.encoder_config)
        self.encoder_out_sizes = self.encoder.get_out_sizes(input_size=[self.input_ch, *self.img_size])

        self.decoder = self.build_decoder(self.decoder_config)

        self.Output_Object = namedtuple("Output_Object", self.output_names)

        self.validate_io()

    def forward(self, input_dict: Dict[str, Any]) -> NamedTuple:
        assert all(
            [input_name in input_dict for input_name in self.input_names]
        ), f"Expecting inputs: {self.input_names}, but got {input_dict.keys()}"

        input_tensor = torch.cat([input_dict[in_name] for in_name in self.input_names], dim=1)

        _outputs = self.decoder(self.encoder(input_tensor))
        if isinstance(_outputs, torch.Tensor):
            assert len(self.output_names) == 1, f"Expecting {len(self.output_names)} outputs, but only got 1"
            _outputs = [_outputs]
        elif isinstance(_outputs, list) and isinstance(_outputs[0], torch.Tensor):
            assert (
                len(self.output_names) == 1
            ), f"Expecting {len(self.output_names)} outputs, but only got {len(_outputs)} outputs"
        else:
            raise ValueError(
                f"Output must either be a torch Tensor or a list of torch Tensors, but got {type(_outputs)}"
            )

        return self.Output_Object(*_outputs)

    def validate_io(self):
        inputs, outputs = self.dummies()
        assert self.input_names == list(inputs.keys()), f"Expecting inputs {self.input_names}, got {inputs.keys()}"
        assert self.output_names == list(
            outputs._asdict().keys()
        ), f"Expecting inputs {self.output_names}, got {outputs._asdict().keys()}"

    def build_encoder(self, encoder_config: Dict[str, Any]) -> nn.Module:
        return build_object(**encoder_config, overrides={"input_size": [self.input_ch, *self.img_size]})

    def build_decoder(self, decoder_config: Dict[str, Any]) -> nn.Module:
        return build_object(**decoder_config, overrides={"input_ch": self.input_ch})

    def get_dummy_inputs(self) -> Dict[str, Any]:
        return {x: trandn(1, 1, *self.img_size) for x in self.input_names}

    def get_dummy_outputs(self) -> NamedTuple:
        self.eval()
        return self.forward(self.get_dummy_inputs())

    def dummies(self) -> Tuple[Dict[str, Any], NamedTuple]:
        return self.get_dummy_inputs(), self.get_dummy_outputs()
