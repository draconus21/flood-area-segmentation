import torch.nn as nn
from floodsegment.utils.builder import build_object

from typing import Any, Dict

"""
Heirarchy

Model -> Net [UNet] -> Cnn [Encoder/Decoder] -> Blocks [GenericBlock] -> Module/Layer
"""


class _Buildable(nn.Module):
    def __init__(self, **kwargs):
        super(__class__, self).__init__()
        self.block = self._build(**kwargs)

    def _build(self, **kwargs) -> nn.ModuleList:
        raise NotImplementedError(f"Please implement this fucntion for your class ({self.__class__.__name__})")

    def forward(self, x):
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
        input_ch: int,
        output_ch: int,
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
        net_name: str = "",
    ):
        super(__class__, self).__init__()

        self.net_name = net_name
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.encoder_config = encoder
        self.decoder_config = decoder
        self.encoder = self.build_encoder(self.encoder_config)
        self.decoder = self.build_decoder(self.decoder_config)

    def build_encoder(self, encoder_config: Dict[str, Any]) -> nn.Module:
        return build_object(**encoder_config, overrides={"input_ch": self.input_ch})

    def build_decoder(self, decoder_config: Dict[str, Any]) -> nn.Module:
        return build_object(**decoder_config, overrides={"input_ch": self.input_ch})
