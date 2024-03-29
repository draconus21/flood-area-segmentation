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
        _activation = build_object(**activation)
        _extra_norm_params = {}
        if "torch.nn.BatchNorm" in normalization["name"]:
            _extra_norm_params["num_features"] = out_channels

        _normalization = build_object(**normalization, params=_extra_norm_params)
        super(__class__, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=_activation,
            normalization=_normalization,
            **kwargs,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = _activation
        self.normalization = _normalization


class BaseEDNet(nn.Module):
    def __init__(
        self,
        input_ch: int,
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
        net_name: str = "",
    ):
        super(__class__, self).__init__()

        self.net_name = net_name
        self.input_ch = input_ch
        self.encoder_config = encoder
        self.decoder_config = decoder
        self.encoder = self.build_encoder(self.encoder_config)
        self.decoder = self.build_decoder(self.decoder_config)

    def build_encoder(self, encoder_config: Dict[str, Any]) -> nn.Module:
        return build_object(**encoder_config, overrides={"input_ch": self.input_ch})

    def build_decoder(self, decoder_config: Dict[str, Any]) -> nn.Module:
        return build_object(**decoder_config, overrides={"input_ch": self.input_ch})
