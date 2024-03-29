import torch.nn as nn
from floodsegment.net.base import BaseEDNet
from floodsegment.net.cnn import GenericDecoder

from typing import Dict, Any


class AENet(BaseEDNet):
    def __init__(self, input_ch: int, output_ch: int, encoder: Dict[str, Any], net_name: str):
        super(__class__, self).__init__(input_ch=input_ch, encoder=encoder, decoder={}, net_name=net_name)

    def build_decoder(self, decoder_config: Dict[str, Any]) -> nn.Module:
        """
        Construct decoder by inverting encoder config
        """
        _encoder_params = self.encoder_config["params"]

        _dec_output_channel = self.output_ch
        _dec_output_kernel_size = _encoder_params["init_kernel_size"]
        _dec_output_stride = _encoder_params["init_stride"]
        _dec_output_activation = _encoder_params["init_activation"]
        _dec_output_normalization = _encoder_params["init_normalization"]

        _dec_layer_config = _encoder_params["layer_config"][::-1]
        _dec_channel_config = _encoder_params["channel_config"][::-1]
        _dec_stride_config = _encoder_params["stride_config"][::-1]
        # _dec_dilation_config = _encoder_params["dilation_config"][::-1]
        _dec_base_config = _encoder_params["base_config"]

        _decoder = GenericDecoder(
            output_ch=_dec_output_channel,
            out_kernel_size=_dec_output_kernel_size,
            out_stride=_dec_output_stride,
            out_activation=_dec_output_activation,
            out_normalization=_dec_output_normalization,
            layer_config=_dec_layer_config,
            channel_config=_dec_channel_config,
            stride_config=_dec_stride_config,
            base_config=_dec_base_config,
        )

        return _decoder
