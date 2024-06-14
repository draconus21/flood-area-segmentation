import torch.nn as nn
from floodsegment.net.base import BaseEDNet
from floodsegment.net.cnn import GenericDecoder

from typing import Dict, Any, List, Tuple, OrderedDict

from logging import getLogger

logger = getLogger(__name__)


class AENet(BaseEDNet):
    def __init__(
        self,
        img_size: Tuple[int, int],  # height, width
        input_names: List[str],
        output_names: List[str],
        encoder: Dict[str, Any],
        decoder_upsample_config: Dict[str, Any],
        net_name: str,
    ):
        self.decoder_upsample_config = decoder_upsample_config
        super(__class__, self).__init__(
            img_size=img_size,
            input_names=input_names,
            output_names=output_names,
            encoder=encoder,
            decoder={},
            net_name=net_name,
        )

    def get_decoder_out_key(self, *, enc_out_key: str = "", dec_idx: int = -1):
        if (not enc_out_key and dec_idx < 0) or (enc_out_key and dec_idx > -1):
            raise ValueError(
                f"Must provide exactly ONE of the following: valid encoder out key (enc_out_key: {enc_out_key}) or decoder index (dec_idx: {dec_idx})"
            )

        return (
            f"dec_{enc_out_key}"
            if enc_out_key
            else self.get_decoder_out_key(enc_out_key=self.encoder.out_key_at(dec_idx))
        )

    def build_decoder(self, decoder_config: Dict[str, Any]) -> nn.Module:
        """
        Construct decoder by inverting encoder config
        """
        _encoder_params = self.encoder_config["params"]
        _encoder_out_sizes = OrderedDict({k: v for k, v in self.encoder_out_sizes.items()})
        _encoder_out_sizes.popitem(last=True)  # remove last layer size

        _dec_output_channel = self.output_ch
        _dec_output_kernel_size = _encoder_params["init_kernel_size"]
        _dec_output_stride = _encoder_params["init_stride"]
        _dec_output_activation = _encoder_params["init_activation"]
        _dec_output_normalization = _encoder_params["init_normalization"]

        _dec_size_config = []
        for i in range(len(self.encoder) - 2, -1, -1):
            _dec_size_config.append(_encoder_out_sizes[self.encoder.out_key_at(i)][1:])

        _dec_layer_config = _encoder_params["layer_config"][::-1]
        _dec_channel_config = _encoder_params["channel_config"][::-1]
        # _dec_stride_config = _encoder_params["stride_config"][::-1]
        # _dec_dilation_config = _encoder_params["dilation_config"][::-1]

        _dec_input_ch = _dec_channel_config[0]
        # _dec_layer_config = _dec_layer_config[1:]
        # _dec_channel_config = _dec_channel_config[1:]
        _dec_stride_config = [1] * len(_dec_channel_config)  # _dec_stride_config[1:]
        # _dec_dilation_config= _dec_dilation_config[1:]

        _dec_base_config = _encoder_params["base_config"]
        _dec_up_config = self.decoder_upsample_config
        _decoder = GenericDecoder(
            input_ch=_dec_input_ch,
            output_ch=_dec_output_channel,
            out_kernel_size=_dec_output_kernel_size,
            out_stride=_dec_output_stride,
            out_activation=_dec_output_activation,
            out_normalization=_dec_output_normalization,
            size_config=_dec_size_config,
            layer_config=_dec_layer_config,
            channel_config=_dec_channel_config,
            stride_config=_dec_stride_config,
            upsample_config=_dec_up_config,
            base_config=_dec_base_config,
        )

        return _decoder
