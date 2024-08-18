import pytest
import numpy as np

from floodsegment.net.cnn import GenericEncoder, GenericDecoder
from floodsegment.utils.builder import build_object
from floodsegment.utils.testers import check_module

from typing import Dict, Any

NUM_TEST_VALS = 1


def simple_conv():
    return {
        "name": "floodsegment.net.module.SimpleConvLayer",
        "params": {
            "kernel_size": 5,
            "activation": {"name": "torch.nn.ReLU"},
            "normalization": {"name": "torch.nn.BatchNorm2d"},
        },
    }


def remove_act_norm(config: Dict[str, Any]):
    assert "params" in config
    keys_to_pop = ["activation", "normalization"]
    _copy_dict = {k: v for k, v in config.items()}
    for k in keys_to_pop:
        if k in _copy_dict["params"]:
            _copy_dict["params"].pop(k)
    return _copy_dict


@pytest.mark.parametrize("input_size", np.random.randint(1, 128, size=[NUM_TEST_VALS, 3]))
@pytest.mark.parametrize("init_kernel_size", 2 * np.random.randint(1, 128, size=NUM_TEST_VALS) + 1)
@pytest.mark.parametrize("init_stride", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("init_activation", [{"name": "torch.nn.ReLU"}])
@pytest.mark.parametrize("init_normalization", [{"name": "torch.nn.BatchNorm2d"}])
@pytest.mark.parametrize("layer_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("channel_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("stride_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("base_config", [simple_conv()])
def test_GenericEncoder(
    input_size,
    init_kernel_size,
    init_stride,
    init_activation,
    init_normalization,
    layer_config,
    channel_config,
    stride_config,
    base_config,
):
    cnn_kwargs = {
        "input_size": input_size,
        "init_kernel_size": init_kernel_size,
        "init_stride": init_stride,
        "init_activation": init_activation,
        "init_normalization": init_normalization,
        "layer_config": layer_config,
        "channel_config": channel_config,
        "stride_config": stride_config,
        "base_config": base_config,
        "net_name": "test",
    }
    g_cnn = GenericEncoder(**cnn_kwargs)

    check_module(
        g_cnn,
        **{
            **cnn_kwargs,
            **{
                "init_activation": init_activation,
                "init_normalization": init_normalization,
                "base_config": base_config,
            },
        },
    )


@pytest.mark.parametrize(("input_ch", "output_ch"), np.random.randint(1, 128, size=[NUM_TEST_VALS, 2]))
@pytest.mark.parametrize("out_kernel_size", 2 * np.random.randint(1, 128, size=NUM_TEST_VALS) + 1)
@pytest.mark.parametrize("out_stride", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("out_activation", [{"name": "torch.nn.ReLU"}])
@pytest.mark.parametrize("out_normalization", [{"name": "torch.nn.BatchNorm2d"}])
@pytest.mark.parametrize("size_config", np.random.randint(1, 128, size=[NUM_TEST_VALS, 2]))
@pytest.mark.parametrize("layer_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("channel_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("stride_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("upsample_config", [{"name": "floodsegment.net.module.UpsampleBilinear2d"}])
@pytest.mark.parametrize("base_config", [simple_conv()])
def test_GenericDecoder(
    input_ch,
    output_ch,
    out_kernel_size,
    out_stride,
    out_activation,
    out_normalization,
    size_config,
    layer_config,
    channel_config,
    stride_config,
    upsample_config,
    base_config,
):
    cnn_kwargs = {
        "input_ch": input_ch,
        "output_ch": output_ch,
        "out_kernel_size": out_kernel_size,
        "out_stride": out_stride,
        "out_activation": out_activation,
        "out_normalization": out_normalization,
        "size_config": size_config,
        "layer_config": layer_config,
        "channel_config": channel_config,
        "stride_config": stride_config,
        "upsample_config": upsample_config,
        "base_config": base_config,
        "net_name": "test",
    }
    g_cnn = GenericDecoder(**cnn_kwargs)
    check_module(
        g_cnn,
        **{
            **cnn_kwargs,
            **{
                "out_activation": out_activation,
                "out_normalization": out_normalization,
                "base_config": base_config,
            },
        },
    )
