import pytest
import numpy as np

from floodsegment.net.cnn import GenericCnn
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


@pytest.mark.parametrize("input_ch", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("init_kernel_size", 2 * np.random.randint(1, 128, size=NUM_TEST_VALS) + 1)
@pytest.mark.parametrize("init_stride", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("init_activation", [{"name": "torch.nn.ReLU"}])
@pytest.mark.parametrize("init_normalization", [{"name": "torch.nn.BatchNorm2d"}])
@pytest.mark.parametrize("layer_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("channel_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("stride_config", np.random.randint(1, 128, size=[3, NUM_TEST_VALS]))
@pytest.mark.parametrize("base_config", [simple_conv])
def test_GenericCnn(
    input_ch,
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
        "input_ch": input_ch,
        "init_kernel_size": init_kernel_size,
        "init_stride": init_stride,
        "init_activation": init_activation,
        "init_normalization": init_normalization,
        "layer_config": layer_config,
        "channel_config": channel_config,
        "stride_config": stride_config,
        "base_config": base_config(),
        "net_name": "test",
    }
    g_cnn = GenericCnn(**cnn_kwargs)

    check_module(
        g_cnn,
        **{
            **cnn_kwargs,
            **{
                "init_activation": build_object(**init_activation),
                "init_normalization": build_object(**init_normalization, params={"num_features": channel_config[0]}),
                "base_config": base_config(),
            },
        },
    )
