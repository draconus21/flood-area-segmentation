import pytest
import numpy as np
from floodsegment.utils.builder import build_object
from floodsegment.utils.testers import check_module
from floodsegment.net.module import SimpleConvLayer, GenericBlock


NUM_TEST_VALS = 2


@pytest.mark.parametrize("in_channels", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("out_channels", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("activation", [{"name": "torch.nn.ReLU"}])
@pytest.mark.parametrize("normalization", [{"name": "torch.nn.BatchNorm2d"}])
@pytest.mark.parametrize("kernel_size", 2 * np.random.randint(1, 128, size=NUM_TEST_VALS) + 1)
@pytest.mark.parametrize("dilation", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("kwargs", [{}])
def test_SimpleConvLayer(in_channels, out_channels, activation, normalization, kernel_size, dilation, kwargs):
    c_layer = SimpleConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        normalization=normalization,
        kernel_size=kernel_size,
        dilation=dilation,
        **kwargs,
    )

    check_module(
        c_layer,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=build_object(**activation),
        normalization=build_object(**normalization, params={"num_features": in_channels}),
        kernel_size=kernel_size,
        dilation=dilation,
        **kwargs,
    )


@pytest.mark.parametrize("kernel_size", 2 * np.random.randint(1, 128, size=NUM_TEST_VALS))
def test_SimpleConvLayer_fail(kernel_size):
    in_channels = np.random.randint(1, 128)
    out_channels = np.random.randint(1, 128)
    dilation = np.random.randint(1, 128)
    activation = {"name": "torch.nn.ReLU"}
    normalization = {"name": "torch.nn.BatchNorm2d"}
    kwargs = {}
    with pytest.raises(AssertionError):
        c_layer = SimpleConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            normalization=normalization,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs,
        )


@pytest.mark.parametrize("in_channels", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("out_channels", np.random.randint(1, 128, size=NUM_TEST_VALS))
@pytest.mark.parametrize("stride", np.random.randint(1, 16, size=NUM_TEST_VALS))
@pytest.mark.parametrize("n_layers", np.random.randint(1, 64, size=NUM_TEST_VALS))
def test_GenericBlock(in_channels, out_channels, stride, n_layers):
    base_config = {
        "name": "floodsegment.net.module.SimpleConvLayer",
        "params": {
            "kernel_size": 5,
            "activation": {"name": "torch.nn.ReLU"},
            "normalization": {"name": "torch.nn.BatchNorm2d"},
        },
    }

    block = GenericBlock(
        in_channels=in_channels, out_channels=out_channels, stride=stride, n_layers=n_layers, base_config=base_config
    )

    assert len(block) == n_layers
    check_module(block, in_channels=in_channels, out_channels=out_channels, n_layers=n_layers, stride=stride)

    ksize = base_config["params"]["kernel_size"]
    for i in range(n_layers - 1):
        mod = block[i]
        check_module(
            mod,
            in_channels=in_channels,
            out_channels=in_channels,
            activation=build_object(**base_config["params"]["activation"]),
            normalization=build_object(**base_config["params"]["normalization"], params={"num_features": in_channels}),
            kernel_size=ksize,
        )
        check_module(
            mod[0],  # test conv layer
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(ksize, ksize),
            stride=(1, 1),
        )

    ## tests for last layer
    last_mod = block[-1]
    check_module(
        last_mod,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=build_object(**base_config["params"]["activation"]),
        normalization=build_object(**base_config["params"]["normalization"], params={"num_features": in_channels}),
        kernel_size=ksize,
    )
    check_module(
        last_mod[0],  # test conv layer
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(ksize, ksize),
        stride=(stride, stride),
    )
