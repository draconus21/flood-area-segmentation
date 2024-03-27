import pytest
import numpy as np
from floodsegment.utils.builder import build_object
from floodsegment.net.module import SimpleConvLayer


@pytest.mark.parametrize("in_channels", np.random.randint(1, 128, size=3))
@pytest.mark.parametrize("out_channels", np.random.randint(1, 128, size=3))
@pytest.mark.parametrize("activation", [{"name": "torch.nn.ReLU"}])
@pytest.mark.parametrize("normalization", [{"name": "torch.nn.BatchNorm2d"}])
@pytest.mark.parametrize("kernel_size", 2 * np.random.randint(1, 128, size=3) + 1)
@pytest.mark.parametrize("dilation", np.random.randint(1, 128, size=3))
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

    assert c_layer.in_channels == in_channels
    assert c_layer.out_channels == out_channels
    assert type(c_layer.activation) == type(build_object(**activation))
    assert type(c_layer.normalization) == type(build_object(**normalization, params={"num_features": in_channels}))
    assert c_layer.kernel_size == kernel_size
    assert c_layer.dilation == dilation
    assert c_layer.kwargs == kwargs


@pytest.mark.parametrize("kernel_size", 2 * np.random.randint(1, 128, size=3))
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
