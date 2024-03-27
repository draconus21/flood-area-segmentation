import pytest
import torch.nn
from floodsegment.net.module import SimpleConvLayer


@pytest.mark.parametrize("in_channels", [2, 16, 32])
@pytest.mark.parametrize("out_channels", [2, 14, 55])
@pytest.mark.parametrize("activation", [torch.nn.ReLU])
@pytest.mark.parametrize("normalization", [torch.nn.BatchNorm2d])
@pytest.mark.parametrize("kernel_size", [2, 3, 4])
@pytest.mark.parametrize("dilation", [2, 3, 4])
@pytest.mark.parametrize("kwargs", [{}, {"dummy": 2}])
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
    assert c_layer.activation == activation
    assert c_layer.normalization == normalization
    assert c_layer.kernel_size == kernel_size
    assert c_layer.dilation == dilation
    assert c_layer.kwargs == kwargs
