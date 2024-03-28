import numpy as np
from floodsegment.utils.misc import recursive_wrapper


def check_module(test_layer, **vals_to_test):
    for k, v in vals_to_test.items():
        assert hasattr(test_layer, k), f"Expecting to find attribute {k}"

        test_val = getattr(test_layer, k)
        if k in ["init_activation", "init_normalization", "activation", "normalization"]:
            assert type(test_val) == type(v)
        else:
            assert type(v) == type(test_val), f"Expecting {K} of type {type(v)}, but got {type(test_val)}"
            if isinstance(v, list) or isinstance(v, np.ndarray):
                assert np.all(v == test_val), f"Expecting {v} for {k}, but got {test_val}"
            else:
                assert v == test_val, f"Expecting {v} for {k}, but got {test_val}"
