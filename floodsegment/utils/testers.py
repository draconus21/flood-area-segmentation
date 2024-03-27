def check_module(test_layer, **vals_to_test):
    for k, v in vals_to_test.items():
        assert hasattr(test_layer, k), f"Expecting to find attribute {k}"

        test_val = getattr(test_layer, k)
        if k in ["activation", "normalization"]:
            assert type(test_val) == type(v)
        else:
            assert test_val == v, f"Expecting {v} for {k}, but got {test_val}"
