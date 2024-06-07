from torch.nn import Module
from typing import Callable, Optional, List

from floodsegment.utils.builder import build_object


def init_transforms(transform_dict: dict):
    _trans_params = ["keys", "skip_keys"]
    t_dict = {}
    for t_name, params in transform_dict.items():
        hyper_params = {k: params.pop(k) for k in _trans_params if k in params}
        t_dict[t_name] = Transform(
            transform_obj=build_object(t_name, params=params),
            **hyper_params,
        )
    return t_dict


class Transform(Module):
    def __init__(
        self,
        transform_obj: Callable,
        keys: Optional[str | List[str]] = None,
        skip_keys: Optional[str | List[str]] = None,
    ):
        super(__class__, self).__init__()
        if keys and not isinstance(keys, (str, list)):
            raise ValueError(f"keys must be of type str or list[str], got {type(keys)}")
        if skip_keys and not isinstance(skip_keys, (str, list)):
            raise ValueError(f"skip_Keys must be of type str or list[str], got {type(skip_keys)}")

        self.keys: set = set()
        self.skip_keys: set = set()

        if keys:
            self.keys = set([keys]) if isinstance(keys, str) else set(keys)
        if skip_keys:
            self.skip_keys = set([skip_keys]) if isinstance(skip_keys, str) else set(skip_keys)

        self.transform: Callable = transform_obj

    def __call__(self, data_dict: dict):
        _keys = self.keys or set(data_dict.keys())
        _keys = _keys - self.skip_keys

        for k in _keys:
            data_dict[k] = self.transform(data_dict[k])
        return data_dict
