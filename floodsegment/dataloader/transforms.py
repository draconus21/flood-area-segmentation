import torch  # needed for eval (dtype) in ToDtype
import numpy as np
from torch.nn import Module
from torchvision.transforms.v2 import Compose
from torchvision.transforms import ToTensor as _ToTensor
from torchvision.transforms.v2 import Transform as _Transform
from torchvision.transforms.v2 import ToDtype as _ToDtype
from typing import Callable, Optional, List, Dict

from floodsegment.utils.builder import build_object

from logging import getLogger

logger = getLogger(__name__)


def init_transforms(transform_dict: dict):
    _trans_params = ["keys", "skip_keys", "io_is_dict"]
    t_compose = []
    for t_name, params in transform_dict.items():
        hyper_params = {k: params.pop(k) for k in _trans_params if k in params}
        t_compose.append(
            Transform(
                transform_obj=build_object(t_name, params=params),
                **hyper_params,
            )
        )
        logger.debug(f"Added transform {t_name}: {t_compose[-1]}")
    return Compose(t_compose)


class Transform(_Transform):
    def __init__(
        self,
        transform_obj: Callable,
        keys: Optional[str | List[str]] = None,
        skip_keys: Optional[str | List[str]] = None,
        io_is_dict: bool = False,
    ):
        super(__class__, self).__init__()
        if keys and not isinstance(keys, (str, list)):
            raise ValueError(f"keys must be of type str or list[str], got {type(keys)}")
        if skip_keys and not isinstance(skip_keys, (str, list)):
            raise ValueError(f"skip_Keys must be of type str or list[str], got {type(skip_keys)}")

        self.io_is_dict = io_is_dict

        self.keys: set = set()
        self.skip_keys: set = set()

        if keys:
            self.keys = set([keys]) if isinstance(keys, str) else set(keys)
        if skip_keys:
            self.skip_keys = set([skip_keys]) if isinstance(skip_keys, str) else set(skip_keys)

        self.transform: Callable = transform_obj

    def __call__(self, data_dict: dict):
        # transform works on dict
        if self.io_is_dict:
            return self.transform(data_dict)

        # transform works on elements of dict
        _keys = self.keys or set(data_dict.keys())
        _keys = _keys - self.skip_keys

        for k in _keys:
            data_dict[k] = self.transform(data_dict[k])
        return data_dict


class Split_RGB(Module):
    def __init__(self, tkeys: List[str]):
        super(__class__, self).__init__()
        self.keys = tkeys

    def __call__(self, x: Dict):
        _keys = [k for k in self.keys if k in x.keys()]

        for k in _keys:
            _x = x.pop(k)
            for ch in range(_x.shape[0]):
                x[f"{k}_{ch}"] = _x[ch].unsqueeze(0)

        return x


class ToTensor(Module):
    def __init__(self):
        super().__init__()
        self._transform = _ToTensor()

    def __call__(self, x):
        if isinstance(x, str) or isinstance(x, torch.Tensor):
            return x
        return self._transform(np.array(x.astype(np.float32)))


class ToDtype(Module):
    def __init__(self, dtype: str, **kwargs):
        super().__init__()
        kwargs["dtype"] = eval(dtype)
        self.kwargs = kwargs
        self._transform = _ToDtype(**kwargs)

    def __call__(self, x):
        return self._transform(x)
