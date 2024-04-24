from torch import nn
from torch.utils.data import Dataset
from torchsummary import summary

from importlib import import_module

from pathlib import Path
from pydantic import BaseModel as _BaseModel, field_validator
from pydantic import ConfigDict
from typing import Any, Dict

from floodsegment.utils.yaml import load_yaml

import logging

logger = logging.getLogger(__name__)


class BaseModel(_BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class BuildableType(BaseModel):
    name: str
    params: Dict[str, Any]


class TrainConfig(BaseModel):
    version: int
    name: str
    dataset: str
    sampler: str
    model: str
    scheduler: str
    optimizer: str
    criterion: str

    @field_validator("dataset", "model", "scheduler", "optimizer", "criterion")
    def v_valid_file(cls, val):
        v = Path(val).absolute()
        assert v.exists() and v.is_file(), f"Must be a valid file that exists, got {v}"
        return str(v)

    @field_validator("version")
    def v_version(cls, val):
        assert val == 1, f"version must be 1, got {val}"
        return val


def build_object(name: str, params: Dict[str, Any] = {}, overrides: Dict[str, Any] = {}) -> Any:
    """
    build_object(somemodule.submodule.MyClass) -> from somemodule.submodule import MyClass; return MyClass()
    build_object(somemodule.submodule.MyClass, params={"my_class_param1": "blah", "my_class_param2": 56}) -> from somemodule.submodule import MyClass; return MyClass(**params)
    """

    c_name = get_class(name)
    try:
        return c_name(**{**params, **overrides})
    except TypeError as e:
        logger.error(f"{e}, params: {params}, overrides: {overrides}")
        raise TypeError(f"Caught {e}. See error log for more information.") from e


def get_class(name: str) -> Any:
    assert len(name.rsplit(".", 1)) > 1, f"name must be the full path to the class you wish to get; got {name}"

    module_name, class_name = name.rsplit(".", 1)

    module = import_module(module_name)
    return getattr(module, class_name)


def construct_model(model_config_path: str) -> nn.Module:
    model_config = BuildableType(**load_yaml(model_config_path))

    assert ".net.model." in model_config.name, f"All models must be placed in model, got {model_config.name}"
    model = build_object(**model_config.model_dump(mode="str"))
    logger.debug(f"{model.net.net_name}\n{summary(model)}")
    return model


def construct_dataset(dataset_config_path: str) -> Dataset:
    relative_path_keys = ["params.split_file"]
    dataset_config = BuildableType(**load_yaml(dataset_config_path, relative_path_keys=relative_path_keys))
    assert (
        ".dataloader." in dataset_config.name
    ), f"All datasets must be placed in dataloader, got {dataset_config.name}"
    return build_object(**dataset_config.model_dump(mode="str"))


def load_train_config(train_config_path: str) -> TrainConfig:
    relative_path_keys = [k for k in TrainConfig.model_fields if k != "version"]
    return TrainConfig(**load_yaml(train_config_path, relative_path_keys=relative_path_keys))
