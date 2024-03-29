import yaml
from torch import nn
from importlib import import_module

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, field_validator
from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)


class BaseModel(_BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class SetupConfig(BaseModel):
    model: Dict[str, Any]


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


def load_yaml(yaml_path: str) -> Dict:
    ydict = {}
    with open(yaml_path, "r") as y:
        ydict = yaml.full_load(y)
    assert ydict is not None

    if "_parent_" in ydict:
        pdict = load_yaml(ydict.pop("_parent_"))
        ydict = {**pdict, **ydict}

    return ydict


def construct_model(model_config: SetupConfig | Dict[str, Any]) -> nn.Module:
    config_dict = model_config if isinstance(model_config, SetupConfig) else SetupConfig(**load_yaml(model_config))

    assert (
        ".net.model." in config_dict.model["name"]
    ), f"All models must be placed in model, got {config_dict.model['name']}"
    return build_object(**config_dict.model)
