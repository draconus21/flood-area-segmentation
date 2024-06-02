from torch import nn
from typing import TYPE_CHECKING, List, Any, Optional, Dict

from importlib import import_module

from pathlib import Path
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, field_validator
from typing import Any, Dict

from floodsegment import CONFIG_VERSION
from floodsegment.utils.yaml import load_yaml

if TYPE_CHECKING:
    from floodsegment.dataloader.base import BaseDataset

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
    samplers: str
    model: str
    scheduler: str
    optimizer: str
    criterion: str
    batch_size: int
    num_workers: int

    @field_validator("dataset", "samplers", "model", "scheduler", "optimizer", "criterion")
    def v_valid_file(cls, val):
        v = Path(val).resolve()
        assert v.exists() and v.is_file(), f"Must be a valid file that exists, got {v}"
        return str(v)

    @field_validator("version")
    def v_version(cls, val):
        assert val == CONFIG_VERSION, f"version must be {CONFIG_VERSION}, got {val}"
        return val


class TrainSetup(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid", arbitrary_types_allowed=True)

    version: int
    name: str
    dataset: Any
    samplers: Dict
    dataloaders: Dict
    model: nn.Module
    scheduler: Any
    optimizer: Any
    criterion: Any

    @field_validator("version")
    def v_version(cls, val):
        assert val == CONFIG_VERSION, f"version must be {CONFIG_VERSION}, got {val}"
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
    return construct_x(x_config_path=model_config_path, scope=".net.model.")


def construct_dataset(dataset_config_path: str) -> "BaseDataset":
    return construct_x(
        x_config_path=dataset_config_path, relative_path_keys=["params.split_file"], scope=".dataloader."
    )


def construct_sampler(sample_config_path: str, dataset: "BaseDataset") -> Dict:
    scope = ".dataloader."
    sample_config = load_yaml(sample_config_path)

    samplers = {}
    for k in sample_config:
        s_config = BuildableType(**sample_config[k])
        assert scope in s_config.name, f"Must be placed in {scope}, got {s_config.name}"
        samplers[k] = build_object(**s_config.model_dump(mode="str"), overrides={"data_source": dataset})
    return samplers


def constrcut_optimizer(optimizer_config_path: str, model: nn.Module):
    scope = ".optim."
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) > 1, f"Model must have at least 1 param to optimize, got {len(params)}"
    logger.info(f"Found {len(params)} params to optimize")

    return construct_x(x_config_path=optimizer_config_path, scope=".optim.", overrides={"params": params})


def construct_criterion(criterion_config_path: str):
    return construct_x(x_config_path=criterion_config_path)


def constrcut_scheduler(scheduler_config_path: str, optimizer):
    return construct_x(x_config_path=scheduler_config_path, scope=".optim.", overrides={"optimizer": optimizer})


def construct_x(
    x_config_path: str, relative_path_keys: List[str] = [], scope: Optional[str] = None, overrides: Dict = {}
) -> Any:
    x_config = BuildableType(**load_yaml(x_config_path, relative_path_keys=relative_path_keys))
    if scope:
        assert scope in x_config.name, f"Must be placed in {scope}, got {x_config.name}"

    return build_object(**x_config.model_dump(mode="str"), overrides=overrides)


def load_train_config(train_config_path: str) -> TrainConfig:
    relative_path_keys = [
        k for k in TrainConfig.model_fields if k not in ["name", "version", "batch_size", "num_workers"]
    ]
    return TrainConfig(**load_yaml(train_config_path, relative_path_keys=relative_path_keys))
