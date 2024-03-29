import pytest
from pathlib import Path
from floodsegment import CONFIG_DIR
from floodsegment.utils.misc import get_files_in_dir
from floodsegment.utils.yaml import load_yaml
from floodsegment.utils.builder import construct_model, load_train_config, ModelType

import logging
from floodsegment.utils.logutils import setupLogging

setupLogging(root_level="DEBUG")

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model_config_path", get_files_in_dir(Path(CONFIG_DIR) / "model", "*.yaml", recursive=True))
def test_construct_mode(model_config_path):
    model_config = ModelType(**load_yaml(model_config_path))
    construct_model(model_config)


@pytest.mark.parametrize("train_config_path", get_files_in_dir(Path(CONFIG_DIR) / "train", "*.yaml", recursive=True))
def test_train_setup(train_config_path):
    tconfig = load_train_config(train_config_path)
    model = construct_model(tconfig.model)
    assert tconfig.model
