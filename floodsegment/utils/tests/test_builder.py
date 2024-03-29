import os
import pytest
from floodsegment import CONFIG_DIR
from floodsegment.utils.misc import get_files_in_dir
from floodsegment.utils.builder import construct_model

import logging

logger = logging.getLogger(__name__)

MODEL_CONFIG_DIR = os.path.join(CONFIG_DIR, "model")


@pytest.mark.parametrize("model_config", get_files_in_dir(MODEL_CONFIG_DIR, "*.yaml", recursive=True))
def test_construct_mode(model_config):
    m = construct_model(model_config)
    logger.info(m)
