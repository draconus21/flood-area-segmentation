import os
import pytest
from floodsegment import PKG_DIR
from floodsegment.utils.misc import get_files_in_dir
from floodsegment.utils.builder import construct_model

import logging

logger = logging.getLogger(__name__)

TESTDATA_DIR = os.path.join(PKG_DIR, "floodsegment", "utils", "tests", "testdata")


@pytest.mark.parametrize("model_config", get_files_in_dir(TESTDATA_DIR, "*.yaml", recursive=True))
def test_construct_mode(model_config):
    construct_model(model_config)
