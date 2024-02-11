# import floodsegment
import csv
import numpy as np
from pathlib import Path
from floodsegment.dataloader.segment import FloodItem, FloodSample, FloodDataset
from floodsegment import DATA_DIR

from pytest import mark

import logging

logger = logging.getLogger(__name__)

from floodsegment.utils.logutils import setupLogging

setupLogging(console_level="DEBUG", root_level="DEBUG")


def test_basic(data_dir: Path = DATA_DIR):
    with open(data_dir / "metadata.csv", "r") as metadata_file:
        m_reader = csv.reader(metadata_file)
        # assumption: the csv has at least 100 rows (excluding the header)
        # pick a random row from the first 100 rows
        row_num = np.random.randint(1, 101)
        for i, row in enumerate(m_reader):
            if i != row_num:
                continue
            break
    im_path = DATA_DIR / "Image" / row[0]
    ma_path = DATA_DIR / "Mask" / row[1]
    logger.debug(f"Using [image, mask]={row}")
    fi = FloodItem(image=im_path, mask=str(ma_path))

    assert isinstance(fi.image, Path), f"FloodItem.image must be a pathlib.Path object, got {type(fi.image)}"
    assert isinstance(fi.mask, Path), f"FloodItem.mask must be a pathlib.Path object , got {type(fi.mask)}y"

    assert fi.image == im_path
    assert fi.mask == Path(ma_path)

    fs = FloodSample(flood_item=fi)
    assert len(fs.image.shape) == 3
    assert len(fs.mask.shape) == 2


@mark.parametrize("split_file", [DATA_DIR / "flood-default-split.json"])
def test_FloodDataset(split_file: str | Path):
    _split_file = split_file if isinstance(split_file, Path) else Path(split_file)
    _split_file = _split_file.absolute()

    fd = FloodDataset(split_file=split_file)

    assert fd.split_file == _split_file
    assert fd._items is not None
    assert "train" in fd.items
    assert "valid" in fd.items
    assert "test" in fd.items
