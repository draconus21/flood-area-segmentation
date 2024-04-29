# import floodsegment
import csv
import pytest
import numpy as np
from pathlib import Path
from floodsegment.dataloader.segment import FloodItem, FloodSample, FloodDataset
from floodsegment import DATA_DIR, Mode


from typing import Dict

import logging

logger = logging.getLogger(__name__)


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


@pytest.mark.parametrize("split_file", [DATA_DIR / "flood-default-split.json"])
@pytest.mark.parametrize(
    "split_ratio",
    [
        {"TRAIN": 0.5, "TEST": 0.5, "VALID": 0.5},
        {"TRAIN": 0.5},
        {"bad_train": 0.1},
        *list(set([1, *np.random.rand(3)])),
    ],
)
def test_FloodDataset(split_file: str | Path, split_ratio: float | Dict[str, float]):
    _split_file = split_file if isinstance(split_file, Path) else Path(split_file)
    _split_file = _split_file.absolute()

    if isinstance(split_ratio, dict) and "bad_train" in split_ratio:
        with pytest.raises(ValueError):
            FloodDataset(split_file=split_file, split_ratio=split_ratio)
        return

    fd = FloodDataset(split_file=split_file, split_ratio=split_ratio)

    assert fd.split_file == _split_file
    assert fd.items is not None
    assert Mode("TRAIN").value in fd.items
    assert Mode("VALID").value in fd.items
    assert Mode("TEST").value in fd.items

    n_total = 0
    for k in fd.items:
        n_total += len(fd.items[k])
    assert fd.n_samples == n_total
