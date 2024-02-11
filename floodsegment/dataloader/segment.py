import click
import csv
import json
import random
import warnings
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from torch.utils.data import Dataset

from typing import Dict, List, Tuple

from floodsegment import DATA_DIR
from floodsegment.utils.logutils import setupLogging

import logging

logger = logging.getLogger(__name__)


def load_img(im_path: str) -> np.ndarray:
    return np.array(iio.imread(im_path))


@click.command()
@click.option(
    "--split-file-name",
    "-s",
    default="flood-default-split.json",
    help="Name of the split file to write splits to.",
    show_default=True,
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DATA_DIR,
    help="Root directory of flood-segment dataset",
    show_default=True,
)
@click.option("--train-split", "-train", default=0.7, help="train split ratio", show_default=True)
@click.option("--valid-split", "-valid", default=0.15, help="valid split ratio", show_default=True)
@click.option("--test-split", "-test", default=0.15, help="testsplit ratio", show_default=True)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="log level",
    show_default=True,
)
def generate_split_cli(split_file_name, data_dir, train_split, valid_split, test_split, log_level):
    """
    This script will generate a train-valid-test split for your flood segment dataset (https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)
    Assumes the following structure

    \b
    data_dir
        |-- metadata.csv
        |-- Image (folder w/ rgb jpg images)
        |-- Mask (folder w/ binary png images)
    """
    setupLogging(log_level, log_level)
    generate_split(
        split_file_name=split_file_name,
        data_dir=data_dir,
        splits={"train": train_split, "valid": valid_split, "test": test_split},
    )


def generate_split(
    split_file_name: str | Path,
    data_dir: str | Path,
    splits: Dict[str, float] = {"train": 0.7, "valid": 0.15, "test": 0.15},
):
    path_dict = {}
    r = 0
    for sn, sr in splits.items():
        r = r + sr
        path_dict[sn] = []
    assert r == 1, f"Ratios for splits must add up to 1: {splits}"

    logger.debug(f"Using split ratio: {splits}")

    _data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
    _split_file_name = (_data_dir / split_file_name).absolute().with_suffix(".json")

    m_data: List[FloodItem] = []
    with open(_data_dir / "metadata.csv", "r") as m:
        m_reader = csv.reader(m)
        header = m_reader.__next__()
        assert header == ["Image", "Mask"]

        for row in m_reader:
            m_data.append(
                FloodItem(
                    image=_data_dir / header[0] / row[0],
                    mask=_data_dir / header[1] / row[1],
                ).model_dump(mode="json")
            )

    logger.debug(f"Found {len(m_data)} samples in {_data_dir}")

    # check if there is enough data
    sizes = {k: max(1, int(v * len(m_data))) for k, v in splits.items()}
    total_sizes = sum(sizes.values())
    if total_sizes > len(m_data):
        logger.warning(f"Not enough data for required splits: Needs {total_sizes}, got {len(m_data)}")
        max_k = max(sizes, key=sizes.get)
        excess = total_sizes - len(m_data)
        sizes[max_k] = sizes[max_k] - excess  # TODO: instead distribute excess amongst all keys, maybe based on ratio?
        logger.warning(f"Using updated splits to instead: {sizes}")
    elif total_sizes < len(m_data):
        logger.warning(f"Current ratios do not use all data. Using {total_sizes}, available {len(m_data)}")
        min_k = min(sizes, key=sizes.get)
        excess = len(m_data) - total_sizes
        sizes[min_k] = sizes[min_k] + excess  # TODO: instead distribute excess amongst all keys, maybe based on ratio?
        logger.warning(f"Using updated splits to instead: {sizes}")

    # shuffle
    random.shuffle(m_data)

    split_json = {}
    cur_idx = 0
    for split_name, split_len in sizes.items():
        end_idx = cur_idx + split_len
        split_json[split_name] = m_data[cur_idx:end_idx]
        cur_idx = end_idx

    with open(_split_file_name, "w") as s_file:
        json.dump(split_json, s_file, indent=4)

    logger.info(f"split file writted to {_split_file_name}")


class FloodItem(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)  # , arbitrary_types_allowed=True)

    image: Path
    mask: Path

    @field_validator("image", "mask")
    def m_before(cls, val):
        # if isinstance(val, np.ndarray):
        #    return val

        if not isinstance(val, str) and not isinstance(val, Path):
            raise TypeError(f"Must be of type pathlib.Path or str, got {type(val)}")

        if isinstance(val, str):
            _val = Path(val).absolute()
        else:
            _val = val

        valid_exts = [".png", ".jpg"]
        assert _val.exists(), f"{str(_val)} does not exists."
        assert _val.is_file(), f"{str(_val)} must be a file."
        assert _val.suffix in valid_exts, f"{str(_val)} must be of type {valid_exts}"
        return _val
        # return load_img(_val)


class FloodSample(BaseModel):
    """
    Must be init w/ flood_item
    sample = FloodSample(flood_item=flood_item)
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    image: np.ndarray
    mask: np.ndarray

    @model_validator(mode="before")
    @classmethod
    def m_before(cls, data: Dict):
        assert "flood_item" in data

        fitem = data.pop("flood_item")
        assert isinstance(fitem, FloodItem)

        _data = {"image": load_img(fitem.image), "mask": load_img(fitem.mask)}

        return _data


class FloodDataset(Dataset):
    def __init__(self, split_file: str | Path, transform_dict: Dict = {}, split_ratio=float | Dict[str, float]):
        self.split_file: Path = None
        self.items: Dict[str, List[FloodSample]] = {}
        self.n_samples: int = 0

        self.split_ratio: Dict[str, float] = {}
        self.transform_dict = transform_dict

        self._update_from_split_file(split_file=split_file, split_ratio=split_ratio)

    def process_flood_item(self, item: FloodItem) -> FloodSample:
        sample = FloodSample(flood_item=item)
        return sample

    def __getitem__(self, idx_tuple: Tuple[str, int]):
        key, idx = idx_tuple
        return self.items[key][idx]

    def __len__(self):
        return self.n_samples

    def _update_from_split_file(self, split_file: str | Path, split_ratio: float | Dict[str, float]):
        """
        sets: self.split_file
        and calls self._populate_items()
        """
        _sp = Path(split_file).absolute()
        logger.debug(f"split_file: {_sp}")
        logger.debug(f"split_file exists: {_sp.exists()}")
        logger.debug(f"split_file is_file: {_sp.is_file()}")
        logger.debug(f"split_file extention: {_sp.suffix}")
        assert (
            _sp.exists() and _sp.is_file and _sp.suffix == ".json"
        ), f"A split file must be valid json files that exists, got {_sp}"
        self.split_file = _sp
        logger.info(f"{__class__.__name__} init with split file: {self.split_file}")

        self._populate_items(split_ratio=split_ratio)

    def _populate_items(self, split_ratio: float | Dict[str, float]):
        """
        sets:
            - self.split_ratio
            - self.items
            - self.n_sample
        """
        split_dict = {}
        with open(self.split_file, "r") as f:
            split_dict = json.load(f)
        logger.debug(f"keys found in split_file: {list(split_dict.keys())}")

        self.split_ratio = split_ratio if isinstance(split_ratio, dict) else {k: split_ratio for k in split_dict}
        assert np.all(
            [0 <= v <= 1 for k, v in self.split_ratio.items()]
        ), f"Split ratios must lie in the interval [0, 1]"

        possible_typos = [k for k in self.split_ratio if k not in split_dict]
        if possible_typos:
            logger.warning(f"found unused keys in split_ratio: {possible_typos}")
            warnings.warn(f"found unused keys in split_ratio: {possible_typos}", category=RuntimeWarning)

        self.items = {}
        for split, item_list in split_dict.items():
            # requested fraction of total samples
            sr = self.split_ratio.get(split, 1)
            n_items = int(np.ceil(len(item_list) * sr))
            item_list = item_list[:n_items]

            self.items[split] = [self.process_flood_item(FloodItem(**fitem)) for fitem in item_list]
            logger.info(f"loaded {len(self.items[split])} {split} sample with split ratio: {sr:.2f}")
        self.n_samples = sum(len(self.items[k]) for k in self.items)


if __name__ == "__main__":
    generate_split_cli()
