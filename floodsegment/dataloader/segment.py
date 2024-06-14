import click
import csv
import json
import random
from torch import Tensor
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


from typing import Dict, List, Optional

from floodsegment import DATA_DIR, Mode
from floodsegment.dataloader.base import BaseDataset
from floodsegment.utils.logutils import setupLogging

from ..utils.tensors import tensor_to_numpy
from floodsegment.utils.viz import quickmatshow_dict


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
        splits={Mode("TRAIN"): train_split, Mode("VALID"): valid_split, Mode("TEST"): test_split},
    )


def generate_split(
    split_file_name: str | Path,
    data_dir: str | Path,
    splits: Dict[Mode, float],
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

    m_data: List[Dict] = []
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
    for mode, split_len in sizes.items():
        split_name = mode.value
        end_idx = cur_idx + split_len
        split_json[split_name] = m_data[cur_idx:end_idx]
        cur_idx = end_idx

    with open(_split_file_name, "w") as s_file:
        json.dump(split_json, s_file, indent=4)

    logger.info(f"split file written to {_split_file_name}")


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

    image: np.ndarray | Tensor
    mask: np.ndarray | Tensor

    @model_validator(mode="before")
    @classmethod
    def m_before(cls, data: Dict):
        if "flood_item" not in data:
            return data

        fitem = data.pop("flood_item")
        assert isinstance(fitem, FloodItem)

        _data = {"image": load_img(fitem.image), "mask": load_img(fitem.mask)}

        return _data


class FloodDataset(BaseDataset):
    def __init__(
        self,
        split_file: str | Path,
        transform_dict: Dict = {},
        split_ratio=float | Dict[str, float],
    ):
        super(__class__, self).__init__(
            split_item_class=FloodItem,
            split_file=split_file,
            transform_dict=transform_dict,
            split_ratio=split_ratio,
        )

    def process_split_item(self, item: FloodItem) -> Dict:
        sample = FloodSample(flood_item=item)
        return sample.model_dump(mode="python")

    def visualize(
        self,
        sample: Dict | FloodSample,
        plotter: SummaryWriter,
        *,
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None,
        **kwargs,
    ):
        """
        Visualize a flood sample
        """
        assert isinstance(sample, dict) or isinstance(
            sample, FloodSample
        ), f"sample must be either a dict or a FloodSample, got {type(sample)}"

        _sample = sample.model_dump(mode="python") if isinstance(sample, FloodSample) else sample
        _sample_dict = {k: tensor_to_numpy(v, channels_last=True) for k, v in _sample.items()}

        plotter.add_figure(
            tag="FloodItem",
            figure=quickmatshow_dict(_sample_dict, title="Flood Sample", **kwargs),
            global_step=global_step,
            close=close,
            walltime=walltime,
        )


if __name__ == "__main__":
    generate_split_cli()
