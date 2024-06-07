import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from floodsegment import Mode
from floodsegment.dataloader.transforms import init_transforms
from floodsegment.utils.builder import BaseModel

from typing import OrderedDict, Dict, List, Tuple, Type


import logging

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        split_item_class: Type[BaseModel],
        split_file: str | Path,
        split_ratio: float | Dict[str, float],
        transform_dict: OrderedDict = OrderedDict(),
    ):
        self.split_item_class = split_item_class

        self.split_file: Path = Path()  # will be assined in _update_from_split_file
        self.items: Dict[str, List[split_item_class]] = {}
        self.n_samples: int = 0

        self.split_ratio: Dict[Mode, float] = {}
        self.transform_dict = init_transforms(transform_dict)

        self._update_from_split_file(split_file=split_file, split_ratio=split_ratio)

    def visualize(self, sample: BaseModel, plotter: SummaryWriter, **kwargs):
        """
        Visualize a sample in BaseDataset
        """
        pass

    def process_split_item(self, item: BaseModel) -> Dict:
        return item.model_dump(mode="python")

    def __getitem__(self, idx_tuple: Tuple[str, int]):
        key, idx = idx_tuple
        assert Mode(key), f"{key} must be a valide Mode"
        key = Mode(key).value

        _item = self.process_split_item(self.items[key][idx])

        # apply transforms
        for k, t in self.transform_dict.items():
            _item = t(_item)
            logger.debug(f"applied {k}", extra={'limit": 1'})
        return _item

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
            _dict = json.load(f)
            for k in _dict:
                split_dict[Mode(k).value] = _dict[k]
        logger.debug(f"keys found in split_file: {list(split_dict.keys())}")

        _split_ratio = split_ratio if isinstance(split_ratio, dict) else {k: split_ratio for k in split_dict}
        self.split_ratio = {Mode(k): v for k, v in _split_ratio.items()}
        assert np.all(
            [0 <= v <= 1 for k, v in self.split_ratio.items()]
        ), f"Split ratios must lie in the interval [0, 1]"

        self.items = {}
        for mode, item_list in split_dict.items():
            # requested fraction of total samples
            sr = self.split_ratio.get(mode, 1)
            n_items = int(np.ceil(len(item_list) * sr))
            item_list = item_list[:n_items]

            self.items[mode] = [self.split_item_class(**fitem) for fitem in item_list]

            logger.info(f"loaded {len(self.items[mode])} {mode} sample with split ratio: {sr:.2f}")

        self.n_samples = sum(len(self.items[k]) for k in self.items)
