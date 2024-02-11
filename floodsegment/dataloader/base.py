import json
import warnings
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from torch.utils.data import Dataset

from typing import Dict, List, Tuple

import logging

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        split_item_class,
        split_file: str | Path,
        transform_dict: Dict = {},
        split_ratio=float | Dict[str, float],
    ):
        self.split_item_class = split_item_class

        self.split_file: Path = None
        self.items: Dict[str, List[self.sample_class]] = {}
        self.n_samples: int = 0

        self.split_ratio: Dict[str, float] = {}
        self.transform_dict = transform_dict

        self._update_from_split_file(split_file=split_file, split_ratio=split_ratio)

    def process_split_item(self, item: BaseModel) -> BaseModel:
        return item

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

            self.items[split] = [self.process_split_item(self.split_item_class(**fitem)) for fitem in item_list]

            logger.info(f"loaded {len(self.items[split])} {split} sample with split ratio: {sr:.2f}")

        self.n_samples = sum(len(self.items[k]) for k in self.items)
