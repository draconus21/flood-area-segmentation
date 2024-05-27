import math
import warnings
from floodsegment import Mode
from floodsegment.dataloader.base import BaseDataset
from torch.utils.data import RandomSampler

from typing import Iterator, Tuple

import logging

logger = logging.getLogger(__name__)


class DictSampler(RandomSampler):
    def __init__(
        # TODO: Create a base Dataset class that uses dicts, and use that for data_source type annotation
        self,
        data_source: BaseDataset,
        mode: str | Mode = Mode.TRAIN,
        ratio: float = 1.0,
        shuffle: bool = True,
        **random_sampler_kwargs,
    ):
        mode = mode.value if isinstance(mode, Mode) else Mode(mode).value
        assert (
            mode in data_source.items
        ), f"{mode} must be a valid key in data_source._items: {data_source.items.keys()}"

        if shuffle and mode != Mode.TRAIN.value:
            warnings.warn(
                f"shuffling is available only in {Mode.TRAIN}. Setting shuffle=False for {mode} mode.",
                category=RuntimeWarning,
            )
            logger.warning(f"shuffling is available only in {Mode.TRAIN}. Setting shuffle=False for {mode} mode.")
            shuffle = False

        self.mode = mode

        self.shuffle = shuffle

        super(__class__, self).__init__(data_source=data_source.items[self.mode], **random_sampler_kwargs)

        self.fraction = int(math.ceil(len(self) * ratio))
        logger.debug(f"sampling {self.fraction}/{len(self)} samples")

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        if self.shuffle:
            _interim_idx_list = list(super().__iter__())[: self.fraction]
        else:
            _interim_idx_list = range(self.fraction)
        yield from [(self.mode, idx) for idx in _interim_idx_list]
