import numpy as np
import pytest
from pathlib import Path
from floodsegment import DATA_DIR
from floodsegment.dataloader.segment import FloodDataset
from floodsegment.dataloader.sampler import DictSampler


@pytest.mark.parametrize("split_file", [DATA_DIR / "flood-default-split.json"])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("ratio", list(set([1, *np.random.rand(3)])))
@pytest.mark.parametrize("mode", ["train", "test", "valid", "invalid"])
def test_DictSample(split_file: str | Path, shuffle: bool, ratio: float, mode: str):
    fd = FloodDataset(split_file=split_file, split_ratio=0.5)

    if mode == "invalid":
        with pytest.raises(AssertionError):
            fs = DictSampler(data_source=fd, mode=mode, shuffle=shuffle, ratio=ratio)
        return

    if mode != "train" and shuffle == True:
        with pytest.warns(RuntimeWarning):
            fs = DictSampler(data_source=fd, mode=mode, shuffle=shuffle, ratio=ratio)
    else:
        fs = DictSampler(data_source=fd, mode=mode, shuffle=shuffle, ratio=ratio)

    idx_list = []
    for key, idx in fs:
        assert key == mode
        idx_list.append(idx)

    if not shuffle:
        assert all([idx_list[i] == idx_list[i - 1] + 1] for i in range(1, len(idx_list)))
    else:
        assert any([idx_list[i] != idx_list[i - 1] + 1] for i in range(1, len(idx_list)))
