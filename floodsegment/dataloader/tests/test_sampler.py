import numpy as np
import pytest
from pathlib import Path
from floodsegment import DATA_DIR, Mode
from floodsegment.dataloader.segment import FloodDataset
from floodsegment.dataloader.sampler import DictSampler


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("split_file", [DATA_DIR / "flood-default-split.json"])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("ratio", list(set([1, *np.random.rand(3)])))
@pytest.mark.parametrize("mode", ["TRAIN", "TEST", "VALID", "invalid"])
def test_DictSample(split_file: str | Path, shuffle: bool, ratio: float, mode: str):
    fd = FloodDataset(split_file=split_file, split_ratio=0.5)

    if mode == "invalid":
        with pytest.raises(ValueError):
            fs = DictSampler(data_source=fd, mode=mode, shuffle=shuffle, ratio=ratio)
        return

    assert mode in fd.items, f"{mode} not found in items: {fd.items.keys()}"

    if Mode(mode) != Mode.TRAIN and shuffle == True:
        with pytest.warns(RuntimeWarning):
            fs = DictSampler(data_source=fd, mode=mode, shuffle=shuffle, ratio=ratio)
    else:
        fs = DictSampler(data_source=fd, mode=mode, shuffle=shuffle, ratio=ratio)
        assert fs.shuffle == shuffle

    idx_list = []
    for key, idx in fs:
        assert key == Mode(mode).value
        idx_list.append(idx)

    if not shuffle:
        assert all(
            [idx_list[i] == idx_list[i - 1] + 1] for i in range(1, len(idx_list))
        ), "Expecting unshuffled items, but they  are shuffled"
    else:
        # NOTE: if this assert fails, try running it again
        assert any(
            [idx_list[i] != idx_list[i - 1] + 1] for i in range(1, len(idx_list))
        ), "Expecting shuffled items, but they  are not shuffled"
