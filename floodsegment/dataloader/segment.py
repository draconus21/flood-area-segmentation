import numpy as np
import logging
import imageio.v3 as iio
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


def load_img(im_path: str) -> np.ndarray:
    return np.array(iio.imread(im_path))


class FloodItem(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)  # , arbitrary_types_allowed=True)

    image: Path
    mask: Path

    @field_validator("image", "mask", mode="before")
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
