import yaml
from pathlib import Path
from floodsegment.utils.misc import normalize_path

from typing import Any, Dict, List


def flatten(_dict: Dict, /) -> Dict:
    def _flatten(_dict: Dict, root: str = "") -> Dict:
        flat_dict = {}
        for k, v in _dict.items():
            new_k = f"{root}.{k}" if root else k
            if isinstance(v, dict):
                flat_dict = {**flat_dict, **_flatten(v, root=new_k)}
            else:
                flat_dict[new_k] = v
        return flat_dict

    return _flatten(_dict)


def add_flat_key(_dict: Dict, *, flat_key: str, val: Any):
    sub_keys = flat_key.split(".", 1)
    if len(sub_keys) == 1:
        _dict[sub_keys[0]] = val
        return

    sub_dict = _dict.setdefault(sub_keys[0], {})
    add_flat_key(sub_dict, flat_key=sub_keys[1], val=val)


def get_flat_key(_dict: Dict, *, flat_key: str, default_value: Any = None) -> Any:
    sub_keys = flat_key.split(".", 1)
    if not sub_keys[0] in _dict:
        return default_value

    if len(sub_keys) == 1:
        return _dict.get(sub_keys[0], default_value)

    return get_flat_key(_dict[sub_keys[0]], flat_key=sub_keys[1])


def unflatten(flat_dict: Dict, /) -> Dict:
    unflat_dict = {}
    for k, v in flat_dict.items():
        add_flat_key(unflat_dict, flat_key=k, val=v)
    return unflat_dict


def load_yaml(yaml_path: str, recursive_load_keys: List[str] = []) -> Dict:
    ydict = {}
    with open(yaml_path, "r") as y:
        ydict = yaml.full_load(y)
    assert ydict is not None

    if "_parent_" in ydict:
        pdict = load_yaml(ydict.pop("_parent_"))
        ydict = {**pdict, **ydict}

    for k in recursive_load_keys:
        val = get_flat_key(ydict, flat_key=k)

        if val and isinstance(val, str):
            _p = normalize_path(Path(yaml_path).parent / val, is_file=True, ext=".yaml")
            _recursive_load_keys = [key.replace(f"{k}.", "") for key in recursive_load_keys if k in key]
            ydict[k] = load_yaml(_p, recursive_load_keys=_recursive_load_keys)

    return ydict
