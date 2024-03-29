import pytest
from floodsegment.utils.yaml import flatten, unflatten, get_flat_key, add_flat_key

from typing import Dict


def unflat_dict():
    return {
        "A": {
            "B": {"C": 123, "D": 432, "F": "xyz"},
            "C": "cat",
            "D": {"E": "eee", "F": "eff"},
        },
        "X": "hello",
    }


def flat_dict():
    return {"A.B.C": 123, "A.B.D": 432, "A.B.F": "xyz", "A.C": "cat", "A.D.E": "eee", "A.D.F": "eff", "X": "hello"}


def dict_assert(d1: Dict, d2: Dict):
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        assert type(d1[k]) == type(d2[k])
        if isinstance(d1[k], dict):
            dict_assert(d1[k], d2[k])
        else:
            assert d1[k] == d2[k]


def test_flatten():
    dict_assert(flatten(unflat_dict()), flat_dict())


def test_unflatten():
    dict_assert(unflatten(flat_dict()), unflat_dict())


def test_misc():
    dict_assert(flatten(unflatten(flat_dict())), flatten(unflat_dict()))
    dict_assert(unflatten(flat_dict()), unflatten(flatten(unflat_dict())))


def test_get_add():
    my_dict = unflat_dict()
    add_flat_key(my_dict, flat_key="Y", val="game")
    assert get_flat_key(my_dict, flat_key="Y") == "game"
    assert my_dict["Y"] == "game"

    with pytest.raises(TypeError):
        add_flat_key(my_dict, flat_key="X.Y", val="yes")

    add_flat_key(my_dict, flat_key="U.T", val="yes")
    assert get_flat_key(my_dict, flat_key="U.T") == "yes"
    assert my_dict["U"]["T"] == "yes"
