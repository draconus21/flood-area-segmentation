import re
import os
import json
import shutil
import logging
from glob import glob


from copy import deepcopy
from pathlib import Path
from configparser import ConfigParser

from floodsegment.utils import logutils

from typing import List


def userInput(prompt, castFunc=None):
    var = None
    if castFunc is None:
        # create dummy identity method
        castFunc = lambda a: a
    while var is None:
        try:
            var = castFunc(input(f"{prompt}: "))
        except Exception as e:
            logutils.error(e)
            pass
    return var


def mkdirs(fname, deleteIfExists=False):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    """
    p = Path(fname)
    if p.is_file():
        p = p.parent
    if p.exists():
        if deleteIfExists:
            confirm = input(f"{p} exists. Do you want to delete it and create an empty directory? (Y/N)")
            if confirm in ["Y", "y"]:
                shutil.rmtree(p)
                logutils.info(f"Deleted directory {p}")
                mkdirs(fname=p, deleteIfExists=deleteIfExists)
        else:
            return True

    logging.debug(f"\ncreating dir: {p}")
    os.makedirs(p, exist_ok=True)


def get_files_in_dir(search_dir: str | Path, match_pattern: str, **iglob_kwargs) -> List[str]:
    return [
        str((Path(search_dir) / fname).resolve())
        for fname in sorted(glob(match_pattern, root_dir=search_dir, **iglob_kwargs))
    ]


def getValue(string, regEx, key):
    """
    get substring corresponding to  group `key` from `string` that matches `regEx`
    Parameters
    ----------
    string: string
    regEx: regular expression instance of re
        Regular expression to match.
    key: string
        Key in regEx to search for.

    Returns
    -------
    result: string

    """

    if not regEx.match(string):
        msg = f"No match found.\nregEx: {regEx.pattern}\nstring: {string}"
        raise ValueError(msg)
    return regEx.match(string).group(key)


def getParent(fname):
    _path = Path(fname)

    parent = _path.parent.resolve()  # get abs path
    return parent


def mergeDicts(dict1: dict, dict2: dict) -> dict:
    """
    This merges `dict2` into `dict1` recursively. It will overwrite leaves that are in both `dict1` and `dict2` with those in `dict2`.
    """

    mergedDict = deepcopy(dict1)

    for key in dict2:
        if isinstance(dict2[key], dict):
            if key not in mergedDict:
                mergedDict[key] = {}
            mergedDict[key] = mergeDicts(mergedDict[key], dict2[key])
        else:
            mergedDict[key] = dict2[key]
    return mergedDict


def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    try:
        options = dict(cfg["options"])
    except KeyError:
        options = {}
    ctx.default_map = options


def cliToDict(ctx, param, dictStr):
    try:
        jdict = json.loads(dictStr)
    except Exception as e:
        logging.error(f"Caught {e} while processing for {param}\n{dictStr}")
        raise e
    return jdict


def get_dir_from_env(key):
    d = os.environ.get(key, os.curdir)
    logutils.debug(f"Fetched {d} for {key}")
    return d
