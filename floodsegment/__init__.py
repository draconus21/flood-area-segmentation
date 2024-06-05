"""Top-level package for Flood Area Segmentation."""

__author__ = """Neeth Kunnath"""
__email__ = "neeth.xavier@gmail.com"
__version__ = "0.0.1"

# constants
from pathlib import Path

PKG_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = PKG_DIR / "configs"
DATA_DIR = PKG_DIR / "data"
EXP_DIR = PKG_DIR / "experiments"
LOG_DIR = PKG_DIR / "logs"
LOG_CFG = PKG_DIR / "default-logging.json"

CONFIG_VERSION = 1

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from enum import Enum


class Mode(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


del Path
del torch
del Enum
