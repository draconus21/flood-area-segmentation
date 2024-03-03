"""Top-level package for Flood Area Segmentation."""

__author__ = """Neeth Kunnath"""
__email__ = "neeth.xavier@gmail.com"
__version__ = "0.0.1"

# constants
from pathlib import Path

PKG_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = PKG_DIR / "data"
EXP_DIR = PKG_DIR / "experiments"
LOG_DIR = PKG_DIR / "logs"
LOG_CFG = PKG_DIR / "default-logging.json"

from enum import Enum


class Mode(Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


del Path
