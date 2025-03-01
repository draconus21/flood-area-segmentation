"""Top-level package for Flood Area Segmentation."""

__author__ = """Neeth Kunnath"""
__email__ = "neeth.xavier@gmail.com"
__version__ = "0.0.1"

# constants
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
EXP_DIR = ROOT_DIR / "experiments"
LOG_DIR = ROOT_DIR / "logs"
LOG_CFG = ROOT_DIR / "default-logging.json"

from enum import Enum


class Mode(Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


del Path
