"""Top-level package for Flood Area Segmentation."""

__author__ = """Neeth Kunnath"""
__email__ = "neeth.xavier@gmail.com"
__version__ = "0.0.0"

# constants
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
EXP_DIR = ROOT_DIR / "experiments"
LOG_DIR = ROOT_DIR / "logs"
LOG_CFG = ROOT_DIR / "default-logging.json"

from floodsegment.utils.logutils import setupLogging

setupLogging(
    console_level="INFO",
    root_level="INFO",
    log_cfg=LOG_CFG,
    log_dir=LOG_DIR,
)

del setupLogging
del utils
del Path
