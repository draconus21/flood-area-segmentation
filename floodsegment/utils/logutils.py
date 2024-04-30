import os
import json
import click
import logging
import logging.config
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from copy import deepcopy

from floodsegment import PKG_DIR, LOG_CFG, LOG_DIR

logger = logging.getLogger(__name__)


def makeDictJsonReady(dictData: dict):
    def sanitize(value):
        if type(value) in [np.float, np.float16, np.float32, np.float64]:
            return float(value)
        elif isinstance(value, list):
            t = [0] * len(value)
            for i, v in enumerate(value):
                t[i] = sanitize(v)
            return t
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value

    jsonDict = deepcopy(dictData)
    for key, value in dictData.items():
        if isinstance(value, dict):
            jsonDict[key] = makeDictJsonReady(value)
        else:
            jsonDict[key] = sanitize(value)
    return jsonDict


def prettyDumpDict(dictData):
    return json.dumps(makeDictJsonReady(dictData), indent=4, sort_keys=True)


def error(message):
    secho(message, fg="red")
    logging.error(message)


def warn(message):
    secho(message, fg="yellow")
    logging.warning(message)


def info(message):
    secho(message, fg="cyan")
    logging.info(message)


def debug(message):
    secho(message, fg=None)
    logging.debug(message)


def secho(message, fg="cyan"):
    if isinstance(message, str):
        click.secho(message, fg=fg)
    elif isinstance(message, dict):
        click.secho(json.dumps(message, indent=" " * 4), fg=fg)


class bcolors:
    HEADER = "\033[95m"  # DEBUG
    WARNING = "\033[94m"  # WARNING
    OKCYAN = "\033[96m"  # INFO
    OKGREEN = "\033[92m"
    OKYELLOW = "\033[93m"
    FAIL = "\033[91m"  # ERROR/CRITICAL
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def set_color_value(levelname, text=None):
    text = text or levelname
    log_set = {
        "DEBUG": f"{bcolors.OKYELLOW}{text}{bcolors.ENDC}",
        "INFO": f"{bcolors.OKCYAN}{text}{bcolors.ENDC}",
        "WARNING": f"{bcolors.WARNING}{text}{bcolors.ENDC}",
        "ERROR": f"{bcolors.FAIL}{text}{bcolors.ENDC}",
        "CRITICAL": f"{bcolors.FAIL}{text}{bcolors.ENDC}",
        "HEADER": f"{bcolors.HEADER}{text}{bcolors.ENDC}",
    }

    return log_set[levelname]


def setupLogging(console_level: str = "INFO", root_level="INFO", log_cfg: str = "", log_dir: str = ""):
    """
    Setup logging
    """

    # prevent other libs from flooding the logs
    libs = ["matplotlib", "PIL"]
    for lib in libs:
        if logging.getLogger(lib).level < logging.WARNING:
            logging.getLogger(lib).setLevel(logging.WARNING)

    # does not work
    # plt.style.use("seaborn")

    logstr = []
    try:
        if not log_cfg:
            log_cfg = os.path.join(PKG_DIR, LOG_CFG)
        logstr.append(f"logging config: {log_cfg}")

        if not log_dir:
            log_dir = os.path.join(PKG_DIR, LOG_DIR)
        logstr.append(f"logs to be written to {log_dir}")
        console_level = console_level.upper()
        root_level = root_level.upper()

        if Path(log_cfg).exists():
            if not Path(log_dir).exists():
                os.makedirs(log_dir)
            with open(log_cfg, "rt") as f:
                config = json.load(f)

                if logging.getLevelName(console_level) < logging.getLevelName(root_level):
                    logstr.append(f"root logger level [{root_level}] < [{console_level}] console logger level.")
                    logstr.append(f"overriding provided root log level: {root_level} -> {console_level}")
                    root_level = console_level

                config["root"]["level"] = root_level
                logstr.append(f"root log level: {root_level}")

                # extra config for console formatter
                config["handlers"]["console"]["level"] = console_level
                c_fmt = config["formatters"]["console"]["format"]
                c_fmt = c_fmt.replace("%(levelname)s", set_color_value(console_level, "%(levelname)s"))
                c_fmt = c_fmt.replace("%(name)s", set_color_value("HEADER", "%(name)s"))
                config["formatters"]["console"]["format"] = c_fmt
                logstr.append(f"console log level: {console_level}")

                # set log dir
                handlers = config.get("handlers", {})
                for h_name, h in handlers.items():
                    file_name = h.get("filename", "")
                    if file_name:
                        h["filename"] = str(Path(log_dir) / file_name)
                    handlers[h_name] = h
                config["handlers"] = handlers
            logging.config.dictConfig(config)
            logstr.append(f"logging init from provided config: {log_cfg}")
        else:
            logging.basicConfig(level=console_level)
            logstr.append(f"using basicConfig with log level: {console_level}")
    finally:
        for msg in logstr:
            logger.debug(msg)
