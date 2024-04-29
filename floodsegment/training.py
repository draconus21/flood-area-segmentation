import click
import torch
from pathlib import Path
from logging import getLogger

from torch.utils.tensorboard.writer import SummaryWriter

from floodsegment import DATA_DIR, Mode
from floodsegment.utils.builder import load_train_config, construct_dataset
from floodsegment.utils.logutils import setupLogging


from typing import Dict, List

logger = getLogger(__name__)


@click.command()
@click.option(
    "-c",
    "--config",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="path to training yaml",
)
@click.option(
    "-e",
    "--exp-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    default=Path(DATA_DIR) / "experiments",
    help="path to experiment dir",
)
@click.option(
    "-l",
    "--log-level",
    required=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "CRITICAL"], case_sensitive=False),
    default="INFO",
    help="log level",
)
@click.option("-n", "--exp-name", required=True, default="flood-exp", help="Experiment name")
@click.option("-p", "--port", required=True, default=8090, help="tensorboard port")
@click.option("-ne", "--n-epochs", required=True, default=100, help="Number of epochs")
def train_cli(config, exp_dir, log_level, exp_name, port, n_epochs):
    train(config, exp_dir, log_level, exp_name, port, n_epochs)


def train(config, exp_dir, log_level, exp_name, port, n_epochs):
    setupLogging(log_level)

    # helpful setting for debugging
    if log_level.upper() == "DEBUG":
        logger.debug("enabling torch.autograd.set_detect_anomaly")
        torch.autograd.set_detect_anomaly(True)

    logger.debug(f"loading training config: {config}")
    train_config = load_train_config(config)

    tboard_dir = Path(exp_dir) / train_config.name / exp_name

    plotters = setup_tboard(tboard_dir, port)

    # get dataset
    dataset = construct_dataset(dataset_config_path=train_config.dataset)

    dataset.visualize(dataset[Mode.TRAIN, 0], plotters["data"])


def setup_tboard(tboard_dir: str, port: int, plotter_names: List[str] = ["data"]) -> Dict[str, SummaryWriter]:
    logger.info(f"tensorboard dir: {tboard_dir}")

    plotters = {k: SummaryWriter(str(Path(tboard_dir) / k)) for k in plotter_names}
    logger.info(f"plotters: {plotters.keys()}")
    return plotters


if __name__ == "__main__":
    train_cli()
