import click
import torch
from pathlib import Path
from logging import getLogger

from torch.utils.tensorboard.writer import SummaryWriter

from floodsegment import DATA_DIR, Mode
from floodsegment.dataloader.segment import FloodSample
from floodsegment.utils.builder import load_train_config, construct_dataset, construct_model, TrainConfig
from floodsegment.utils.logutils import setupLogging


from typing import Dict, List

logger = getLogger(__name__)

GEN_PLT = "general"
DATA_PLT = "data"
TRAIN_PLOTTER_NAMES = [GEN_PLT, DATA_PLT]
TRAIN_PLOTTER_NAMES = TRAIN_PLOTTER_NAMES + [m.value for m in Mode]
TRAIN_PLOTTER_NAMES = TRAIN_PLOTTER_NAMES + [f"{m.value}_fixed" for m in Mode]


def prep_from_config(train_config: TrainConfig, plotters: Dict[str, SummaryWriter]):
    # get dataset
    dataset = construct_dataset(dataset_config_path=train_config.dataset)
    logger.debug("created dataset")
    sample: FloodSample = dataset[Mode.TRAIN, 0]
    img_size = sample.image.shape
    dataset.visualize(sample, plotters[DATA_PLT])
    logger.debug("added sample data to tboard")

    # get model
    model = construct_model(model_config_path=train_config.model)
    # TODO: Load from checkpoint (in construct_model)

    plotters[GEN_PLT].add_graph(model=model, input_to_model=model.net.get_dummy_inputs())


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
@click.option("-ne", "--n-epochs", required=True, default=100, help="Number of epochs")
def train_cli(config, exp_dir, log_level, exp_name, n_epochs):
    train(config, exp_dir, log_level, exp_name, n_epochs)


def train(config, exp_dir, log_level, exp_name, n_epochs):
    setupLogging(log_level.upper())

    # helpful setting for debugging
    if log_level.upper() == "DEBUG":
        logger.debug("enabling torch.autograd.set_detect_anomaly")
        torch.autograd.set_detect_anomaly(True)

    logger.debug(f"loading training config: {config}")
    train_config = load_train_config(config)

    # tensorboard
    tboard_dir = (Path(exp_dir) / train_config.name / exp_name).resolve()
    plotters = setup_tboard(tboard_dir)

    _ = prep_from_config(train_config, plotters)


def setup_tboard(tboard_dir: str, plotter_names: List[str] = []) -> Dict[str, SummaryWriter]:
    logger.info(f"tensorboard dir: {tboard_dir}")

    plotter_names.extend(TRAIN_PLOTTER_NAMES)

    plotters = {k: SummaryWriter(str(Path(tboard_dir) / k)) for k in plotter_names}
    logger.info(f"plotters: {plotters.keys()}")
    return plotters


if __name__ == "__main__":
    train_cli()
