import time
import click
import torch
from pathlib import Path
from logging import getLogger
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from floodsegment import DATA_DIR, Mode, DEVICE
from floodsegment.dataloader.segment import FloodSample
from floodsegment.utils.builder import (
    TrainConfig,
    TrainSetup,
    load_train_config,
    construct_dataset,
    construct_model,
    construct_sampler,
    constrcut_optimizer,
    constrcut_scheduler,
    construct_criterion,
)
from floodsegment.utils.logutils import setupLogging


from typing import Dict, List

logger = getLogger(__name__)

GEN_PLT = "general"
DATA_PLT = "data"
TRAIN_PLOTTER_NAMES = [GEN_PLT, DATA_PLT]
TRAIN_PLOTTER_NAMES = TRAIN_PLOTTER_NAMES + [m.value for m in Mode]
TRAIN_PLOTTER_NAMES = TRAIN_PLOTTER_NAMES + [f"{m.value}_fixed" for m in Mode]

BEST_PARAMS_PATH = "best_model_params.pt"


def prep_from_config(train_config: TrainConfig, plotters: Dict[str, SummaryWriter]) -> TrainSetup:
    # get dataset
    dataset = construct_dataset(train_config.dataset)
    logger.debug(f"Created dataset from {train_config.dataset}")
    sample: Dict = dataset[Mode.TRAIN, 0]
    dataset.visualize(sample, plotters[DATA_PLT])
    logger.debug("Added sample data to tboard")

    # get samplers
    samplers = construct_sampler(train_config.samplers, dataset=dataset)
    logger.info(f"Loaded samplers: {[k for k in samplers]} from {train_config.samplers}")

    # get dataloaders
    dataloaders = {
        x: DataLoader(
            dataset,
            sampler=samplers[x],
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            pin_memory=False,
            drop_last=True,
        )
        for x in samplers
    }

    # get model
    model = construct_model(train_config.model)
    # TODO: Load from checkpoint (in construct_model)

    plotters[GEN_PLT].add_graph(model=model, input_to_model=model.net.get_dummy_inputs())
    logger.info(f"Loaded model from {train_config.model}")

    # get optimizer
    optimizer = constrcut_optimizer(train_config.optimizer, model=model)
    logger.info(f"Loaded optimzier from {train_config.optimizer}")

    # get scheduler
    scheduler = constrcut_scheduler(train_config.scheduler, optimizer=optimizer)
    logger.info(f"Loaded scheduler from {train_config.scheduler}")

    # get criterion
    criterion = construct_criterion(train_config.criterion)
    logger.info(f"Loaded criterion from {train_config.criterion}")

    return TrainSetup(
        version=train_config.version,
        name=train_config.name,
        dataset=dataset,
        dataloaders=dataloaders,
        samplers=samplers,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )


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
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    required=True,
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default=DEVICE,
    help="device",
    show_default=True,
)
@click.option(
    "-l",
    "--log-level",
    required=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "CRITICAL"], case_sensitive=False),
    default="INFO",
    help="log level",
    show_default=True,
)
@click.option("-n", "--exp-name", required=True, default="flood-exp", help="Experiment name")
@click.option("-ne", "--n-epochs", required=True, default=100, help="Number of epochs")
def train_cli(config, exp_dir, log_level, exp_name, n_epochs, device):
    train(config, exp_dir, log_level, exp_name, n_epochs, device)


def train(config, exp_dir, log_level, exp_name, n_epochs, device):
    setupLogging(log_level.upper())

    # helpful setting for debugging
    if log_level.upper() == "DEBUG":
        logger.debug("enabling torch.autograd.set_detect_anomaly")
        torch.autograd.set_detect_anomaly(True)

    assert device == "cpu" or torch.cuda.is_available(), f"No cuda device found, please use device=cpu"

    logger.debug(f"loading training config: {config}")
    train_config = load_train_config(config)

    # tensorboard
    tboard_dir = (Path(exp_dir) / train_config.name / exp_name).resolve()
    plotters = setup_tboard(tboard_dir)

    train_setup = prep_from_config(train_config, plotters)
    train_setup.model.to(device)

    ## training

    if device == "cuda":
        logger.info(f"Using {torch.cuda.get_device_name(0)}")
        logger.info(f"Cached {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")
        logger.info(f"Allocated {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")

    loss = _train(train_setup, plotters, exp_dir, n_epochs, device)


def _train(
    train_setup: TrainSetup, plotters: Dict[str, SummaryWriter], exp_dir: str, n_epochs: int, device: str
) -> float:
    since = time.time()
    with TemporaryDirectory(prefix=train_setup.name, dir=exp_dir) as tempdir:
        best_model_params_path = (Path(tempdir) / BEST_PARAMS_PATH).resolve()
        best_loss = float("inf")

        model = train_setup.model
        dataloaders = train_setup.dataloaders

        def _save_best():
            logger.debug(f"best params [loss: {best_loss}] saved to: {best_model_params_path}")
            torch.save(model.state_dict(), best_model_params_path)

        _save_best()

        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch}/{n_epochs-1}")

            for phase in dataloaders:
                _phase = Mode(phase)
                _dataloader = dataloaders[_phase.value]
                kwargs = {
                    "criterion": train_setup.criterion,
                    "plotters": plotters,
                    "device": device,
                    "sample_viz": train_setup.dataset.visualize,
                }
                if _phase == Mode.TRAIN:
                    step_func = model.train_step
                    kwargs["optimizer"] = train_setup.optimizer
                elif _phase == Mode.VALID:
                    step_func = model.valid_step
                else:
                    raise NotImplementedError(f"No step function implemeted for phase: {phase}")

                running_loss = 0.0

                for step, sample in enumerate(_dataloader):
                    kwargs["global_step"] = epoch * step
                    loss, outputs = step_func(sample, **kwargs)
                    running_loss += loss.item() * _dataloader.batch_size

                if _phase == Mode.TRAIN:
                    train_setup.scheduler.step(running_loss)

                epoch_loss = running_loss / len(_dataloader)
                logger.info(f"{_phase} loss: {epoch_loss}")

    time_elapsed = time.time() - since
    logger.info(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")


def setup_tboard(tboard_dir: str, plotter_names: List[str] = []) -> Dict[str, SummaryWriter]:
    logger.info(f"tensorboard dir: {tboard_dir}")

    plotter_names.extend(TRAIN_PLOTTER_NAMES)

    plotters = {k: SummaryWriter(str(Path(tboard_dir) / k)) for k in plotter_names}
    logger.info(f"plotters: {plotters.keys()}")

    # TODO: add profiler
    return plotters


if __name__ == "__main__":
    train_cli()
