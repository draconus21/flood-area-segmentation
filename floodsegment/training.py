import click
from floodsegment import DATA_DIR
from floodsegment.utils.logutils import setupLogging


@click.command()
@click.option("-d", "--data-dir", required=True, type=click.Path(), default=DATA_DIR, help="path to root dir")
@click.option(
    "-l",
    "log-level",
    required=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "CRITICAL"], case_sensitive=False),
    default="INFO",
    help="log level",
)
@click.option("-n", "--exp-name", required=True, default="flood-exp", help="Experiment name")
@click.option("-ne", "--n-epochs", required=True, default=100, help="Number of epochs")
def train_cli(data_dir, log_level, exp_name, n_epocsh):
    setupLogging(log_level)
