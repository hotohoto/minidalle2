import click

from minidalle2.logging import init_logging
from minidalle2.usecases.download_index import download_index
from minidalle2.values.trainer_config import TrainerConfig


@click.command()
def execute():
    init_logging()

    config = TrainerConfig().load()

    download_index(config)

    click.echo("Done.")
