import click

from minidalle2.logging import init_logging
from minidalle2.server.usecases.build_index import build_index
from minidalle2.server.values.server_config import ServerConfig


@click.command()
def execute(skip_na_entries=True):
    init_logging()

    config = ServerConfig().load()

    build_index(config=config, skip_na_entries=skip_na_entries)

    click.echo("Done.")
