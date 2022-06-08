import click

from minidalle2.usecases.build_index import build_index
from minidalle2.values.server_config import ServerConfig


@click.command()
def execute(skip_na_entries=True):
    config = ServerConfig().load()

    build_index(config=config, skip_na_entries=skip_na_entries)

    click.echo("Done.")
