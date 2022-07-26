import click

from minidalle2.logging import init_logging
from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.usecases.update_splits import update_splits
from minidalle2.server.values.server_config import ServerConfig


@click.command()
def execute():
    init_logging()

    config = ServerConfig().load()
    repo = AnnotationRepository(config)

    update_splits(annotation_repo=repo)

    click.echo("Done.")


if __name__ == "__main__":
    execute()
