import click

from minidalle2.logging import init_logging
from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.usecases.import_annotations import import_annotations
from minidalle2.server.values.server_config import ServerConfig


@click.command()
def execute():
    init_logging()

    config = ServerConfig().load()
    repo = AnnotationRepository(config)

    import_annotations(config=config, annotation_repo=repo)

    click.echo("Done.")
