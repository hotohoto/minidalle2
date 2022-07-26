import click

from minidalle2.logging import init_logging
from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.usecases.download_images import download_images
from minidalle2.server.values.server_config import ServerConfig


@click.command()
def execute(retry_failed=False):
    init_logging()

    config = ServerConfig().load()
    repo = AnnotationRepository(config)

    download_images(config=config, annotation_repo=repo, retry_failed=retry_failed)

    click.echo("Done.")


if __name__ == "__main__":
    execute()
