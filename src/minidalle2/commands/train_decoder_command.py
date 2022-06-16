import click

from minidalle2.logging import init_logging
from minidalle2.repositories.mlflow_repository import MlflowRepository
from minidalle2.usecases.model import build_decoder
from minidalle2.usecases.train_decoder import train_decoder
from minidalle2.values.config import ModelType
from minidalle2.values.trainer_config import TrainerConfig


@click.command()
@click.option("--n-epochs", type=int)
@click.option("--resume/--reset", default=True)
def execute(n_epochs, resume):
    init_logging()

    config = TrainerConfig().load()
    if n_epochs:
        config.n_epochs_decoder = n_epochs

    repo = MlflowRepository(config=config)

    if resume:
        decoder = repo.load_model(ModelType.DECODER)
        if not decoder:
            clip = repo.load_model(ModelType.CLIP)
            decoder = build_decoder(config, clip=clip)
    else:
        clip = repo.load_model(ModelType.CLIP)
        decoder = build_decoder(config, clip=clip)

    train_decoder(decoder, config, repo)

    click.echo("Done.")
