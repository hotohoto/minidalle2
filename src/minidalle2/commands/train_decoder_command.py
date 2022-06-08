import click

from minidalle2.usecases.model import build_decoder
from minidalle2.usecases.repository import Repository
from minidalle2.usecases.train_decoder import train_decoder
from minidalle2.values.trainer_config import TrainerConfig


@click.command()
@click.option("--n-epochs", type=int)
@click.option("--resume/--reset", default=True)
def execute(n_epochs, resume):
    config = TrainerConfig().load()
    if n_epochs:
        config.n_epochs_decoder = n_epochs

    repo = Repository(config=config)

    if resume:
        decoder = repo.load_decoder()
        if not decoder:
            clip = repo.load_clip()
            decoder = build_decoder(config, clip=clip)
    else:
        clip = repo.load_clip()
        decoder = build_decoder(config, clip=clip)

    train_decoder(decoder, config)

    repo.save_decoder(decoder)

    click.echo("Done.")
