import click

from minidalle2.usecases.model import build_prior
from minidalle2.usecases.repository import Repository
from minidalle2.usecases.train_prior import train_prior
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
        prior = repo.load_prior()
        if not prior:
            clip = repo.load_clip()
            prior = build_prior(config, clip=clip)
    else:
        clip = repo.load_clip()
        prior = build_prior(config, clip=clip)

    train_prior(prior, config)

    repo.save_prior(prior)

    click.echo("Done.")
