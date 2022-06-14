import click

from minidalle2.usecases.model import build_clip
from minidalle2.usecases.repository import Repository
from minidalle2.usecases.train_clip import train_clip
from minidalle2.values.config import ModelType
from minidalle2.values.trainer_config import TrainerConfig


@click.command()
@click.option("--n-epochs", type=int)
@click.option("--resume/--reset", default=True)
def execute(n_epochs, resume):
    config = TrainerConfig().load()
    if n_epochs:
        config.n_epochs_clip = n_epochs

    repo = Repository(config=config)

    if resume:
        clip = repo.load_model(ModelType.CLIP)
        if not clip:
            clip = build_clip(config)

    else:
        clip = build_clip(config)

    train_clip(clip, config, repo)

    click.echo("Done.")


if __name__ == "__main__":
    execute()
