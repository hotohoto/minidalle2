import click

from minidalle2.logging import init_logging
from minidalle2.repositories.mlflow_repository import MlflowRepository
from minidalle2.usecases.model import build_clip
from minidalle2.usecases.train_clip import train_clip
from minidalle2.values.config import ModelType
from minidalle2.values.trainer_config import TrainerConfig


@click.command()
@click.option("--n-epochs", type=int)
@click.option("--resume/--reset", default=True)
def execute(n_epochs, resume):
    init_logging()

    config = TrainerConfig().load()
    if n_epochs:
        config.n_epochs_clip = n_epochs

    repo = MlflowRepository(config=config)

    if resume:
        registered_clip = repo.load_model(ModelType.CLIP)
        if not registered_clip:
            clip = build_clip(config)
            n_steps_to_skip = 0
        else:
            clip = registered_clip.model
            n_steps_to_skip = registered_clip.trained_steps

    else:
        clip = build_clip(config)
        n_steps_to_skip = 0

    train_clip(clip, config, repo, n_steps_to_skip=n_steps_to_skip)

    click.echo("Done.")


if __name__ == "__main__":
    execute()
