import click
import mlflow

from minidalle2.logging import init_logging
from minidalle2.trainer.repositories.mlflow_repository import MlflowRepository
from minidalle2.trainer.usecases.model import build_clip
from minidalle2.trainer.usecases.train_clip import train_clip
from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.config import ModelType


@click.command()
@click.option("--n-epochs", type=int)
@click.option("--resume/--reset", default=True)
def execute(n_epochs, resume):
    init_logging()

    config = TrainerConfig().load()
    if n_epochs is not None:
        config.n_epochs_clip = n_epochs
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    repo = MlflowRepository(config=config)

    if resume:
        registered_clip = repo.load_model(ModelType.CLIP)
        if not registered_clip:
            clip = build_clip(config)
            trained_steps = 0
        else:
            clip = registered_clip.model
            trained_epochs = registered_clip.trained_epochs
            trained_steps = registered_clip.trained_steps

    else:
        clip = build_clip(config)
        trained_steps = 0

    train_clip(clip, config, repo, trained_epochs=trained_epochs, trained_steps=trained_steps)

    click.echo("Done.")


if __name__ == "__main__":
    execute()
