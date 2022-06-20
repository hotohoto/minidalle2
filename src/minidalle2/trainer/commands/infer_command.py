from pathlib import Path

import click
from torchvision.utils import save_image

from minidalle2.logging import init_logging
from minidalle2.trainer.repositories.mlflow_repository import MlflowRepository
from minidalle2.trainer.usecases.infer import infer
from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.config import ModelType


@click.Argument("output_image_path", nargs=1, type=click.Path(exists=False))
@click.Argument("input_text", nargs=1)
@click.command
def execute(output_image_path: Path, input_text: str):
    init_logging()

    config = TrainerConfig().load()
    repo = MlflowRepository(config)
    dalle2 = repo.load_model(ModelType.DALLE2)
    image_data = infer(dalle2, input_text)
    save_image(image_data, output_image_path)

    click.echo("Done.")
