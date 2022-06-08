from pathlib import Path

import click
from torchvision.utils import save_image

from minidalle2.usecases.infer import infer
from minidalle2.usecases.repository import Repository, load_existing_model
from minidalle2.values.trainer_config import TrainerConfig


@click.Argument("output_image_path", nargs=1, type=click.Path(exists=False))
@click.Argument("input_text", nargs=1)
@click.command
def execute(output_image_path: Path, input_text: str):
    config = TrainerConfig().load()

    repo = Repository(config)

    dalle2 = repo.load_existing_model()
    image_data = infer(dalle2, input_text)
    save_image(image_data, output_image_path)

    click.echo("Done.")
