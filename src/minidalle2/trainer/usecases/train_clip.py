import logging
import math

import mlflow
import torch
from dalle2_pytorch import CLIP
from dalle2_pytorch.dalle2_pytorch import (
    freeze_model_and_make_eval_,
    unfreeze_all_layers_,
)
from dalle2_pytorch.tokenizer import tokenizer
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from tqdm import tqdm

from minidalle2.trainer.repositories.mlflow_repository import (
    MlflowRepository,
    start_run,
)
from minidalle2.trainer.values.custom_remote_dataset import CustomRemoteDataset
from minidalle2.trainer.values.sampler import SkippedSampler
from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.config import ModelType
from minidalle2.values.datasets import DatasetType

_LOGGER = logging.getLogger(__name__)


def train_clip(
    clip: CLIP, config: TrainerConfig, repo: MlflowRepository, trained_epochs=0, trained_steps=0
):

    device = config.device

    # mock data
    # dummy_text = torch.randint(0, 49408, (4, 256), device=device)
    # dummy_images = torch.randn(4, 3, 256, 256, device=device)

    trainset = CustomRemoteDataset(config, DatasetType.TRAIN)
    batch_size = 64
    total_steps_per_epoch = math.ceil(len(trainset) / batch_size)

    was_eval = not clip.training
    if was_eval:
        unfreeze_all_layers_(clip)
        clip.train()
    with start_run(repo, ModelType.CLIP):
        for epoch in range(trained_epochs, config.n_epochs_clip):
            dataloader = DataLoader(
                trainset,
                batch_sampler=SkippedSampler(
                    BatchSampler(
                        SequentialSampler(trainset),
                        batch_size=batch_size,
                        drop_last=False,
                    ),
                    n_skip=trained_steps,
                ),
                num_workers=config.n_dataloader_workers,
            )

            total_relative_steps = total_steps_per_epoch - trained_steps

            for relative_step, batch in tqdm(
                enumerate(dataloader), total=total_relative_steps, desc=f"epoch={epoch}"
            ):
                step = relative_step + trained_steps
                _LOGGER.debug(f"{epoch}:{step}")

                images = batch["image"]
                captions = (
                    torch.stack([tokenizer.tokenize(caption) for caption in batch["caption"]])
                    .squeeze(1)
                    .to(device)
                )
                _LOGGER.debug(f"{epoch}:{step} forward")
                loss = clip(captions, images, return_loss=True)
                _LOGGER.debug(f"{epoch}:{step} backward")
                loss.backward()

                mlflow.log_metric(key="loss", value=loss, step=step)

                trained_10_steps = step % 10 == 10 - 1
                is_last_step = relative_step == total_relative_steps - 1

                if trained_10_steps or is_last_step:
                    repo.register_model(
                        model=clip,
                        trained_epochs=epoch,
                        trained_steps=step + 1,
                    )

            trained_steps = 0

    if was_eval:
        freeze_model_and_make_eval_(clip)
