import mlflow
import torch
from dalle2_pytorch import CLIP
from dalle2_pytorch.dalle2_pytorch import (
    freeze_model_and_make_eval_,
    unfreeze_all_layers_,
)
from dalle2_pytorch.tokenizer import tokenizer
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from minidalle2.repositories.mlflow_repository import MlflowRepository, start_run
from minidalle2.values.config import DatasetType, ModelType
from minidalle2.values.custom_remote_dataset import CustomRemoteDataset
from minidalle2.values.sampler import SkippedSampler
from minidalle2.values.trainer_config import TrainerConfig


def train_clip(clip: CLIP, config: TrainerConfig, repo: MlflowRepository, n_steps_to_skip=0):

    device = config.device

    # mock data
    # dummy_text = torch.randint(0, 49408, (4, 256), device=device)
    # dummy_images = torch.randn(4, 3, 256, 256, device=device)

    trainset = CustomRemoteDataset(config, config.get_index_db_path(DatasetType.TRAIN))
    batch_size = 64

    was_eval = not clip.training
    if was_eval:
        unfreeze_all_layers_(clip)
        clip.train()
    with start_run(repo, ModelType.CLIP):
        for epoch in range(config.n_epochs_clip):
            for step, batch in enumerate(
                DataLoader(
                    trainset,
                    batch_sampler=SkippedSampler(
                        BatchSampler(
                            SequentialSampler(trainset),
                            batch_size=batch_size,
                            drop_last=False,
                        ),
                        n_skip=n_steps_to_skip,
                    ),
                )
            ):
                images = batch["image"]
                captions = (
                    torch.stack([tokenizer.tokenize(caption) for caption in batch["caption"]])
                    .squeeze(1)
                    .to(device)
                )
                loss = clip(captions, images, return_loss=True)
                loss.backward()

                if step % 5 == 0:
                    mlflow.log_metric(key="loss", value=loss, step=step)

                    if step % 100 == 0:

                        repo.register_model(
                            model=clip,
                            trained_epochs=epoch,
                            trained_steps=step + 1,
                        )

    if was_eval:
        freeze_model_and_make_eval_(clip)
