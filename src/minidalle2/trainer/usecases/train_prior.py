import torch
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.tokenizer import tokenizer
from torch.utils.data import DataLoader

from minidalle2.trainer.values.custom_remote_dataset import CustomRemoteDataset
from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.datasets import DatasetType


def train_prior(prior: DiffusionPrior, config: TrainerConfig):

    device = config.device

    trainset = CustomRemoteDataset(config, DatasetType.TRAIN)

    for _ in range(config.n_epochs_prior):
        for batch in DataLoader(trainset, batch_size=64, shuffle=False):
            images = batch["image"]
            captions = (
                torch.stack([tokenizer.tokenize(caption) for caption in batch["caption"]])
                .squeeze(1)
                .to(device)
            )

            loss = prior(captions, images)
            loss.backward()
