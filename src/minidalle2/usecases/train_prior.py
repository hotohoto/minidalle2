import torch
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import (
    freeze_model_and_make_eval_,
    unfreeze_all_layers_,
)
from dalle2_pytorch.tokenizer import tokenizer
from torch.utils.data import DataLoader

from minidalle2.values.config import DatasetType
from minidalle2.values.custom_remote_dataset import CustomRemoteDataset
from minidalle2.values.trainer_config import TrainerConfig


def train_prior(prior: DiffusionPrior, config: TrainerConfig):

    device = config.device

    trainset = CustomRemoteDataset(config, config.get_index_db_path(DatasetType.TRAIN))

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
