import torch
from dalle2_pytorch import CLIP
from dalle2_pytorch.dalle2_pytorch import (
    freeze_model_and_make_eval_,
    unfreeze_all_layers_,
)
from dalle2_pytorch.tokenizer import tokenizer
from torch.utils.data import DataLoader

from minidalle2.values.config import DatasetType
from minidalle2.values.custom_remote_dataset import CustomRemoteDataset
from minidalle2.values.trainer_config import TrainerConfig


def train_clip(clip: CLIP, config: TrainerConfig):

    device = config.device

    # mock data
    # dummy_text = torch.randint(0, 49408, (4, 256), device=device)
    # dummy_images = torch.randn(4, 3, 256, 256, device=device)

    trainset = CustomRemoteDataset(config, config.get_index_db_path(DatasetType.TRAIN))

    was_eval = not clip.training
    if was_eval:
        unfreeze_all_layers_(clip)
        clip.train()
    for _ in range(config.n_epochs_clip):
        for batch in DataLoader(trainset, batch_size=64, shuffle=True):
            images = batch["image"]
            captions = torch.stack(
                [tokenizer.tokenize(caption) for caption in batch["caption"]]
            ).squeeze(1).to(device)
            loss = clip(captions, images, return_loss=True)
            loss.backward()
    if was_eval:
        freeze_model_and_make_eval_(clip)
