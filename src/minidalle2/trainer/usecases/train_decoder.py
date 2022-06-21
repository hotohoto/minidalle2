from dalle2_pytorch import Decoder
from dalle2_pytorch.tokenizer import tokenizer
from torch.utils.data import DataLoader

from minidalle2.trainer.values.custom_remote_dataset import CustomRemoteDataset
from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.datasets import DatasetType


def train_decoder(decoder: Decoder, config: TrainerConfig):

    trainset = CustomRemoteDataset(config, DatasetType.TRAIN)

    for _ in range(config.n_epochs_decoder):
        for batch in DataLoader(trainset, batch_size=64, shuffle=False):
            images = batch["image"]

            for unet_number in (1, 2):
                loss = decoder(
                    images, unet_number=unet_number
                )  # this can optionally be decoder(images, text) if you wish to condition on the text encodings as well, though it was hinted in the paper it didn't do much
                loss.backward()
