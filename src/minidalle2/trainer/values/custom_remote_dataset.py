import typing as t

import requests
from PIL import Image
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.datasets import DatasetType


class CustomRemoteDataset(Dataset):
    def __init__(self, config: TrainerConfig, dataset_type: DatasetType):
        super().__init__()

        self.config = config
        self.dataset_type = dataset_type
        self.pil_image_to_tensor = transforms.ToTensor()
        self._length = None

    def __getitem__(self, index) -> t.Dict[Tensor, str]:
        rowid = index + 1
        data_url = self.config.get_data_url(rowid=rowid, dataset_type=self.dataset_type)
        response = requests.get(data_url).json()
        image_url = self.config.get_image_url(rowid=rowid, dataset_type=self.dataset_type)
        local_image_path = self.config.get_image_path(response["subreddit"], response["image_id"])

        if not local_image_path.exists():
            respnose = requests.get(image_url)
            local_image_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_image_path, "wb") as f:
                f.write(respnose.content)

        return dict(
            image=self.pil_image_to_tensor(Image.open(local_image_path)),
            caption=response["caption"],
        )

    def __len__(self):
        if self._length is None:
            length_url = self.config.get_data_length_url(self.dataset_type)
            self._length = requests.get(length_url).json()

        return self._length
