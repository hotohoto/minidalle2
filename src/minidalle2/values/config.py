from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch import DeviceObjType


class ModelType(Enum):
    CLIP = "dalle2-clip"
    PRIOR = "dalle2-prior"
    DECODER = "dalle2-decoder"
    DALLE2 = "dalle2"


class Stage(Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class Config:
    device: DeviceObjType = None
    image_width_height: int = None

    def load(self) -> "Config":
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_width_height = 128
        return self

    @property
    def datasets_path(self):
        raise NotImplementedError()

    @datasets_path.setter
    def datasets_path(self, v):
        raise NotImplementedError()

    def get_image_path(self, subreddit, image_id) -> Path:
        assert self.datasets_path is not None

        return Path(
            self.datasets_path,
            f"images_{str(self.image_width_height)}",
            subreddit,
            f"{image_id}.jpg",
        )
