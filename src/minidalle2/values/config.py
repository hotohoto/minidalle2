from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch import DeviceObjType


class DatasetType(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"


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

    def load(self) -> "Config":
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self

    @property
    def datasets_path(self):
        raise NotImplementedError()

    def get_index_db_path(self, type_: DatasetType) -> Path:
        assert self.datasets_path is not None

        if type_ is DatasetType.TRAIN:
            return Path(self.datasets_path, "index_trainset.db")
        elif type_ is DatasetType.TEST:
            return Path(self.datasets_path, "index_testset.db")
        else:
            raise ValueError()

    def get_image_path(self, subreddit, image_id) -> Path:
        assert self.datasets_path is not None

        return Path(self.datasets_path, "images", subreddit, f"{image_id}.jpg")
