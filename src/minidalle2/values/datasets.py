from dataclasses import dataclass
from enum import Enum


class DatasetType(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"


class DownloadStatus(Enum):
    DONE = "DONE"
    NEW = "NEW"
    FAILED = "FAILED"


@dataclass
class Annotation:
    idx: int = None
    subreddit: str = None
    image_id: str = None
    caption: str = None
    url: str = None
    dataset_type: DatasetType = None
    download_status: DownloadStatus = None

    def to_dict(self):
        return {
            "int": self.idx,
            "subreddit": self.subreddit,
            "image_id": self.image_id,
            "caption": self.caption,
            "url": self.url,
            "dataset_type": self.dataset_type.value,
            "download_status": self.download_status.value,
        }
