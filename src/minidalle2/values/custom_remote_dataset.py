import sqlite3
import typing as t
from pathlib import Path

import requests
from PIL import Image
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from minidalle2.values.trainer_config import TrainerConfig


class CustomRemoteDataset(Dataset):
    def __init__(self, config: TrainerConfig, index_db_path: Path):
        super().__init__()

        self.config = config
        self.index_db_path = index_db_path
        self.pil_image_to_tensor = transforms.ToTensor()
        self._length = None

    def __getitem__(self, index) -> t.Dict[Tensor, str]:
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT subreddit, image_id, caption FROM redcaps WHERE rowid=?", (index + 1,)
            )
            result = cursor.fetchone()
            subreddit, image_id, caption = result
            image_url = self.config.get_image_url(subreddit, image_id)
            local_image_path = self.config.get_image_path(subreddit, image_id)

            if not local_image_path.exists():
                respnose = requests.get(image_url)
                local_image_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_image_path, "wb") as f:
                    f.write(respnose.content)

            return dict(
                image=self.pil_image_to_tensor(Image.open(local_image_path)),
                caption=caption,
            )

    def __len__(self):
        if self._length is None:
            with sqlite3.connect(self.index_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM redcaps")
                self._length = cursor.fetchone()[0]
        return self._length
