import os
from pathlib import Path

from minidalle2.values.config import Config


class ServerConfig(Config):
    DATASETS_PATH: str = None

    def load(self) -> "ServerConfig":
        super().load()
        self.DATASETS_PATH = os.environ.get("DATASETS_PATH", str(Path("datasets").resolve()))
        return self

    @property
    def datasets_path(self):
        assert self.DATASETS_PATH is not None

        return self.DATASETS_PATH

    @property
    def index_db_path(self):
        return Path(self.datasets_path, "index.db")

    def get_annotations_path(self):
        return Path(self.datasets_path, "annotations")
