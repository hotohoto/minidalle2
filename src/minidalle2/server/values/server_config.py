import os
from pathlib import Path

from minidalle2.values.config import Config


class ServerConfig(Config):
    _datasets_path: str = None

    def load(self) -> "ServerConfig":
        super().load()
        self.datasets_path = os.environ.get("DATASETS_PATH", str(Path("datasets").resolve()))
        return self

    @property
    def datasets_path(self):
        assert self._datasets_path is not None

        return self._datasets_path

    @datasets_path.setter
    def datasets_path(self, v):
        self._datasets_path = v

    @property
    def index_db_path(self):
        return Path(self.datasets_path, "index.db")

    def get_annotations_path(self):
        return Path(self.datasets_path, "annotations")
