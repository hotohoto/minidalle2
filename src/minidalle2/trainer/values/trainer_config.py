import os
import typing as t
from pathlib import Path

from minidalle2.values.config import Config, ModelType, Stage
from minidalle2.values.datasets import DatasetType


class TrainerConfig(Config):
    datasets_url: str
    mlflow_tracking_uri: str
    tmp_datasets_path: str
    n_dataloader_workers: int
    n_epochs_clip: int
    n_epochs_prior: int
    n_epochs_decoder: int
    overwrite_index: bool

    def load(self) -> "TrainerConfig":
        super().load()
        self.datasets_url = os.environ.get("DATASETS_URL", "http://localhost:8000")
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.tmp_datasets_path = os.environ.get(
            "TMP_DATASETS_PATH", str(Path("tmp_datasets").resolve())
        )
        self.n_dataloader_workers = 8
        self.n_epochs_clip = 1
        self.n_epochs_prior = 1
        self.n_epochs_decoder = 1
        self.overwrite_index = False
        return self

    @property
    def datasets_path(self):
        assert self.tmp_datasets_path is not None

        return self.tmp_datasets_path

    def get_data_length_url(self, dataset_type: DatasetType):
        assert self.datasets_url is not None
        return f"{self.datasets_url}/data/{dataset_type.value.lower()}/length"

    def get_data_url(self, rowid: int, dataset_type: DatasetType):
        assert rowid > 0
        assert self.datasets_url is not None

        return f"{self.datasets_url}/data/{dataset_type.value.lower()}/{rowid}"

    def get_image_url(self, subreddit: str, image_id: str):
        assert self.datasets_url is not None

        return f"{self.datasets_url}/data/image/{subreddit}/{image_id}.jpg"

    def get_model_uri(self, run_id, model_type: ModelType):
        # Note that "--serve-artifacts" may require "mlflow-artifacts:"
        return f"mlflow-artifacts:/{run_id}/{model_type.value}"

    def get_registered_model_name(self, model_type: ModelType, version: t.Union[Stage, int] = None):
        # Do we actually needs "mlflow-artifacts:" ??
        if version is None:
            return f"{model_type.value}"
        elif isinstance(version, Stage):
            return f"{model_type.value}/{version.value}"
        else:
            return f"{model_type.value}/{version}"
