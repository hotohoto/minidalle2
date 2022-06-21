import os
import typing as t
from pathlib import Path

from minidalle2.values.config import Config, ModelType, Stage
from minidalle2.values.datasets import DatasetType


class TrainerConfig(Config):
    DATASETS_URL: str
    MLFLOW_TRACKING_URI: str
    TMP_DATASETS_PATH: str
    n_epochs_clip: int
    n_epochs_prior: int
    n_epochs_decoder: int
    overwrite_index: bool

    def load(self) -> "TrainerConfig":
        super().load()
        self.DATASETS_URL = os.environ.get("DATASETS_URL", "http://localhost:8000")
        self.MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.TMP_DATASETS_PATH = os.environ.get(
            "TMP_DATASETS_PATH", str(Path("tmp_datasets").resolve())
        )
        self.n_epochs_clip = 0
        self.n_epochs_prior = 0
        self.n_epochs_decoder = 0
        self.overwrite_index = False
        return self

    @property
    def datasets_path(self):
        assert self.TMP_DATASETS_PATH is not None

        return self.TMP_DATASETS_PATH

    def get_data_length_url(self, dataset_type: DatasetType):
        assert self.DATASETS_URL is not None
        return f"{self.DATASETS_URL}/data/{dataset_type.value.lower()}/length"

    def get_data_url(self, rowid: int, dataset_type: DatasetType):
        assert rowid > 0
        assert self.DATASETS_URL is not None

        return f"{self.DATASETS_URL}/data/{dataset_type.value.lower()}/{rowid}"

    def get_image_url(self, rowid: int, dataset_type: DatasetType):
        assert rowid > 0
        assert self.DATASETS_URL is not None

        return f"{self.DATASETS_URL}/data/{dataset_type.value.lower()}/image/{rowid}"

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
