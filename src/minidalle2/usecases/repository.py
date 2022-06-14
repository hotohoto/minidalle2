import typing as t

import mlflow
from dalle2_pytorch import CLIP, DALLE2, Decoder, DiffusionPrior
from mlflow.tracking import MlflowClient

from minidalle2.values.config import ModelType, Stage
from minidalle2.values.trainer_config import TrainerConfig


class Repository:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.client = MlflowClient(config.MLFLOW_TRACKING_URI)

    def check_if_exists(self, model_type: ModelType):
        models = self.client.search_registered_models()
        for m in models:
            if m.name == self.config.get_registered_model_name(model_type):
                return True
        return False

    def load_model(self, model_type: ModelType, stage: Stage = None) -> t.Union[CLIP, DiffusionPrior, Decoder, DALLE2]:
        if not self.check_if_exists(model_type):
            return None
        name = self.config.get_registered_model_name(model_type)
        if stage is None:
            run_id = self.client.get_latest_versions(name=name)[0].run_id
        else:
            run_id = self.client.get_latest_versions(name=name, stages=[stage.value]).run_id

        return mlflow.pytorch.load_model(self.config.get_model_uri(run_id, model_type))

    def save_model(self, run_id, model_type: ModelType):
        mlflow.register_model(
            self.config.get_model_uri(run_id, model_type),
            self.config.get_registered_model_name(model_type),
        )
