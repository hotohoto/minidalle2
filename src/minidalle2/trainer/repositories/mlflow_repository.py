from contextlib import contextmanager

import mlflow
from mlflow.tracking import MlflowClient

from minidalle2.trainer.values.registered_model import (
    TAG_KEY_TRAINED_EPOCHS,
    TAG_KEY_TRAINED_STEPS,
    RegisteredModel,
)
from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.config import ModelType, Stage


class MlflowRepository:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.client = MlflowClient(config.MLFLOW_TRACKING_URI)
        self.active_run = None
        self.active_run_model_type = None

    def check_if_exists(self, model_type: ModelType):
        models = self.client.search_registered_models()

        for m in models:
            if m.name == self.config.get_registered_model_name(model_type):
                return True
        return False

    def get_latest_model_version(self, model_type: ModelType, stage: Stage = None):
        if not self.check_if_exists(model_type):
            return None
        name = self.config.get_registered_model_name(model_type)
        if stage is None:
            fetched_versions = self.client.get_latest_versions(name=name)
        else:
            fetched_versions = self.client.get_latest_versions(name=name, stages=[stage.value])

        if not fetched_versions:
            return None

        latest_version = fetched_versions[0]
        for v in fetched_versions[1:]:
            if int(v.version) > int(latest_version.version):
                latest_version = v

        return latest_version

    def load_model(self, model_type: ModelType, stage: Stage = None) -> RegisteredModel:
        latest_version = self.get_latest_model_version(model_type, stage)

        if not latest_version:
            return None

        run_id = latest_version.run_id
        tags = latest_version.tags
        model = mlflow.pytorch.load_model(self.config.get_model_uri(run_id, model_type))

        return RegisteredModel(model=model, tags=tags)

    def register_model(self, model, trained_epochs=None, trained_steps=None):
        model_type = self.active_run_model_type

        registered_model_name = self.config.get_registered_model_name(model_type)

        mlflow.pytorch.log_model(
            model, model_type.value, registered_model_name=registered_model_name
        )

        latest_version = self.get_latest_model_version(model_type).version

        self.client.set_model_version_tag(
            registered_model_name,
            version=latest_version,
            key=TAG_KEY_TRAINED_EPOCHS,
            value=str(trained_epochs),
        )
        self.client.set_model_version_tag(
            registered_model_name,
            version=latest_version,
            key=TAG_KEY_TRAINED_STEPS,
            value=str(trained_steps),
        )


@contextmanager
def start_run(repo: MlflowRepository, model_type: ModelType):
    with mlflow.start_run() as run:
        try:
            if repo.active_run or repo.active_run_model_type:
                raise Exception("The repository has an active run already.")
            repo.active_run = run
            repo.active_run_model_type = model_type
            yield
        finally:
            repo.active_run = None
            repo.active_run_model_type = None
