import mlflow
import torch
import torchvision.models

from minidalle2.repositories.mlflow_repository import MlflowRepository, start_run
from minidalle2.values.config import ModelType
from minidalle2.values.trainer_config import TrainerConfig


class TestTmp:
    def test_save_and_load_clip(self):
        config = TrainerConfig().load()
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

        repo = MlflowRepository(config)
        with start_run(repo, ModelType.CLIP):
            model = torchvision.models.vgg16(pretrained=True)
            repo.register_model(model, trained_epochs=3, trained_steps=5)

        registered_model = repo.load_model(ModelType.CLIP)

        assert all(
            torch.equal(v, w)
            for v, w in zip(registered_model.model.parameters(), model.parameters())
        )
        assert registered_model.trained_epochs == 3
        assert registered_model.trained_steps == 5
