import mlflow
import torch
import torchvision.models

from minidalle2.usecases.repository import Repository
from minidalle2.values.config import ModelType
from minidalle2.values.trainer_config import TrainerConfig


class TestTmp:
    def test_save_and_load_clip(self):
        config = TrainerConfig().load()
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        repo = Repository(config)
        with mlflow.start_run() as run:
            model = torchvision.models.vgg16(pretrained=True)
            mlflow.pytorch.log_model(model, ModelType.CLIP.value)
        repo.save_model(run_id=run.info.run_id, model_type=ModelType.CLIP)
        model_loaded = repo.load_model(ModelType.CLIP)
        all(torch.equal(v, w) for v, w in zip(model_loaded.parameters(), model.parameters()))
