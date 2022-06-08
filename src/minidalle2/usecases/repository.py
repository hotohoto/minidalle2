import mlflow
from dalle2_pytorch import CLIP, DALLE2, Decoder, DiffusionPrior
from mlflow.tracking import MlflowClient

from minidalle2.values.trainer_config import TrainerConfig


class Repository:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.client = MlflowClient()

    def check_if_exists(self, model_name):
        models = self.client.search_registered_models()
        for m in models:
            if m.name == model_name:
                return True
        return False

    def load_clip(self) -> CLIP:
        if not self.check_if_exists("clip"):
            return None
        return mlflow.pytorch.load_model("mlflow-artifacts:/clip/Staging")

    def load_prior(self) -> DiffusionPrior:
        if not self.check_if_exists("prior"):
            return None
        return mlflow.pytorch.load_model("mlflow-artifacts:/prior/Staging")

    def load_decoder(self) -> Decoder:
        if not self.check_if_exists("decoder"):
            return None
        return mlflow.pytorch.load_model("mlflow-artifacts:/decoder/Staging")

    def load_dalle2(self) -> DALLE2:
        return mlflow.pytorch.load_model("mlflow-artifacts:/dalle2/Staging")

    def save_clip(self, clip: CLIP):
        mlflow.pytorch.log_model(clip, "clip")

    def save_prior(self, prior: DiffusionPrior):
        mlflow.pytorch.log_model(prior, "prior")

    def save_decoder(self, decoder: Decoder):
        mlflow.pytorch.log_model(decoder, "decoder")

    def save_dalle2(self, dalle2: DALLE2):
        mlflow.pytorch.log_model(dalle2, "dalle2")
