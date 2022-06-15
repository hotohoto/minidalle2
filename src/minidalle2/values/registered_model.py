import dataclasses
import typing as t
from dataclasses import dataclass

import torch

from minidalle2.values.config import ModelType

TAG_KEY_TRAINED_STEPS = "minidalle2.trained_steps"
TAG_KEY_TRAINED_EPOCHS = "minidalle2.trained_epochs"


@dataclass
class RegisteredModel:
    model: torch.nn.Module
    tags: t.Dict[str, str]

    @property
    def trained_steps(self):
        if not self.tags or TAG_KEY_TRAINED_STEPS not in self.tags:
            return None
        return int(self.tags[TAG_KEY_TRAINED_STEPS])

    @property
    def trained_epochs(self):
        if not self.tags or TAG_KEY_TRAINED_EPOCHS not in self.tags:
            return None
        return int(self.tags[TAG_KEY_TRAINED_EPOCHS])
