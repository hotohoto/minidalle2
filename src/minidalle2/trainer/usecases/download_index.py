import urllib.request
from pathlib import Path

from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.config import DatasetType


def download_index(config: TrainerConfig):
    def mkdir(target_path: Path):
        target_path.parent.mkdir(parents=True, exist_ok=True)

    local_train_db_path = config.get_index_db_path(DatasetType.TRAIN)
    local_test_db_path = config.get_index_db_path(DatasetType.TEST)

    if not local_train_db_path.exists() or config.overwrite_index:
        mkdir(local_train_db_path)
        train_db_url = config.get_index_db_url(DatasetType.TRAIN)
        urllib.request.urlretrieve(train_db_url, local_train_db_path)

    if not local_test_db_path.exists() or config.overwrite_index:
        mkdir(local_test_db_path)
        test_db_url = config.get_index_db_url(DatasetType.TEST)
        urllib.request.urlretrieve(test_db_url, local_test_db_path)
