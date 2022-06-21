from functools import lru_cache

from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.values.datasets import Annotation, DatasetType


@lru_cache(maxsize=128)
def get_annotation_with_cache(
    annotation_repo: AnnotationRepository, dataset_type: DatasetType, rowid: int
) -> Annotation:
    if dataset_type is DatasetType.TRAIN:
        return annotation_repo.get_train_sample(rowid)
    elif dataset_type is DatasetType.TEST:
        return annotation_repo.get_test_sample(rowid)
    else:
        raise ValueError()
