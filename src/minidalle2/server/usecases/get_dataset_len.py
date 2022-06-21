from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.values.datasets import DatasetType


def get_dataset_len(annotation_repo: AnnotationRepository, dataset_type: DatasetType) -> int:
    if dataset_type is DatasetType.TRAIN:
        return annotation_repo.count_trainset()
    elif dataset_type is DatasetType.TEST:
        return annotation_repo.count_testset()
    else:
        raise ValueError()
