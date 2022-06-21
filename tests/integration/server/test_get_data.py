from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.usecases.get_data import get_annotation_with_cache
from minidalle2.server.usecases.import_annotations import import_annotations
from minidalle2.server.values.server_config import ServerConfig
from minidalle2.values.datasets import DatasetType


class TestImportAnnotationsCommand:
    def test_import_annotations(self):
        config = ServerConfig().load()
        repo = AnnotationRepository(config)
        get_annotation_with_cache(annotation_repo=repo, dataset_type=DatasetType.TRAIN, rowid=1)
