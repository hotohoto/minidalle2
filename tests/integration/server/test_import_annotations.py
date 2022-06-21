from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.usecases.import_annotations import import_annotations
from minidalle2.server.values.server_config import ServerConfig


class TestImportAnnotationsCommand:
    def test_import_annotations(self):
        config = ServerConfig().load()
        repo = AnnotationRepository(config)
        import_annotations(config, repo)
