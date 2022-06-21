from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.usecases.download_images import download_images
from minidalle2.server.values.server_config import ServerConfig


class TestDownloadImagesCommand:
    def test_download_images(self):
        config = ServerConfig().load()
        repo = AnnotationRepository(config)
        download_images(config, repo, retry_failed=False)
