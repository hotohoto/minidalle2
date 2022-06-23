import logging
import math

from PIL import Image
from tqdm import tqdm

from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.values.server_config import ServerConfig
from minidalle2.utils.validate_image import validate_image
from minidalle2.values.datasets import DownloadStatus

_LOGGER = logging.getLogger(__name__)


def validate_annotations(
    config: ServerConfig,
    annotation_repo: AnnotationRepository,
):
    # validate download_status and reset it if required
    total_annotations_downloaded = annotation_repo.count_annotations_by_download_status(
        DownloadStatus.DONE
    )

    limit = 500
    offset = 0
    n_iterations = math.ceil(total_annotations_downloaded / limit)

    for _ in tqdm(range(n_iterations), total=n_iterations, desc=f"{limit} imgs/it"):
        annotations = annotation_repo.get_annotations_downloaded(limit=limit, offset=offset)
        if not annotations:
            break

        annotations_to_reset = []
        for a in annotations:
            image_path = config.get_image_path(a.subreddit, a.image_id)
            if validate_image(image_path):
                continue

            a.download_status = DownloadStatus.FAILED
            annotations_to_reset.append(a)

        if annotations_to_reset:
            n_updated = annotation_repo.update_download_status(annotations_to_reset)
            assert n_updated == len(annotations_to_reset)
            _LOGGER.info(f"Fixed {n_updated} annotations")

        offset += limit
