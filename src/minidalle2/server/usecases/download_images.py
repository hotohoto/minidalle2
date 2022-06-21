import io
import logging
import math
from functools import partial
from multiprocessing import Pool, cpu_count

import requests
from PIL import Image
from tqdm import tqdm

from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.values.server_config import ServerConfig
from minidalle2.values.datasets import Annotation, DownloadStatus

_LOGGER = logging.getLogger(__name__)


def runner(config: ServerConfig, annotation: Annotation):
    class SilentError(Exception):
        pass

    try:
        image_path = config.get_image_path(annotation.subreddit, annotation.image_id)
        if not image_path.exists() or not image_path.is_file():
            response = requests.get(annotation.url)

            # Check if image was downloaded (response must be 200). One exception:
            # Imgur gives response 200 with "removed.png" image if not found.
            if response.status_code != 200 or "removed.png" in response.url:
                raise SilentError(f"Image not available: {annotation.url}")

            # Write image to disk if it was downloaded successfully.
            pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image_width, image_height = pil_image.size
            scale = config.image_width_height / float(min(image_width, image_height))
            new_width, new_height = tuple(
                int(round(d * scale)) for d in (image_width, image_height)
            )
            if new_width >= new_height:
                new_left = int((new_width - config.image_width_height) / 2)
                new_top = 0
                new_right = new_left + config.image_width_height
                new_bottom = config.image_width_height
            else:
                new_left = 0
                new_top = int((new_height - config.image_width_height) / 2)
                new_right = config.image_width_height
                new_bottom = new_top + config.image_width_height
            pil_image = pil_image.resize((new_width, new_height)).crop(
                (new_left, new_top, new_right, new_bottom)
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(image_path)

        annotation.download_status = DownloadStatus.DONE
    except Exception as e:
        annotation.download_status = DownloadStatus.FAILED
        if not isinstance(e, SilentError):
            _LOGGER.warn(e)
    finally:
        return annotation


def download_images(
    config: ServerConfig,
    annotation_repo: AnnotationRepository,
    retry_failed: bool,
):
    total_annotations_to_download = annotation_repo.count_annotations_to_download(
        include_failed=retry_failed
    )
    limit = 300
    n_iterations = math.ceil(total_annotations_to_download / limit)

    pool = Pool(cpu_count())
    for _ in tqdm(range(n_iterations), total=n_iterations, desc=f"{limit} imgs/it"):
        input_annotations = annotation_repo.get_annotations_to_download(
            limit=limit, include_failed=retry_failed
        )
        _runner = partial(runner, config)
        output_annotations = tuple(a for a in pool.map(_runner, input_annotations) if a)
        if not output_annotations:
            continue
        n_updated = annotation_repo.update_download_status(output_annotations)
        assert n_updated == len(output_annotations)

    pool.close()
    pool.join()
