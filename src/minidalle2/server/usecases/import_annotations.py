import json
import logging

import pandas as pd
from tqdm import tqdm

from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.values.server_config import ServerConfig
from minidalle2.values.datasets import Annotation, DatasetType, DownloadStatus

_LOGGER = logging.getLogger(__name__)


def import_annotations(
    config: ServerConfig,
    annotation_repo: AnnotationRepository,
    trainset_ratio=0.99,
    random_state=200,
):
    annotation_repo.init(delete=True)

    paths = tuple(config.get_annotations_path().iterdir())
    max_name_len = max(len(p.name) for p in paths)

    n_total_added = 0
    n_total_entries = 0

    progress = tqdm(paths, total=len(paths))
    for p in progress:
        display_name = (p.name + " " * max_name_len)[:max_name_len]
        progress.set_description(display_name)
        if not p.is_file():
            continue

        subreddits = []
        image_ids = []
        captions = []
        urls = []

        pkset = set()
        urlset = set()

        with open(p) as f:
            raw_annotations = json.load(f)["annotations"]

            for a in raw_annotations:
                subreddit = a["subreddit"]
                image_id = a["image_id"]
                caption = a["caption"]
                url = a["url"]

                if not subreddit or not image_id or not caption or not url:
                    continue

                pk = (subreddit, image_id)
                if pk in pkset:
                    continue

                if url in urlset:
                    continue

                pkset.add(pk)
                urlset.add(url)

                subreddits.append(subreddit)
                image_ids.append(image_id)
                captions.append(caption)
                urls.append(url)

        n_entries = len(raw_annotations)

        if n_entries == 0:
            n_added = 0
            percentage = 0

        else:
            df = pd.DataFrame(
                {
                    "subreddit": subreddits,
                    "image_id": image_ids,
                    "caption": captions,
                    "url": urls,
                }
            )

            idx_train = df.index.to_series().sample(frac=trainset_ratio, random_state=random_state)
            df["dataset_type"] = None
            df.loc[idx_train, "dataset_type"] = DatasetType.TRAIN
            df["dataset_type"] = df["dataset_type"].fillna(DatasetType.TEST)

            annotations = []
            for _, row in df.iterrows():
                image_path = config.get_image_path(subreddit, image_id)
                annotations.append(
                    Annotation(
                        subreddit=row["subreddit"],
                        image_id=row["image_id"],
                        caption=row["caption"],
                        url=row["url"],
                        dataset_type=row["dataset_type"],
                        download_status=(
                            DownloadStatus.DONE
                            if image_path.exists() and image_path.is_file()
                            else DownloadStatus.NEW
                        ),
                    )
                )

            n_added = annotation_repo.add(annotations) if annotations else 0
            percentage = (n_added / n_entries) * 100

        _LOGGER.debug("{} {:.2f}% ({}/{})".format(display_name, percentage, n_added, n_entries))

        n_total_added += n_added
        n_total_entries += n_entries

    total_percentage = (n_total_added / n_total_entries) * 100 if n_total_entries else 0

    _LOGGER.debug(
        "(total) {:.2f}% ({}/{})".format(total_percentage, n_total_added, n_total_entries)
    )
