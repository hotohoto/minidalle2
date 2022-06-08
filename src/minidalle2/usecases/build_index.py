import json
import sqlite3

import pandas as pd

from minidalle2.values.config import DatasetType
from minidalle2.values.server_config import ServerConfig


def build_index(config: ServerConfig, trainset_ratio=0.99, skip_na_entries=True, random_state=200):
    trainset_db_path = config.get_index_db_path(DatasetType.TRAIN)
    testset_db_path = config.get_index_db_path(DatasetType.TEST)

    if trainset_db_path.exists() and trainset_db_path.is_file():
        trainset_db_path.unlink()

    with sqlite3.connect(trainset_db_path) as conn_train, sqlite3.connect(
        testset_db_path
    ) as conn_test:
        cursor_train = conn_train.cursor()
        cursor_test = conn_test.cursor()
        QUERY_CREATE = "CREATE TABLE redcaps (subreddit TEXT, image_id TEXT, caption TEXT)"
        cursor_train.execute(QUERY_CREATE)
        cursor_test.execute(QUERY_CREATE)

        for p in config.get_annotations_path().iterdir():
            if not p.is_file():
                continue

            subreddits = []
            image_ids = []
            captions = []

            with open(p) as f:
                for a in json.load(f)["annotations"]:
                    subreddit = a["subreddit"]
                    image_id = a["image_id"]
                    caption = a["caption"]

                    if skip_na_entries and not config.get_image_path(subreddit, image_id).exists():
                        continue

                    subreddits.append(subreddit)
                    image_ids.append(image_id)
                    captions.append(caption)

            df = pd.DataFrame(
                {
                    "subreddit": subreddits,
                    "image_id": image_ids,
                    "caption": captions,
                }
            )

            df_train = df.sample(frac=trainset_ratio, random_state=random_state)
            df_test = df.drop(df_train.index)

            QUERY_INSERT = "INSERT INTO redcaps VALUES (?, ?, ?)"

            for _, row in df_train.iterrows():
                cursor_train.execute(
                    QUERY_INSERT,
                    (row["subreddit"], row["image_id"], row["caption"]),
                )

            for _, row in df_test.iterrows():
                cursor_test.execute(
                    QUERY_INSERT,
                    (row["subreddit"], row["image_id"], row["caption"]),
                )
