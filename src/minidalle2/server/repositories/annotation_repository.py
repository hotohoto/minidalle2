import typing as t
from contextlib import contextmanager
from textwrap import dedent

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from minidalle2.server.values.server_config import ServerConfig
from minidalle2.values.datasets import Annotation, DatasetType, DownloadStatus


class AnnotationRepository:
    QUERY_CREATE = dedent(
        """
        CREATE TABLE redcaps
        (
            idx INTEGER PRIMARY KEY AUTOINCREMENT,
            subreddit TEXT NOT NULL,
            image_id TEXT NOT NULL,
            caption TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            dataset_type TEXT NOT NULL,
            download_status TEXT NOT NULL,
            CONSTRAINT constraint_unique_subreddit_image_id UNIQUE(subreddit, image_id)
        )
        """
    )
    QUERY_INSERT = dedent(
        """
        INSERT INTO redcaps (subreddit, image_id, caption, url, dataset_type, download_status)
        SELECT :subreddit, :image_id, :caption, :url, :dataset_type, :download_status
        WHERE NOT EXISTS(
            SELECT 1
            FROM redcaps
            WHERE (url = :url) OR (subreddit = :subreddit AND image_id = :image_id)
        )
        """
    )
    QUERY_SELECT_ROWS_TO_DOWNLOAD = dedent(
        """
        SELECT idx, subreddit, image_id, caption, url, dataset_type, download_status
        FROM redcaps
        WHERE download_status = 'NEW'
        ORDER BY random()
        LIMIT :limit
        """
    )
    QUERY_SELECT_ROWS_TO_DOWNLOAD_INCLUDE_FAILED = dedent(
        """
        SELECT idx, subreddit, image_id, caption, url, dataset_type, download_status
        FROM redcaps
        WHERE download_status = 'NEW' or download_status = 'FAILED'
        ORDER BY random()
        LIMIT :limit
        """
    )
    QUERY_COUNT_ROWS_TO_DOWNLOAD = dedent(
        """
        SELECT COUNT(*)
        FROM redcaps
        WHERE download_status = 'NEW'
        """
    )
    QUERY_COUNT_ROWS_TO_DOWNLOAD_INCLUDE_FAILED = (
        "SELECT COUNT(*) FROM redcaps WHERE download_status = 'NEW' or download_status = 'FAILED'"
    )
    QUERY_UPDATE_DOWNLOAD_STATUS = (
        "UPDATE redcaps SET download_status=:download_status WHERE idx = :idx"
    )

    QUERY_CREATE_TRAIN = dedent(
        """
        CREATE TABLE redcaps_trainset (
            redcaps_idx INTEGER,
            FOREIGN KEY(redcaps_idx) REFERENCES redcaps(idx)
        )"""
    )
    QUERY_CREATE_TEST = dedent(
        """
        CREATE TABLE redcaps_testset (
            redcaps_idx INTEGER,
            FOREIGN KEY(redcaps_idx) REFERENCES redcaps(idx)
        )
        """
    )
    QUERY_DELETE_TRAIN = "DELETE FROM redcaps_trainset"
    QUERY_DELETE_TEST = "DELETE FROM redcaps_testset"
    QUERY_BUILD_TRAIN = dedent(
        f"""
        INSERT INTO redcaps_trainset(redcaps_idx)
        SELECT idx
        FROM redcaps
        WHERE (
            download_status = '{DownloadStatus.DONE.value}'
            AND dataset_type = '{DatasetType.TRAIN.value}'
        )
        ORDER BY random()
        """
    )
    QUERY_BUILD_TEST = dedent(
        f"""
        INSERT INTO redcaps_testset(redcaps_idx)
        SELECT idx
        FROM redcaps
        WHERE (
            download_status = '{DownloadStatus.DONE.value}'
            AND dataset_type = '{DatasetType.TEST.value}'
        )
        ORDER BY random()
        """
    )
    QUERY_SELECT_TRAIN = dedent(
        """
        SELECT r.subreddit, r.image_id, r.caption, r.url
        FROM redcaps_trainset AS t
        JOIN redcaps as r
        ON r.idx = t.redcaps_idx
        WHERE t.ROWID = :rowid
        """
    )
    QUERY_SELECT_TEST = dedent(
        """
        SELECT r.subreddit, r.image_id, r.caption, r.url
        FROM redcaps_testset AS t
        JOIN redcaps as r
        ON r.idx = t.redcaps_idx
        WHERE t.ROWID = :rowid
        """
    )
    QUERY_COUNT_TRAIN = "SELECT COUNT(*) FROM redcaps_trainset"
    QUERY_COUNT_TEST = "SELECT COUNT(*) FROM redcaps_testset"

    def __init__(self, config: ServerConfig):
        self.config = config
        self.engine = create_engine(
            f"sqlite:///{self.config.index_db_path}",
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
        )
        self.transaction = None
        self.n_transaction_requests = 0

    def init(self, delete=False):
        db_path = self.config.index_db_path
        if db_path.exists() and db_path.is_file():
            if delete:
                db_path.unlink()
            else:
                raise Exception("The annotation database already exists.")

        with self.begin_transaction() as conn:
            conn.execute(self.QUERY_CREATE)
            conn.execute(self.QUERY_CREATE_TRAIN)
            conn.execute(self.QUERY_CREATE_TEST)

    def add(self, annotations: t.List[Annotation]) -> int:
        assert annotations

        with self.engine.connect() as conn:
            params = [a.to_dict() for a in annotations]
            for p in params:
                if not p or len(p.keys()) == 0:
                    print("error!")
            return conn.execute(self.QUERY_INSERT, params).rowcount

    def get_annotations_to_download(self, limit, include_failed) -> t.List[Annotation]:
        with self.begin_transaction() as conn:
            query = (
                self.QUERY_SELECT_ROWS_TO_DOWNLOAD_INCLUDE_FAILED
                if include_failed
                else self.QUERY_SELECT_ROWS_TO_DOWNLOAD
            )
            params = {"limit": limit}
            ret = tuple(conn.execute(query, params))
            return [
                Annotation(
                    idx=a[0],
                    subreddit=a[1],
                    image_id=a[2],
                    caption=a[3],
                    url=a[4],
                    dataset_type=DatasetType(a[5]),
                    download_status=DownloadStatus(a[6]),
                )
                for a in ret
            ]

    def update_download_status(self, annotations: t.List[Annotation]) -> int:
        assert annotations

        with self.begin_transaction() as conn:
            params = [
                {"idx": a.idx, "download_status": a.download_status.value} for a in annotations
            ]
            return conn.execute(self.QUERY_UPDATE_DOWNLOAD_STATUS, params).rowcount

    def count_annotations_to_download(self, include_failed) -> int:
        with self.begin_transaction() as conn:
            query = (
                self.QUERY_COUNT_ROWS_TO_DOWNLOAD_INCLUDE_FAILED
                if include_failed
                else self.QUERY_COUNT_ROWS_TO_DOWNLOAD
            )
            ret = tuple(conn.execute(query))
            assert len(ret) == 1
            assert len(ret[0]) == 1
            return ret[0][0]

    def get_train_sample(self, rowid: int) -> Annotation:
        with self.begin_transaction() as conn:
            params = {"rowid": rowid}
            ret = tuple(conn.execute(self.QUERY_SELECT_TRAIN, params))
            assert len(ret) == 1
            return Annotation(
                subreddit=ret[0][0],
                image_id=ret[0][1],
                caption=ret[0][2],
            )

    def get_test_sample(self, rowid: int) -> Annotation:
        with self.begin_transaction() as conn:
            params = {"rowid": rowid}
            ret = conn.execute(self.QUERY_SELECT_TEST, params)
            assert len(ret) == 1
            return Annotation(
                subreddit=ret[0][0],
                image_id=ret[0][1],
                caption=ret[0][2],
            )

    def count_trainset(self) -> int:
        with self.begin_transaction() as conn:
            ret = tuple(conn.execute(self.QUERY_COUNT_TRAIN))
            assert len(ret) == 1
            assert len(ret[0]) == 1
            return ret[0][0]

    def count_testset(self) -> int:
        with self.begin_transaction() as conn:
            ret = tuple(conn.execute(self.QUERY_COUNT_TEST))
            assert len(ret) == 1
            assert len(ret[0]) == 1
            return ret[0][0]

    def reset_indices(self):
        with self.begin_transaction() as conn:
            conn.execute(self.QUERY_DELETE_TRAIN)
            conn.execute(self.QUERY_BUILD_TRAIN)
            conn.execute(self.QUERY_DELETE_TEST)
            conn.execute(self.QUERY_BUILD_TEST)

    @contextmanager
    def begin_transaction(self):
        self.n_transaction_requests += 1
        if self.n_transaction_requests > 1:
            yield self.transaction
        else:
            with self.engine.begin() as conn:
                self.transaction = conn
                yield self.transaction
                assert self.n_transaction_requests == 1
                self.transaction = None
        self.n_transaction_requests -= 1
