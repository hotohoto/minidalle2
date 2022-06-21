from fastapi import FastAPI
from starlette.responses import FileResponse

from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.usecases.get_data import get_annotation_with_cache
from minidalle2.server.usecases.get_dataset_len import get_dataset_len
from minidalle2.server.values.server_config import ServerConfig
from minidalle2.values.datasets import DatasetType


def register_api(app: FastAPI, config: ServerConfig, repo: AnnotationRepository):
    def upper(dataset_type: str) -> DatasetType:
        return DatasetType(dataset_type.upper())

    @app.get("/data/{dataset_type}/length")
    async def _get_dataset_length(dataset_type: str):
        return get_dataset_len(repo, dataset_type=upper(dataset_type))

    @app.get("/data/{dataset_type}/image/{rowid}")
    async def _get_image(dataset_type: str, rowid: int):
        assert rowid > 0
        annotation = get_annotation_with_cache(repo, upper(dataset_type), rowid)
        file_location = config.get_image_path(annotation.subreddit, annotation.image_id)
        return FileResponse(
            file_location,
            media_type="application/octet-stream",
            filename=f"{annotation.image_id}.jpg",
        )

    @app.get("/data/{dataset_type_}/{rowid}")
    async def _get_data(dataset_type_: str, rowid: int):
        dataset_type = upper(dataset_type_)
        assert rowid > 0
        annotation = get_annotation_with_cache(repo, dataset_type, rowid)
        return {
            "caption": annotation.caption,
            "subreddit": annotation.subreddit,
            "image_id": annotation.image_id,
        }
