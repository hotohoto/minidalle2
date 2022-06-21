from fastapi import FastAPI

from minidalle2.server.api.datasets import register_api
from minidalle2.server.repositories.annotation_repository import AnnotationRepository
from minidalle2.server.values.server_config import ServerConfig

app = FastAPI()
config = ServerConfig().load()
repo = AnnotationRepository(config=config)


@app.get("/health")
def health():
    return "ok"


register_api(app=app, config=config, repo=repo)
