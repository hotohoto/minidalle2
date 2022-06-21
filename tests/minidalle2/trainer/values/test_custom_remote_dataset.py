import os
import signal
import socket
import subprocess
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
import torch
from requests.adapters import HTTPAdapter, Retry

from minidalle2.server.values.server_config import ServerConfig
from minidalle2.trainer.values.custom_remote_dataset import CustomRemoteDataset
from minidalle2.trainer.values.trainer_config import TrainerConfig
from minidalle2.values.datasets import DatasetType


class TestCustomDataset:
    @staticmethod
    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @classmethod
    @contextmanager
    def launch_web_server(cls, server_config: ServerConfig):
        server_port = int(cls.find_free_port())

        try:
            web_server_process = subprocess.Popen(
                f"DATASETS_PATH={server_config.DATASETS_PATH} uvicorn --port={server_port} --app-dir=src minidalle2.server.api.main:app",
                shell=True,
                preexec_fn=os.setsid,
            )

            datasets_url = f"http://localhost:{server_port}"

            session = requests.Session()
            session.mount(
                "http://",
                HTTPAdapter(
                    max_retries=Retry(
                        total=10,
                        backoff_factor=0.2,
                        status_forcelist=[500, 502, 503, 504],
                    )
                ),
            )
            session.get(datasets_url)

            yield datasets_url
        finally:
            if web_server_process is not None:
                os.killpg(os.getpgid(web_server_process.pid), signal.SIGTERM)

    def test_len_and_getitem(self):
        server_config = ServerConfig()
        server_config.DATASETS_PATH = "tests/fixtures/datasets"

        with TemporaryDirectory() as tmpdir, self.launch_web_server(server_config) as datasets_url:
            trainer_config = TrainerConfig()
            trainer_config.DATASETS_URL = datasets_url
            trainer_config.TMP_DATASETS_PATH = str(Path(tmpdir))

            dataset = CustomRemoteDataset(trainer_config, DatasetType.TRAIN)

            assert len(dataset) == 3
            sample = dataset[0]
            assert sample is not None
            assert sample["image"].shape == torch.Size([3, 128, 128])
            assert sample["caption"]
