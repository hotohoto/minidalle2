import logging
import logging.config
from pathlib import Path


def init_logging():
    logging.config.fileConfig(Path(f"{Path(__file__).parent.parent}/logging.conf"))
