import logging
from pathlib import Path
from time import strftime


def setup_logging(log_dir: str = "./logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("forge_llm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(log_dir) / f"run_{strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
