import logging
from datetime import datetime
from pathlib import Path
from experiments.config import ExperimentConfig


def setup_logging(cfg: ExperimentConfig, run_name: str) -> str:
    """Set up logging to both file and console. Returns the log file path."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"{run_name}.log"

    # Configure the root logger so all modules (training.*, experiments.*) share these handlers.
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    for key, value in vars(cfg).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    return str(log_file)


def get_logger(name: str) -> logging.Logger:
    """Get the experiment logger."""
    return logging.getLogger(name)
