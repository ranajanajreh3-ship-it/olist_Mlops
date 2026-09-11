import logging
import logging.handlers
import sys


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    logger_name: str = "mlops_pipeline",
    log_file: str = "pipeline.log",
    level: int = logging.INFO,
) -> logging.Logger:

    logger = logging.getLogger(logger_name)

    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


logger = setup_logging()

