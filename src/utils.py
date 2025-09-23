import logging


def get_logger(name: str, level=logging.INFO, log_file="train.log"):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
