from __future__ import annotations

import logging


def setup_logger(name: str = "hh_level_poc") -> logging.Logger:
    """Создаёт и настраивает логгер.

    Parameters
    ----------
    name
        Имя логгера.

    Returns
    -------
    logging.Logger
        Настроенный логгер.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
