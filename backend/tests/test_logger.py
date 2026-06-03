import logging
from utils.logger import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name():
    logger = get_logger("mymodule")
    assert logger.name == "mymodule"


def test_get_logger_level():
    logger = get_logger("test")
    assert logger.level == logging.INFO
