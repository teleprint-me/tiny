"""
Module: tiny.logger
Description: A simple automated logger factory.
"""

import logging
import sys
from logging import DEBUG, INFO, Logger


class TinyLogger:

    @classmethod
    def new_logger(cls, cls_name: str, verbose: bool) -> Logger:
        level = DEBUG if verbose else INFO
        logger = logging.getLogger(name=cls_name)
        logger.setLevel(level)

    @classmethod
    def set_handler(cls, logger: Logger):
        # Check if the logger has handlers to avoid adding duplicates
        if not logger.hasHandlers():
            fmt = "%(levelname)s:%(filename)s:%(lineno)d: %(message)s"
            handler = logging.StreamHandler(stream=sys.stdout)
            formatter = logging.Formatter(fmt=fmt)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    @classmethod
    def get_logger(cls, cls_name: str, verbose: bool) -> logging.Logger:
        """
        Initialize and return a Logger instance.

        :param cls_name: The name of the class that inherits from Logger.
        :param verbose: A boolean indicating whether to enable verbose logging.
        :return: Configured logger instance.
        """
        logger = TinyLogger.new_logger(cls_name, verbose)
        TinyLogger.set_handler(logger)
        return logger
