"""
Project-wide logger utility.
Logs with module name, time, status, and message.
"""
import logging
import sys
from typing import Optional

def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(module_name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
