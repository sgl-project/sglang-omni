"""Application-owned routing for dependency logs."""

import logging

from huggingface_hub.utils import logging as hf_logging


def configure_dependency_loggers() -> None:
    hub_logger = hf_logging.get_logger()
    for console_handler in hub_logger.handlers[:]:
        if type(console_handler) is logging.StreamHandler:
            hub_logger.removeHandler(console_handler)
        else:
            pass
    hf_logging.enable_propagation()
    for logger_name in ("httpx", "httpcore"):
        http_logger = logging.getLogger(logger_name)
        if http_logger.level == logging.NOTSET:
            http_logger.setLevel(logging.WARNING)
        else:
            pass
