import logging


class RankedLogger(logging.LoggerAdapter):
    def __init__(self, name=__name__, **kwargs):
        super().__init__(logging.getLogger(name), kwargs)
