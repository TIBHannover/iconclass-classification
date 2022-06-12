import logging
from .strategy import get_node_rank


class LoggingHandler:
    @classmethod
    def __init_subclass__(
        cls,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls.logging = logging.getLogger(cls.__name__)

    def __init__(self, *args, **kwargs):
        self.logging = logging.getLogger(self.__class__.__name__)