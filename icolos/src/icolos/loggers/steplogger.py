import logging
import multiprocessing

from icolos.loggers.base_logger import BaseLogger


class StepLogger(BaseLogger):
    def __init__(self):
        super().__init__()

    def _initialize_logger(self):
        # logger = multiprocessing.log_to_stderr()
        # logger.name = self._LE.LOGGER_STEP
        logger = logging.getLogger(self._LE.LOGGER_STEP)
        return logger
