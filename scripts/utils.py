import logging
from logging import getLogger


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
