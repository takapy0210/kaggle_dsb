import os
import pandas as pd

from utils import get_logger

logger = get_logger()
file_path = os.path.dirname(__file__)


def read_data_all():
    train = read_train()
    test = read_test()
    # specs = read_specs()
    train_labels = read_train_labels()
    submission = read_submission()
    logger.info('Reading data finished')
    return train, test, train_labels, submission


def read_train():
    logger.info('Reading train.csv')
    return pd.read_csv(os.path.join(file_path, '../data/input/train.csv'))


def read_test():
    logger.info('Reading test.csv')
    return pd.read_csv(os.path.join(file_path, '../data/input/test.csv'))


def read_specs():
    logger.info('Reading specs.csv')
    return pd.read_csv(os.path.join(file_path, '../data/input/specs.csv'))


def read_train_labels():
    logger.info('Reading train_labels.csv')
    return pd.read_csv(os.path.join(file_path, '../data/input/train_labels.csv'))


def read_submission():
    logger.info('Reading sample_submission.csv')
    return pd.read_csv(os.path.join(file_path, '../data/output/sample_submission.csv'))


# test
if __name__ == '__main__':
    read_data_all()
