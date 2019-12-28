import os
import pandas as pd

from utils import get_logger

logger = get_logger()
file_path = os.path.dirname(__file__)


def read_data_all(dev=False):
    train = read_train(dev)
    test = read_test(dev)
    # specs = read_specs()
    train_labels = read_train_labels()
    submission = read_submission()
    logger.info('Reading data finished')
    return train, test, train_labels, submission


def read_train(dev=False):
    logger.info('Reading train.csv')
    N_ROW = None if not dev else 100000
    return pd.read_csv(os.path.join(file_path, '../data/input/train.csv'), nrows=N_ROW)


def read_test(dev=False):
    logger.info('Reading test.csv')
    N_ROW = None if not dev else 100000
    return pd.read_csv(os.path.join(file_path, '../data/input/test.csv'), nrows=N_ROW)


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
