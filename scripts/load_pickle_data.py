import os
import pandas as pd
import yaml
from utils import get_logger

logger = get_logger()
file_path = os.path.dirname(__file__)

CONFIG_FILE = '../config/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
INPUT_DIR_NAME = yml['SETTING']['INPUT_DIR_NAME']


def read_pickle_data_all():
    train = read_train()
    test = read_test()
    specs = read_specs()
    train_labels = read_train_labels()
    return train, test, specs, train_labels


def read_train():
    logger.info('Reading train.pkl')
    return pd.read_pickle(os.path.join(file_path, INPUT_DIR_NAME + 'train.pkl'))


def read_test():
    logger.info('Reading test.pkl')
    return pd.read_pickle(os.path.join(file_path, INPUT_DIR_NAME + 'test.pkl'))


def read_specs():
    logger.info('Reading specs.pkl')
    return pd.read_pickle(os.path.join(file_path, INPUT_DIR_NAME + 'specs.pkl'))


def read_train_labels():
    logger.info('Reading train_labels.pkl')
    return pd.read_pickle(os.path.join(file_path, INPUT_DIR_NAME + 'train_labels.pkl'))


if __name__ == '__main__':
    read_pickle_data_all()
