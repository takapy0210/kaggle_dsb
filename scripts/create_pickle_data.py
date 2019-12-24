import os
import pandas as pd
import yaml
import load_data
import reduce_mem_usage as rm
from utils import get_logger

logger = get_logger()
file_path = os.path.dirname(__file__)

CONFIG_FILE = '../config/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
INPUT_DIR_NAME = yml['SETTING']['INPUT_DIR_NAME']


def create_pickle(train, test, specs, train_labels):
    logger.info('save pickle file')
    train.to_pickle(file_path + INPUT_DIR_NAME + 'train.pkl')
    test.to_pickle(file_path + INPUT_DIR_NAME + 'test.pkl')
    specs.to_pickle(file_path + INPUT_DIR_NAME + 'specs.pkl')
    train_labels.to_pickle(file_path + INPUT_DIR_NAME + 'train_labels.pkl')


if __name__ == '__main__':
    train = rm.reduce_mem_usage(load_data.read_train())
    test = rm.reduce_mem_usage(load_data.read_test())
    specs = rm.reduce_mem_usage(load_data.read_specs())
    train_labels = rm.reduce_mem_usage(load_data.read_train_labels())
    create_pickle(train, test, specs, train_labels)
