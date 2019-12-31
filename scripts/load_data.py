from typing import List

import os
import pandas as pd

from util import get_logger

logger = get_logger()
file_path = os.path.dirname(__file__)


def read_data_all(mode='prd') -> List[pd.DataFrame]:
    """
    パイプライン実行に必要なデータの読み込み。mainからはこれだけ呼べば良くする。
    mode='dev'だとtrainとtestで行数を制限して読み込みするので高速に動作確認できる
    """
    data = [
        read_train(mode),
        read_test(mode),
        # specs = read_specs(mode),
        read_train_labels(mode),
        read_submission()
    ]
    logger.info('Reading data finished')
    return data


def read_train(mode='prd') -> pd.DataFrame:
    logger.info('Reading train.csv')
    if mode=='pkl':
        return pd.read_pickle(os.path.join(file_path, '../data/input/train.pkl'))
    N_ROW = None if mode=='prd' else 100000
    return pd.read_csv(os.path.join(file_path, '../data/input/train.csv'), nrows=N_ROW)


def read_test(mode='prd') -> pd.DataFrame:
    logger.info('Reading test.csv')
    if mode=='pkl':
        return pd.read_pickle(os.path.join(file_path, '../data/input/test.pkl'))
    N_ROW = None if mode=='prd' else 100000
    return pd.read_csv(os.path.join(file_path, '../data/input/test.csv'), nrows=N_ROW)


def read_specs(mode='prd') -> pd.DataFrame:
    logger.info('Reading specs.csv')
    if mode=='pkl':
        return pd.read_pickle(os.path.join(file_path, '../data/input/specs.pkl'))
    return pd.read_csv(os.path.join(file_path, '../data/input/specs.csv'))


def read_train_labels(mode='prd') -> pd.DataFrame:
    logger.info('Reading train_labels.csv')
    if mode=='pkl':
        return pd.read_pickle(os.path.join(file_path, '../data/input/train_labels.pkl'))
    return pd.read_csv(os.path.join(file_path, '../data/input/train_labels.csv'))


def read_submission() -> pd.DataFrame:
    logger.info('Reading sample_submission.csv')
    return pd.read_csv(os.path.join(file_path, '../data/input/sample_submission.csv'))
