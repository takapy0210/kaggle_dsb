from typing import List

import os
import pandas as pd

from utils import get_logger

logger = get_logger()
file_path = os.path.dirname(__file__)


def read_data_all(dev=False, use_pkl=False) -> List[pd.DataFrame]:
    """
    パイプライン実行に必要なデータの読み込み。mainからはこれだけ呼べば良くする。
    devがTrueだとtrainとtestで行数を制限して読み込みするので高速に動作確認できる
    """
    data = [
        read_train(dev, use_pkl),
        read_test(dev, use_pkl),
        # specs = read_specs(use_pkl),
        read_train_labels(use_pkl),
        read_submission()
    ]
    logger.info('Reading data finished')
    return data


def read_train(dev=False, use_pkl=False) -> pd.DataFrame:
    logger.info('Reading train.csv')
    N_ROW = None if not dev else 100000
    if not dev & use_pkl:
        return pd.read_pickle(os.path.join(file_path, '../data/input/train.pkl'))
    return pd.read_csv(os.path.join(file_path, '../data/input/train.csv'), nrows=N_ROW)


def read_test(dev=False, use_pkl=False) -> pd.DataFrame:
    logger.info('Reading test.csv')
    N_ROW = None if not dev else 100000
    if not dev & use_pkl:
        return pd.read_pickle(os.path.join(file_path, '../data/input/test.pkl'))
    return pd.read_csv(os.path.join(file_path, '../data/input/test.csv'), nrows=N_ROW)


def read_specs(use_pkl=False) -> pd.DataFrame:
    logger.info('Reading specs.csv')
    if use_pkl:
        return pd.read_pickle(os.path.join(file_path, '../data/input/specs.pkl'))
    return pd.read_csv(os.path.join(file_path, '../data/input/specs.csv'))


def read_train_labels(use_pkl=False) -> pd.DataFrame:
    logger.info('Reading train_labels.csv')
    if use_pkl:
        return pd.read_pickle(os.path.join(file_path, '../data/input/train_labels.pkl'))
    return pd.read_csv(os.path.join(file_path, '../data/input/train_labels.csv'))


def read_submission() -> pd.DataFrame:
    logger.info('Reading sample_submission.csv')
    return pd.read_csv(os.path.join(file_path, '../data/input/sample_submission.csv'))
