import os

import pandas as pd
import yaml

from util import get_logger

logger = get_logger()

CONFIG_FILE = '../config/config.yaml'
file_path = os.path.dirname(__file__)
with open(os.path.join(file_path, CONFIG_FILE)) as file:
    yml = yaml.load(file)
FEATURE_DIR_NAME = os.path.join(file_path, yml['SETTING']['OUTPUT_DIR_NAME'])


def staging_train(train_labels: pd.DataFrame, features: pd.DataFrame, save=False) -> (pd.DataFrame, pd.Series):
    """
    加工したデータからモデルにインプットできる形のデータに変換する
    """
    train_labels = train_labels.set_index('game_session').merge(features, how='left', left_index=True, right_index=True)
    train_labels = _drop_columns_train(train_labels)
    X_train = train_labels.drop('accuracy_group', axis=1)
    X_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため
    y_train = train_labels['accuracy_group']
    if save:
        # pd.concat([X_train, y_train], axis=1).to_pickle(FEATURE_DIR_NAME + 'train.pkl')
        X_train.to_pickle(FEATURE_DIR_NAME + 'X_train.pkl')
        y_train.to_pickle(FEATURE_DIR_NAME + 'y_train.pkl')
    logger.info(train_labels.head())
    return X_train, y_train


def staging_test(test: pd.DataFrame, features: pd.DataFrame, submission: pd.DataFrame, save=False) -> pd.DataFrame:
    """
    加工したデータからモデルにインプットできる形のデータに変換する
    """
    target_session = test.loc[test.groupby('installation_id')['timestamp'].idxmax(), ['installation_id', 'game_session']]
    target_session = target_session.set_index('game_session').merge(features, how='left', left_index=True, right_index=True)
    target_session = submission.merge(target_session, how='left', on='installation_id')  # submissionファイルの順番と揃える
    target_session = _drop_columns_test(target_session)
    if save:
        target_session.to_pickle(FEATURE_DIR_NAME + 'X_test.pkl')
    logger.info(target_session.head())
    return target_session


def _drop_columns_train(df):
    """
    不要なカラムの削除
    """
    df.drop([
        'installation_id',
        'title',
        'num_correct',
        'num_incorrect',
        'accuracy',
        'assignment_False_counts',
        'assignment_True_counts'
    ], axis=1, inplace=True)
    return df


def _drop_columns_test(df):
    """
    不要なカラムの削除
    """
    df.drop([
        'installation_id',
        'accuracy_group',
        'assignment_False_counts',
        'assignment_True_counts'
    ], axis=1, inplace=True)
    return df
