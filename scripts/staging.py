import pandas as pd

from utils import get_logger

logger = get_logger()


def staging_train(train_labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    train_labels = train_labels.set_index('game_session').merge(features, how='left', left_index=True, right_index=True)
    train_labels = _drop_columns_train(train_labels)
    logger.info(train_labels.head())
    return train_labels


def staging_test(test: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    target_session = test.loc[test.groupby('installation_id')['timestamp'].idxmax(), ['installation_id', 'game_session']]
    target_session = target_session.set_index('game_session').merge(features, how='left', left_index=True, right_index=True)
    target_session = _drop_columns_test(target_session)
    logger.info(target_session.head())
    return target_session


def _drop_columns_train(df):
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
    df.drop([
        'installation_id',
        'assignment_False_counts',
        'assignment_True_counts'
    ], axis=1, inplace=True)
    return df
