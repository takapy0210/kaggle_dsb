
import pandas as pd

from utils import get_logger

logger = get_logger()


def create_feature(df: pd.DataFrame) -> pd.DataFrame:
    type_features = _type_feature(df)
    event_code_features = _event_code_feature(df)
    title_features = _title_feature(df)

    features = pd.concat([
        type_features,
        event_code_features,
        title_features
    ], axis=1)
    logger.info('Create feature finished')
    logger.info(features.head())
    logger.info('Shape of feature: {}'.format(features.shape))
    return features


def _type_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create type features')
    count_features = df.groupby(['game_session', 'type'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: col+'_counts' for col in count_features.columns}, inplace=True)
    type_features = pd.concat([count_features], axis=1)
    return type_features


def _event_code_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create event_code features')
    count_features = df.groupby(['game_session', 'event_code'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: str(col)+'_counts' for col in count_features.columns}, inplace=True)
    event_code_features = pd.concat([count_features], axis=1)
    return event_code_features


def _title_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create title features')
    count_features = df.groupby(['game_session', 'title'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: col+'_counts' for col in count_features.columns}, inplace=True)
    title_features = pd.concat([count_features], axis=1)
    return title_features
