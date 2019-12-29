
import pandas as pd

from utils import get_logger

logger = get_logger()


def create_feature(df: pd.DataFrame) -> pd.DataFrame:
    type_features = _type_feature(df)
    event_code_features = _event_code_feature(df)
    title_features = _title_feature(df)
    datetime_features = _datetime_feature(df)
    game_time_features = _game_time_feature(df)
    attempt_features = _attempt_features(df)

    features = pd.concat([
        type_features,
        event_code_features,
        title_features,
        datetime_features,
        game_time_features,
        attempt_features
    ], axis=1)
    logger.info('Create feature finished')
    logger.info(features.head())
    logger.info('Shape of feature: {}'.format(features.shape))
    return features


def _type_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create type features')
    count_features = df.groupby(['game_session', 'type'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'type_'+col+'_counts' for col in count_features.columns}, inplace=True)
    type_features = pd.concat([count_features], axis=1)
    return type_features


def _event_code_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create event_code features')
    count_features = df.groupby(['game_session', 'event_code'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'event_code_'+str(col)+'_counts' for col in count_features.columns}, inplace=True)
    event_code_features = pd.concat([count_features], axis=1)
    return event_code_features


def _title_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create title features')
    count_features = df.groupby(['game_session', 'title'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'title_'+col+'_counts' for col in count_features.columns}, inplace=True)
    title_features = pd.concat([count_features], axis=1)
    return title_features


def _datetime_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create datetime features')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek

    hour_count_features = df.groupby(['game_session', 'hour'])['event_id'].count().unstack().fillna(0)
    hour_count_features.rename(columns={col: 'hour_'+str(col)+'_counts' for col in hour_count_features.columns}, inplace=True)
    weekday_count_features = df.groupby(['game_session', 'weekday'])['event_id'].count().unstack().fillna(0)
    weekday_count_features.rename(columns={col: 'weekday_'+str(col)+'_counts' for col in weekday_count_features.columns}, inplace=True)
    datetime_features = pd.concat([hour_count_features, weekday_count_features], axis=1)
    return datetime_features


def _game_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Create gametime features')
    agg_features = df.groupby(['game_session'])['game_time'].agg(['sum', 'mean'])
    agg_features.rename(columns={col: 'game_time_'+col for col in agg_features.columns}, inplace=True)
    game_time_features = pd.concat([agg_features], axis=1)
    return game_time_features


def _attempt_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Creaet attempt features')
    all_attempts = df.loc[
        (df.type == "Assessment") & (df.title == 'Bird Measurer (Assessment)') & (df.event_code == 4110) |
        (df.type == "Assessment") & (df.title != 'Bird Measurer (Assessment)') & (df.event_code == 4100)
    ]
    all_attempts['pass_assessment'] = all_attempts['event_data'].str.contains('true')
    count_features = all_attempts.groupby(['game_session', 'pass_assessment'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'assignment_'+str(col)+'_counts' for col in count_features.columns}, inplace=True)
    attempt_features = pd.concat([count_features]).fillna(0)
    return attempt_features
