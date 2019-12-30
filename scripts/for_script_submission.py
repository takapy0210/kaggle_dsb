
from typing import List

import pandas as pd
import lightgbm as lgb






def read_data_all(dev=False) -> List[pd.DataFrame]:
    """
    パイプライン実行に必要なデータの読み込み。mainからはこれだけ呼べば良くする。
    devがTrueだとtrainとtestで行数を制限して読み込みするので高速に動作確認できる
    """
    data = [
        read_train(dev),
        read_test(dev),
        # specs = read_specs(),
        read_train_labels(),
        read_submission()
    ]
    return data


def read_train(dev=False) -> pd.DataFrame:
    N_ROW = None if not dev else 100000
    return pd.read_csv(('../input/data-science-bowl-2019/train.csv'), nrows=N_ROW)


def read_test(dev=False) -> pd.DataFrame:
    N_ROW = None if not dev else 100000
    return pd.read_csv(('../input/data-science-bowl-2019/test.csv'), nrows=N_ROW)


def read_specs() -> pd.DataFrame:
    return pd.read_csv(('../input/data-science-bowl-2019/specs.csv'))


def read_train_labels() -> pd.DataFrame:
    return pd.read_csv(('../input/data-science-bowl-2019/train_labels.csv'))


def read_submission() -> pd.DataFrame:
    return pd.read_csv(('../input/data-science-bowl-2019/sample_submission.csv'))




def create_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    全ての特徴を生成する。mainからはこれだけ呼べば良くする。
    """
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
    return features


def _type_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    typeに関する特徴（ex. Assessment, Clipなど）
    """
    count_features = df.groupby(['game_session', 'type'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'type_'+col+'_counts' for col in count_features.columns}, inplace=True)
    type_features = pd.concat([count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return type_features


def _event_code_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    event_codeに関する特徴
    """
    count_features = df.groupby(['game_session', 'event_code'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'event_code_'+str(col)+'_counts' for col in count_features.columns}, inplace=True)
    event_code_features = pd.concat([count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return event_code_features


def _title_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    titleに関する特徴（ex. Sandcastle Builder (Activity)など）
    """
    count_features = df.groupby(['game_session', 'title'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'title_'+col+'_counts' for col in count_features.columns}, inplace=True)
    title_features = pd.concat([count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return title_features


def _datetime_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    timestampに関する特徴
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek

    hour_count_features = df.groupby(['game_session', 'hour'])['event_id'].count().unstack().fillna(0)
    hour_count_features.rename(columns={col: 'hour_'+str(col)+'_counts' for col in hour_count_features.columns}, inplace=True)
    weekday_count_features = df.groupby(['game_session', 'weekday'])['event_id'].count().unstack().fillna(0)
    weekday_count_features.rename(columns={col: 'weekday_'+str(col)+'_counts' for col in weekday_count_features.columns}, inplace=True)
    datetime_features = pd.concat([hour_count_features, weekday_count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return datetime_features


def _game_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    game_timeに関する特徴
    """
    agg_features = df.groupby(['game_session'])['game_time'].agg(['sum', 'mean'])
    agg_features.rename(columns={col: 'game_time_'+col for col in agg_features.columns}, inplace=True)
    game_time_features = pd.concat([agg_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return game_time_features


def _attempt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assessmentの試行に関する特徴
    """
    all_attempts = df.loc[
        (df.type == "Assessment") & (df.title == 'Bird Measurer (Assessment)') & (df.event_code == 4110) |
        (df.type == "Assessment") & (df.title != 'Bird Measurer (Assessment)') & (df.event_code == 4100)
    ]
    all_attempts['pass_assessment'] = all_attempts['event_data'].str.contains('true')
    count_features = all_attempts.groupby(['game_session', 'pass_assessment'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'assignment_'+str(col)+'_counts' for col in count_features.columns}, inplace=True)
    attempt_features = pd.concat([count_features]).fillna(0)  # 後々コメントアウトで必要な特徴を調整できるように
    return attempt_features




def staging_train(train_labels: pd.DataFrame, features: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    加工したデータからモデルにインプットできる形のデータに変換する
    """
    train_labels = train_labels.set_index('game_session').merge(features, how='left', left_index=True, right_index=True)
    train_labels = _drop_columns_train(train_labels)
    X_train = train_labels.drop('accuracy_group', axis=1)
    X_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため
    y_train = train_labels['accuracy_group']
    return X_train, y_train


def staging_test(test: pd.DataFrame, features: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    """
    加工したデータからモデルにインプットできる形のデータに変換する
    """
    target_session = test.loc[test.groupby('installation_id')['timestamp'].idxmax(), ['installation_id', 'game_session']]
    target_session = target_session.set_index('game_session').merge(features, how='left', left_index=True, right_index=True)
    target_session = submission.merge(target_session, how='left', on='installation_id')  # submissionファイルの順番と揃える
    target_session = _drop_columns_test(target_session)
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

params = {
    'n_estimators': 2000,
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'subsample': 0.75,
    'subsample_freq': 1,
    'learning_rate': 0.04,
    'feature_fraction': 0.9,
    'max_depth': 15,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'verbose': 100,
    # 'early_stopping_rounds': 100,
    'eval_metric': 'cappa'
}


def training(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    """
    学習用のデータを使ってモデルを学習
    TODO: validationを入れる
    """
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model





def main(dev=False) -> str:
    """
    データ読み込み -> submissionファイルの出力までのパイプライン実行を担う
    """
    train, test, train_labels, submission = read_data_all(dev)
    features_train = create_feature(train)
    X_train, y_train = staging_train(train_labels, features_train)
    model = training(X_train, y_train)
    features_test = create_feature(test)
    X_test = staging_test(test, features_test, submission)
    prediction = model.predict(X_test)  # TODO: 出力が適切な形になってない
    submission['accuracy_group'] = prediction
    submission.to_csv('submission.csv', index=False)
    return 'Success!'


if __name__ == '__main__':
    main()
