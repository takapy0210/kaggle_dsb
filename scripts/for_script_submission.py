
from sklearn.metrics import log_loss, mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from typing import Callable, List, Optional, Tuple, Union
from tqdm import tqdm_notebook
from tqdm import tqdm
from collections import Counter
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
import os
import datetime
import yaml
import json
import collections as cl
import warnings
import joblib

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

FEATURE_DIR_NAME = '../input/data-science-bowl-2019/'
MODEL_DIR_NAME = ''

class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def dump_df_pickle(cls, df, path):
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        return pd.read_pickle(path)






def read_data_all(mode='prd') -> List[pd.DataFrame]:
    """
    パイプライン実行に必要なデータの読み込み。mainからはこれだけ呼べば良くする。
    mode='dev'だとtrainとtestで行数を制限して読み込みするので高速に動作確認できる
    """
    data = [
        read_train(mode),
        read_test(mode),
        # read_specs(mode),
        read_train_labels(mode),
        read_submission()
    ]
    return data


def read_train(mode='prd') -> pd.DataFrame:
    if mode=='pkl':
        return pd.read_pickle(('../input/data-science-bowl-2019/train.pkl'))
    N_ROW = None if mode=='prd' else 100000
    return pd.read_csv(('../input/data-science-bowl-2019/train.csv'), nrows=N_ROW)


def read_test(mode='prd') -> pd.DataFrame:
    if mode=='pkl':
        return pd.read_pickle(('../input/data-science-bowl-2019/test.pkl'))
    N_ROW = None if mode=='prd' else 100000
    return pd.read_csv(('../input/data-science-bowl-2019/test.csv'), nrows=N_ROW)


def read_specs(mode='prd') -> pd.DataFrame:
    if mode=='pkl':
        return pd.read_pickle(('../input/data-science-bowl-2019/specs.pkl'))
    return pd.read_csv(('../input/data-science-bowl-2019/specs.csv'))


def read_train_labels(mode='prd') -> pd.DataFrame:
    if mode=='pkl':
        return pd.read_pickle(('../input/data-science-bowl-2019/train_labels.pkl'))
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


def encode_title(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
    # return train, test


def get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
            activities_labels, assess_titles, list_of_event_id, all_title_event_code, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()}
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session

        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]

        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens:
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())

            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter

        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments


def get_train_and_test(train, test,
                        win_code, list_of_user_activities, list_of_event_code,
                        activities_labels, assess_titles, list_of_event_id, all_title_event_code):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        # user_sampleに1つのins_idの全てのデータが入っている
        compiled_train += get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                activities_labels, assess_titles, list_of_event_id, all_title_event_code)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                activities_labels, assess_titles, list_of_event_id, all_title_event_code, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    return reduce_train, reduce_test


def preprocess(reduce_train, reduce_test, assess_titles):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021,
                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035,
                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

    return reduce_train, reduce_test, features






def staging_train(train_labels: pd.DataFrame, features: pd.DataFrame, save=False) -> (pd.DataFrame, pd.Series):
    """
    加工したデータからモデルにインプットできる形のデータに変換する
    """
    train_labels = train_labels.set_index('game_session').merge(features, how='left', left_index=True, right_index=True)
    train_labels = _drop_columns_train(train_labels)
    X_train = train_labels.drop('accuracy_group', axis=1)
    X_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため
    y_train = train_labels['accuracy_group']
    return X_train, y_train


def staging_test(test: pd.DataFrame, features: pd.DataFrame, submission: pd.DataFrame, save=False) -> pd.DataFrame:
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


def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def eval_qwk_lgb(y_pred, data):
    """
    Fast cappa eval function for lgb.
    """
    y_true = data.get_label()
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'cappa', qwk(y_true, y_pred), True


def eval_qwk_lgb_regr(y_pred, data):
    """
    Fast cappa eval function for lgb.
    """
    y_true = data.get_label()
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3

    return 'cappa', qwk(y_true, y_pred), True


class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """コンストラクタ
        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
                va_x: Optional[pd.DataFrame] = None,
                va_y: Optional[pd.Series] = None) -> None:
        """モデルの学習を行い、学習済のモデルを保存する
        :param tr_x: 学習データの特徴量
        :param tr_y: 学習データの目的変数
        :param va_x: バリデーションデータの特徴量
        :param va_y: バリデーションデータの目的変数
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """学習済のモデルでの予測値を返す
        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """モデルの保存を行う"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """モデルの読み込みを行う"""
        pass

# 各foldのモデルを保存する配列
model_array = []


class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        dtrain = lgb.Dataset(tr_x, tr_y)

        if validation:
            dvalid = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')
        verbose_eval = params.pop('verbose_eval')
        eval_metric = eval_qwk_lgb_regr

        # 学習
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = lgb.train(
                                params,
                                dtrain,
                                num_boost_round=num_round,
                                valid_sets=(dtrain, dvalid),
                                early_stopping_rounds=early_stopping_rounds,
                                feval=eval_metric,
                                verbose_eval=verbose_eval
                                )
            model_array.append(self.model)

        else:
            watchlist = [(dtrain, 'train')]
            self.model = lgb.train(params, dtrain, num_round, evals=watchlist)
            model_array.append(self.model)

    # shapを計算しないver
    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    # shapを計算するver
    def predict_and_shap(self, te_x, shap_sampling):
        fold_importance = shap.TreeExplainer(self.model).shap_values(te_x[:shap_sampling])
        valid_prediticion = self.model.predict(te_x, num_iteration=self.model.best_iteration)
        return valid_prediticion, fold_importance

    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print('modelの保存先:{}'.format(model_path))
        Util.dump(self.model, model_path)

    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        print('modelのロード先:{}'.format(model_path))
        self.model = Util.load(model_path)

    @classmethod
    def calc_feature_importance(self, dir_name, run_name, features):
        """feature importanceの計算
        """

        val_split = model_array[0].feature_importance(importance_type='split')
        val_gain = model_array[0].feature_importance(importance_type='gain')
        val_split = pd.Series(val_split)
        val_gain = pd.Series(val_gain)

        for m in model_array[1:]:
            s = pd.Series(m.feature_importance(importance_type='split'))
            val_split = pd.concat([val_split, s], axis=1)
            s = pd.Series(m.feature_importance(importance_type='gain'))
            val_gain = pd.concat([val_gain, s], axis=1)

        # -----------
        # splitの計算
        # -----------
        # 各foldの平均を算出
        val_mean = val_split.mean(axis=1)
        val_mean = val_mean.values
        importance_df_mean = pd.DataFrame(val_mean, index=features, columns=['importance']).sort_values('importance')

        # 各foldの標準偏差を算出
        val_std = val_split.std(axis=1)
        val_std = val_std.values
        importance_df_std = pd.DataFrame(val_std, index=features, columns=['importance']).sort_values('importance')

        # マージ
        df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True, suffixes=['_mean', '_std'])

        df['coef_of_var'] = df['importance_std'] / df['importance_mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('importance_mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize=(10, 30))
        plt.tick_params(labelsize=12)  # 図のラベルのfontサイズ
        plt.tight_layout()

        # 棒グラフを出力
        ax1.set_title('feature importance split')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df.index, df['importance_mean'], label='importance_mean',  align="center", alpha=0.6)
        ax1.barh(df.index, df['importance_std'], label='importance_std',  align="center", alpha=0.6)

        # 折れ線グラフを出力
        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df.index, linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.94), loc='upper right', borderaxespad=0.5, fontsize=12)

        # グリッド表示(ax1のみ)
        ax1.grid(True)
        ax2.grid(False)

        plt.savefig(dir_name + run_name + '_fi_split.png', dpi=300, bbox_inches="tight")
        plt.close()

        # -----------
        # gainの計算
        # -----------
        # 各foldの平均を算出
        val_mean = val_gain.mean(axis=1)
        val_mean = val_mean.values
        importance_df_mean = pd.DataFrame(val_mean, index=features, columns=['importance']).sort_values('importance')

        # 各foldの標準偏差を算出
        val_std = val_gain.std(axis=1)
        val_std = val_std.values
        importance_df_std = pd.DataFrame(val_std, index=features, columns=['importance']).sort_values('importance')

        # マージ
        df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True, suffixes=['_mean', '_std'])

        # 変動係数を算出
        df['coef_of_var'] = df['importance_std'] / df['importance_mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('importance_mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize=(10, 30))
        plt.tick_params(labelsize=12)  # 図のラベルのfontサイズ
        plt.tight_layout()

        # 棒グラフを出力
        ax1.set_title('feature importance gain')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df.index, df['importance_mean'], label='importance_mean',  align="center", alpha=0.6)
        ax1.barh(df.index, df['importance_std'], label='importance_std',  align="center", alpha=0.6)

        # 折れ線グラフを出力
        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df.index, linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.94), loc='upper right', borderaxespad=0.5, fontsize=12)

        # グリッド表示(ax1のみ)
        ax1.grid(True)
        ax2.grid(False)

        plt.savefig(dir_name + run_name + '_fi_gain.png', dpi=300, bbox_inches="tight")
        plt.close()

# 定数
shap_sampling = 10000
corr_sampling = 10000

class Runner:

    def __init__(self
                , run_name: str
                , model_cls: Callable[[str, dict], Model]
                , setting: dict
                , params: dict
                , cv: dict
                , feature_dir_name: str
                , model_dir_name: str
                , x_train=None
                , y_train=None
                , x_test=None):
        """コンストラクタ
        :run_name: runの名前
        :model_cls: モデルのクラス
        :setting: 設定リスト
        :params: ハイパーパラメータ
        :cv: CVの設定
        :feature_dir_name: 特徴量を読み込むディレクトリ
        :model_dir_name: 学習に使用するファイルを保存するディレクトリ
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.target = setting.get('target')
        self.calc_shap = setting.get('calc_shap')
        self.save_train_pred = setting.get('save_train_pred')
        self.params = params
        self.cv_method = cv.get('method')
        self.n_splits = cv.get('n_splits')
        self.random_state = cv.get('random_state')
        self.shuffle = cv.get('shuffle')
        self.cv_target_column = cv.get('cv_target')
        self.feature_dir_name = feature_dir_name
        self.model_dir_name = model_dir_name
        self.remove_train_index = None # trainデータからデータを絞り込む際に使用する。除外するindexを保持。
        self.train_x = self.load_x_train() if x_train is None else x_train
        self.train_y = self.load_y_train() if y_train is None else y_train
        self.test_x = x_test
        self.out_dir_name = model_dir_name + run_name + '/'
        if self.calc_shap:
            self.shap_values = np.zeros(self.train_x.shape)
        self.metrics = qwk


    def shap_feature_importance(self) -> None:
        """計算したshap値を可視化して保存する
        """
        all_columns = self.train_x.columns.values.tolist() + [self.target]
        ma_shap = pd.DataFrame(sorted(zip(abs(self.shap_values).mean(axis=0), all_columns), reverse=True),
                        columns=['Mean Abs Shapley', 'Feature']).set_index('Feature')
        ma_shap = ma_shap.sort_values('Mean Abs Shapley', ascending=True)

        fig = plt.figure(figsize = (8,25))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        ax = fig.add_subplot(1,1,1)
        ax.set_title('shap value')
        ax.barh(ma_shap.index, ma_shap['Mean Abs Shapley'] , label='Mean Abs Shapley',  align="center", alpha=0.8)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=0, fontsize=10)
        ax.legend(loc = 'upper left')
        plt.savefig(self.out_dir_name + self.run_name + '_shap.png', dpi=300, bbox_inches="tight")
        plt.close()


    def get_feature_name(self):
        """ 学習に使用する特徴量を返却
        """
        features_list = self.train_x.columns.values.tolist()
        return [str(n) for n in features_list]


    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != 'all'
        train_x = self.train_x.copy()
        train_y = self.train_y.copy()

        if validation:

            # 学習データ・バリデーションデータのindexを取得
            if self.cv_method == 'KFold':
                tr_idx, va_idx = self.load_index_k_fold(i_fold)
            elif self.cv_method == 'StratifiedKFold':
                tr_idx, va_idx = self.load_index_sk_fold(i_fold)
            elif self.cv_method == 'GroupKFold':
                tr_idx, va_idx = self.load_index_gk_fold(i_fold)
            else:
                print('CVメソッドが正しくないため終了します')
                sys.exit(0)

            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # バリデーションデータへの予測・評価を行う
            if self.calc_shap:
                va_pred, self.shap_values[va_idx[:shap_sampling]] = model.predict_and_shap(va_x, shap_sampling)
            else:
                va_pred = model.predict(va_x)

            score = self.metrics(va_y, va_pred)

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None


    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """

        scores = [] # 各foldのscoreを保存
        va_idxes = [] # 各foldのvalidationデータのindexを保存
        preds = [] # 各foldの推論結果を保存

        # 各foldで学習を行う
        for i_fold in range(self.n_splits):
            # 学習を行う
            model, va_idx, va_pred, score = self.train_fold(i_fold)

            # モデルを保存する
            model.save_model(self.out_dir_name)

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]


        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(pd.DataFrame(preds), self.out_dir_name + f'.{self.run_name}-train.pkl')

        # 評価結果の保存

        # shap feature importanceデータの保存
        if self.calc_shap:
            self.shap_feature_importance()


    def run_predict_cv(self, is_kernel=False) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        test_x = self.load_x_test() if self.test_x is None else self.test_x
        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_splits):
            model = self.build_model(i_fold)
            model.load_model(self.out_dir_name)
            pred = model.predict(test_x)
            preds.append(pred)

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)
        pred_avg[pred_avg <= 1.12232214] = 0
        pred_avg[np.where(np.logical_and(pred_avg > 1.12232214, pred_avg <= 1.73925866))] = 1
        pred_avg[np.where(np.logical_and(pred_avg > 1.73925866, pred_avg <= 2.22506454))] = 2
        pred_avg[pred_avg > 2.22506454] = 3

        if is_kernel:
            return pd.Series(pred_avg)
        else:
            # 推論結果の保存（submit対象データ）
            Util.dump_df_pickle(pd.DataFrame(pred_avg), self.out_dir_name + f'{self.run_name}-pred.pkl')


        return None


    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存する"""

        # 学習データ全てで学習を行う
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model(self.out_dir_name)



    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う
        あらかじめrun_train_allを実行しておく必要がある
        """

        test_x = self.load_x_test() if self.test_x is None else self.test_x

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model(self.out_dir_name)
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')



    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-fold{i_fold}'
        return self.model_cls(run_fold_name, self.params)


    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む
        列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        :return: 学習データの特徴量
        """
        # 学習データの読込を行う
        df = pd.read_pickle(self.feature_dir_name + 'X_train.pkl')
        return df


    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む
        対数変換や使用するデータを削除する場合には、このメソッドの修正が必要
        :return: 学習データの目的変数
        """
        # 目的変数の読込を行う
        train_y = pd.read_pickle(self.feature_dir_name + 'y_train.pkl')
        return pd.Series(train_y)


    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む
        :return: テストデータの特徴量
        """
        df = pd.read_pickle(self.feature_dir_name + 'X_test.pkl')
        return df


    def load_stratify_or_group_target(self) -> pd.Series:
        """
        groupKFoldで同じグループが異なる分割パターンに出現しないようにデータセットを分割したい対象カラムを取得する
        または、StratifiedKFoldで分布の比率を維持したいカラムを取得する
        :return: 分布の比率を維持したいデータの特徴量
        """
        df = pd.read_pickle(self.feature_dir_name + self.cv_target_column + '_train.pkl')
        return pd.Series(df[self.cv_target_column])


    def load_index_k_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        train_y = self.train_y
        dummy_x = np.zeros(len(train_y))
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(dummy_x))[i_fold]


    def load_index_sk_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        stratify_data = self.load_stratify_or_group_target() # 分布の比率を維持したいデータの対象
        dummy_x = np.zeros(len(stratify_data))
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(dummy_x, stratify_data))[i_fold]


    def load_index_gk_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        group_data = self.load_stratify_or_group_target()
        train_y = self.train_y
        dummy_x = np.zeros(len(group_data))
        kf = GroupKFold(n_splits=self.n_splits)
        return list(kf.split(dummy_x, train_y, groups=group_data))[i_fold]



now = datetime.datetime.now()
suffix = now.strftime("_%m%d_%H%M")
key_list = ['use_features', 'model_params', 'cv', 'setting']




def confirm(text=None):
    # 通常の実行確認
    print(str(text) + '実行しますか？[Y/n]')
    x = input('>> ')
    if x != 'Y':
        print('終了します')
        sys.exit(0)


def exist_check(path, run_name):
    """学習ファイルの存在チェックと実行確認
    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)


def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def save_model_config(key_list, value_list, dir_name, run_name):
    """jsonファイル生成
    """
    ys = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        ys[v] = data

    fw = open(dir_name + run_name + '_param.json', 'w')
    json.dump(ys, fw, indent=4, default=set_default)


def set_default(obj):
    """json出力の際にset型のオブジェクトをリストに変更する
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def main(mode='prd', create_features=True) -> str:


    if create_features:
        # データ生成
        train, test, train_labels, submission = read_data_all(mode)
        # features_train = create_feature(train)
        # features_test = create_feature(test)
        # _, _ = staging_train(train_labels, features_train, save=True)
        # _ = staging_test(test, features_test, submission, save=True)

        features_train, features_test, win_code, list_of_user_activities, list_of_event_code, \
            activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test)
        features_train, features_test = get_train_and_test(features_train, features_test,
                                        win_code, list_of_user_activities, list_of_event_code,
                                        activities_labels, assess_titles, list_of_event_id, all_title_event_code)
        reduce_train, reduce_test, _ = preprocess(features_train, features_test, assess_titles)

        cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
        cols_to_drop = [col for col in cols_to_drop if col in reduce_train.columns]
        X_train = reduce_train.drop(cols_to_drop, axis=1)
        X_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため
        y_train = reduce_train['accuracy_group']

        cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
        cols_to_drop = [col for col in cols_to_drop if col in reduce_test.columns]
        X_test = reduce_test.drop(cols_to_drop, axis=1)
        X_test.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため



    # CVの設定.methodは[KFold, StratifiedKFold ,GroupKFold]から選択可能
    # CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    # StratifiedKFold or GroupKFoldの場合は、cv_targetに対象カラム名を設定する
    cv = {
        'method': 'KFold',
        'n_splits': 3,
        'random_state': 42,
        'shuffle': True,
        'cv_target': 'hoge'
    }

    # ######################################################
    # 学習・推論 LightGBM ###################################

    # run nameの設定
    run_name = 'lgb'
    run_name = run_name + suffix
    dir_name = MODEL_DIR_NAME + run_name + '/'

    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # 諸々の設定
    setting = {
        'run_name': run_name,  # run名
        'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'target': 'accuracy_group',  # 目的変数
        'calc_shap': False,  # shap値を計算するか否か
        'save_train_pred': False  # trainデータでの推論値を保存するか否か（trainデータでの推論値を特徴量として加えたい場合はTrueに設定する）
    }

    # モデルのパラメータ
    model_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'subsample': 0.75,
        'subsample_freq': 1,
        'feature_fraction': 0.9,
        'num_leaves': 40,
        'max_depth': 8,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'num_round': 500,
        'early_stopping_rounds': 500,
        'verbose': -1,
        'verbose_eval': 500,
        'random_state': 999
    }

    runner = Runner(run_name, ModelLGB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME, X_train, y_train, X_test)

    use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

    # モデルのconfigをjsonで保存
    value_list = [use_feature_name, model_params, cv, setting]
    save_model_config(key_list, value_list, dir_name, run_name)

    if cv.get('method') == 'None':
        # TODO: こちらも動くように修正する
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 推論
    else:
        runner.run_train_cv()  # 学習
        _pred = runner.run_predict_cv(is_kernel=True)  # 推論

    if _pred is not None:
        # _predに値が存在する場合（kaggleでのカーネル実行）はsubの作成
        submission[setting.get('target')] = _pred
        submission.to_csv('submission.csv', index=False)
    else:
        # ローカルでの実行
        Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    return 'Success!'


if __name__ == '__main__':
    main()
