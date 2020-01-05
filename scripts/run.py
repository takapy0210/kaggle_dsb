import sys
import os
import datetime
import yaml
import json
import collections as cl
import warnings
import fire
import pandas as pd

from load_data import read_data_all
from create_feature import create_feature, encode_title, get_train_and_test, preprocess, create_agg
from staging import staging_train, staging_test
from model_lgb import ModelLGB
from model_cb import ModelCB
from runner import Runner
from util import Submission

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

now = datetime.datetime.now()
suffix = now.strftime("_%m%d_%H%M")
key_list = ['use_features', 'model_params', 'cv', 'setting']

CONFIG_FILE = '../config/config.yaml'
file_path = os.path.dirname(__file__)

with open(os.path.join(file_path, CONFIG_FILE)) as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = os.path.join(file_path, yml['SETTING']['MODEL_DIR_NAME'])
FEATURE_DIR_NAME = os.path.join(file_path, yml['SETTING']['OUTPUT_DIR_NAME'])


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


def main(mode='prd', create_features=True, model_type='lgb') -> str:

    confirm('mode:{}, create_feature:{} '.format(str(mode), str(create_features)))

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

        # reduce_train, reduce_test = create_agg(reduce_train, reduce_test)

        cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
        cols_to_drop = [col for col in cols_to_drop if col in reduce_train.columns]
        X_train = reduce_train.drop(cols_to_drop, axis=1)
        X_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため
        y_train = reduce_train['accuracy_group']

        cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
        cols_to_drop = [col for col in cols_to_drop if col in reduce_test.columns]
        X_test = reduce_test.drop(cols_to_drop, axis=1)
        X_test.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため

        X_train.to_pickle(FEATURE_DIR_NAME + 'X_train.pkl')
        y_train.to_pickle(FEATURE_DIR_NAME + 'y_train.pkl')
        X_test.to_pickle(FEATURE_DIR_NAME + 'X_test.pkl')

    # CVの設定.methodは[KFold, StratifiedKFold ,GroupKFold]から選択可能
    # CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    # StratifiedKFold or GroupKFoldの場合は、cv_targetに対象カラム名を設定する
    cv = {
        'method': 'KFold',
        'n_splits': 5,
        'random_state': 42,
        'shuffle': True,
        'cv_target': 'hoge'
    }

    if model_type == 'lgb':
        # ######################################################
        # 学習・推論 LightGBM ###################################

        # run nameの設定
        run_name = 'lgb'
        run_name = run_name + suffix
        dir_name = MODEL_DIR_NAME + run_name + '/'

        exist_check(MODEL_DIR_NAME, run_name)
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
            'num_round': 50000,
            'early_stopping_rounds': 500,
            'verbose': -1,
            'verbose_eval': 500,
            'random_state': 999
        }

        runner = Runner(run_name, ModelLGB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)

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
            ModelLGB.calc_feature_importance(dir_name, run_name, use_feature_name)  # feature_importanceを計算
            _pred = runner.run_predict_cv()  # 推論

        if _pred is not None:
            # _predに値が存在する場合（kaggleでのカーネル実行）はsubの作成
            submission[setting.get('target')] = _pred.astype(int)
            submission.to_csv('submission.csv', index=False)
        else:
            # ローカルでの実行
            Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成


    if model_type == 'cb':
        # ######################################################
        # 学習・推論 Catboost ###################################
        # run nameの設定
        run_name = 'cb'
        run_name = run_name + suffix
        dir_name = MODEL_DIR_NAME + run_name + '/'

        exist_check(MODEL_DIR_NAME, run_name)
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
            'loss_function': 'RMSE',
            'task_type': "CPU",
            'iterations': 50000,
            'od_type': "Iter",
            'depth': 10,
            'colsample_bylevel': 0.5,
            'early_stopping_rounds': 500,
            'l2_leaf_reg': 18,
            'random_seed': 42,
            'verbose_eval': 500,
            'use_best_model': True
        }

        runner = Runner(run_name, ModelCB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)

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
            _pred = runner.run_predict_cv()  # 推論

        if _pred is not None:
            # _predに値が存在する場合（kaggleでのカーネル実行）はsubの作成
            submission[setting.get('target')] = _pred.astype(int)
            submission.to_csv('submission.csv', index=False)
        else:
            # ローカルでの実行
            Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    return 'Success!'


if __name__ == '__main__':
    fire.Fire(main)
