import sys
import os
import datetime
import yaml
import json
import collections as cl
import warnings
import fire

from load_data import read_data_all
from create_feature import create_feature
from staging import staging_train, staging_test
from model_lgb import ModelLGB
from runner import Runner
from util import Submission

warnings.filterwarnings("ignore")
now = datetime.datetime.now()
suffix = now.strftime("_%m%d_%H%M")
warnings.simplefilter('ignore')
key_list = ['use_features', 'model_params', 'cv', 'setting']

CONFIG_FILE = '../config/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['OUTPUT_DIR_NAME']


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

    confirm('mode:{}, create_feature:{} '.format(str(mode), str(create_features)))

    if create_features:
        # データ生成
        train, test, train_labels, submission = read_data_all(mode)
        features_train = create_feature(train)
        _, _ = staging_train(train_labels, features_train, save=True)
        features_test = create_feature(test)
        _ = staging_test(test, features_test, submission, save=True)

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
        'num_round': 100,
        'early_stopping_rounds': 100,
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
        runner.run_predict_cv()  # 推論

    Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    return 'Success!'


if __name__ == '__main__':
    fire.Fire(main)
