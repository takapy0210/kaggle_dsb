import sys
import os
import datetime
import yaml
import json
import collections as cl
import warnings
import fire
import pandas as pd
import category_encoders as ce

from load_data import read_data_all
from create_feature import encode_title, get_train_and_test, preprocess, create_user_profile_test, create_user_profile_train, add_session_order_to_train
from feature_selection import select_ajusted
from model_lgb import ModelLGB
from model_cb import ModelCB
from model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
from util import Submission, Util
import gc

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


def main(mode='prd', create_features=True, model_type='lgb', is_kernel=False) -> str:

    confirm('mode:{}, create_feature:{} '.format(str(mode), str(create_features)))

    if create_features:
        # データ生成
        train, test, specs, train_labels, submission = read_data_all(mode)

        features_train, features_test, win_code, list_of_user_activities, list_of_event_code, \
            activities_labels, assess_titles, list_of_event_id, all_title_event_code, all_type_world = encode_title(train, test)

        del train, test
        gc.collect()

        features_train = features_train.merge(specs, how='left', on='event_id', suffixes=('','_y'))
        features_test = features_test.merge(specs, how='left', on='event_id', suffixes=('','_y'))
        features_train, features_test = get_train_and_test(features_train, features_test,
                                        win_code, list_of_user_activities, list_of_event_code,
                                        activities_labels, assess_titles, list_of_event_id, all_title_event_code, all_type_world, is_kernel)
        reduce_train, reduce_test, _ = preprocess(features_train, features_test, assess_titles)

        # user属性情報の生成とマージ
        """ スコア悪くなるので一旦コメント
        train, train_session_master = add_session_order_to_train(train, train_labels)
        user_profiles_train = create_user_profile_train(train)
        user_profiles_test = create_user_profile_test(test)
        train_session_master = train_session_master.merge(user_profiles_train, how='left', on=['installation_id', 'session_order'])
        reduce_train = reduce_train.merge(train_session_master, how='left', on=['installation_id', 'game_session'])
        reduce_test = reduce_test.merge(user_profiles_test, how='left', on='installation_id')
        """

        del features_train, features_test, _
        gc.collect()

        # 不要なカラムの削除
        # cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'session_order', 'accuracy_group', 'timestampDate'] + ['acc_' + title for title in assess_titles] # installation_idでGroupKFoldしない場合はこちらを使用
        cols_to_drop = ['game_session', 'timestamp', 'session_order', 'accuracy_group', 'timestampDate'] + ['acc_' + title for title in assess_titles]
        cols_to_drop = [col for col in cols_to_drop if col in reduce_train.columns]
        X_train = reduce_train.drop(cols_to_drop, axis=1)
        X_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため
        y_train = reduce_train['accuracy_group']

        cols_to_drop = [col for col in cols_to_drop if col in reduce_test.columns]
        X_test = reduce_test.drop(cols_to_drop, axis=1)
        X_test.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]  # カラム名にカンマなどが含まれており、lightgbmでエラーが出るため

        # 特徴量選択
        """ スコア悪くなるので一旦コメント
        to_exclude, X_test = select_ajusted(X_train, X_test)
        X_train = X_train.drop(to_exclude, axis=1)
        X_test = X_test.drop(to_exclude, axis=1)
        """

        X_train.to_pickle(FEATURE_DIR_NAME + 'X_train.pkl')
        y_train.to_pickle(FEATURE_DIR_NAME + 'y_train.pkl')
        X_test.to_pickle(FEATURE_DIR_NAME + 'X_test.pkl')

    # CVの設定.methodは[KFold, StratifiedKFold ,GroupKFold]から選択可能
    # CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    # StratifiedKFold or GroupKFoldの場合は、cv_targetに対象カラム名を設定する
    cv = {
        'method': 'GroupKFold',
        'n_splits': 5,
        'random_state': 42,
        'shuffle': True,
        'cv_target': 'installation_id'
    }

    if model_type == 'lgb' or model_type == 'all':
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
            'learning_rate': 0.01,
            'subsample': 0.75,
            'subsample_freq': 1,
            'feature_fraction': 0.9,
            'max_depth': 15,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'num_round': 50000,
            'early_stopping_rounds': 300,
            'verbose': -1,
            'verbose_eval': 500,
            'random_state': 999
        }

        if is_kernel:
            runner = Runner(run_name, ModelLGB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME, X_train, y_train, X_test)
        else:
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
            _pred = runner.run_predict_cv(is_kernel)  # 推論

        if is_kernel:
            # kaggleカーネル実行
            if model_type == 'lgb':
                # シングルモデルでのcsv作成
                submission[setting.get('target')] = _pred.astype(int)
                submission.to_csv('submission.csv', index=False)
            else:
                # ブレンドするためのcsv作成
                submission_lgb = submission.copy()
                submission_lgb[setting.get('target')] = _pred.astype(int)
                submission_lgb.to_csv('submission_lgb.csv', index=False)
        else:
            # ローカルでの実行
            Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成


    if model_type == 'cb' or model_type == 'all':
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

        if is_kernel:
            runner = Runner(run_name, ModelCB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME, X_train, y_train, X_test)
        else:
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
            _pred = runner.run_predict_cv(is_kernel)  # 推論

        if is_kernel:
            # kaggleカーネル実行
            if model_type == 'cb':
                # シングルモデルでのcsv作成
                submission[setting.get('target')] = _pred.astype(int)
                submission.to_csv('submission.csv', index=False)
            else:
                # ブレンドするためのcsv作成
                submission_cb = submission.copy()
                submission_cb[setting.get('target')] = _pred.astype(int)
                submission_cb.to_csv('submission_cb.csv', index=False)
        else:
            # ローカルでの実行
            Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    if model_type == 'nn' or model_type == 'all':
        # ######################################################
        # 学習・推論 NN(MLP) ###################################
        # run nameの設定
        run_name = 'nn'
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
            'layers': 4,
            'nb_epoch': 500,
            'patience': 20,
            'dropout': 0.3,
            'units': 512,
            'classes': 1
        }

        if is_kernel:
            runner = Runner(run_name, ModelNN, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME, X_train, y_train, X_test)
        else:
            runner = Runner(run_name, ModelNN, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)

        use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

        # モデルのconfigをjsonで保存
        value_list = [use_feature_name, model_params, cv, setting]
        save_model_config(key_list, value_list, dir_name, run_name)

        # one-hot-encoding
        if len(runner.categoricals) > 0:
            one_hot_encoder = ce.OneHotEncoder(cols=runner.categoricals, drop_invariant=True)
            one_hot_encoder.fit(runner.train_x[runner.categoricals])
            ohe_path = os.path.join('.', 'one-hot-enc.pkl')
            Util.dump(one_hot_encoder, ohe_path)

        if cv.get('method') == 'None':
            # TODO: こちらも動くように修正する
            runner.run_train_all()  # 全データで学習
            runner.run_predict_all()  # 推論
        else:
            runner.run_train_cv()  # 学習
            _pred = runner.run_predict_cv(is_kernel)  # 推論

        if is_kernel:
            # kaggleカーネル実行
            if model_type == 'nn':
                # シングルモデルでのcsv作成
                submission[setting.get('target')] = _pred.astype(int)
                submission.to_csv('submission.csv', index=False)
            else:
                # ブレンドするためのcsv作成
                submission_nn = submission.copy()
                submission_nn[setting.get('target')] = _pred.astype(int)
                submission_nn.to_csv('submission_nn.csv', index=False)
        else:
            # ローカルでの実行
            Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    if model_type == 'xgb' or model_type == 'all':
        # ######################################################
        # 学習・推論 NN(MLP) ###################################
        # run nameの設定
        run_name = 'xgb'
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
            # from: https://www.kaggle.com/servietsky/xgb-simple-and-efficient
            'colsample_bytree': 0.4603,
            'gamma': 0.0468,
            'learning_rate': 0.05,
            'max_depth': 3,
            'min_child_weight': 1.7817,
            'n_estimators': 2200,
            'reg_alpha': 0.4640,
            'reg_lambda': 0.8571,
            'subsample': 0.5213,
            'silent': 1,
            'random_state': 7,
            'nthread': -1
        }

        if is_kernel:
            runner = Runner(run_name, ModelXGB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME, X_train, y_train, X_test)
        else:
            runner = Runner(run_name, ModelXGB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)

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
            _pred = runner.run_predict_cv(is_kernel)  # 推論

        if is_kernel:
            # kaggleカーネル実行
            if model_type == 'xgb':
                # シングルモデルでのcsv作成
                submission[setting.get('target')] = _pred.astype(int)
                submission.to_csv('submission.csv', index=False)
            else:
                # ブレンドするためのcsv作成
                submission_xgb = submission.copy()
                submission_xgb[setting.get('target')] = _pred.astype(int)
                submission_xgb.to_csv('submission_xgb.csv', index=False)
        else:
            # ローカルでの実行
            Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    # 推論のブレンド
    # TODO: xbgの結果も入れる
    if model_type == 'all' and is_kernel:
        weights = {'lgb': 0.30, 'cb': 0.60, 'nn': 0.10}
        blend_pred = (submission_lgb[setting.get('target')] * weights['lgb']) \
                        + (submission_cb[setting.get('target')] * weights['cb']) \
                        + (submission_nn[setting.get('target')] * weights['nn'])

        dist = Counter(reduce_train[setting.get('target')])
        for k in dist:
            dist[k] /= len(reduce_train)

        acum = 0
        bound = {}
        for i in range(3):
            acum += dist[i]
            bound[i] = np.percentile(blend_pred, acum * 100)

        def classify(x):
            if x <= bound[0]:
                return 0
            elif x <= bound[1]:
                return 1
            elif x <= bound[2]:
                return 2
            else:
                return 3

        blend_pred = np.array(list(map(classify, blend_pred)))

        submission[setting.get('target')] = blend_pred.astype(int)
        submission.to_csv('submission.csv', index=False)

    return 'Success!'


if __name__ == '__main__':
    fire.Fire(main)
