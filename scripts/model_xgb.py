import os
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoost, CatBoostRegressor
from catboost import Pool
from qwk import eval_qwk_lgb_regr
from model import Model
from util import Util

import xgboost as xgb

# 各foldのモデルを保存する配列
model_array = []


class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        dtrain = {'X': tr_x, 'y': tr_y}

        if validation:
            dvalid = {'X': va_x, 'y': va_y}

        # ハイパーパラメータの設定
        params = dict(self.params)
        # verbose_eval = params.pop('verbose_eval')

        # 学習
        if validation:
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(
                            dtrain['X'],
                            dtrain['y']
                            # eval_set=(dvalid['X'], dvalid['y']),
                            # verbose=verbose_eval
                            )
            model_array.append(self.model)

        else:
            # 動かない
            watchlist = [(dtrain, 'train')]
            self.model = lgb.train(params, dtrain, num_round, evals=watchlist)
            model_array.append(self.model)

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
