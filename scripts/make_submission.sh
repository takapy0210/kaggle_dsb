#!/bin/sh

OUTPUT_FILE='scripts/for_script_submission.py'
OUTPUT_TMP_FILE='scripts/for_script_submission_tmp.py'
INITIAL_STATEMENTS="
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
"

# kaggleに提出するスクリプトファイルを生成
rm $OUTPUT_FILE
touch $OUTPUT_FILE

# 空のファイルに追記
cat scripts/load_data.py >> $OUTPUT_FILE
cat scripts/create_feature.py >> $OUTPUT_FILE
cat scripts/staging.py >> $OUTPUT_FILE
# cat scripts/training.py >> $OUTPUT_FILE
cat scripts/qwk.py >> $OUTPUT_FILE
cat scripts/model.py >> $OUTPUT_FILE
cat scripts/model_lgb.py >> $OUTPUT_FILE
cat scripts/runner.py >> $OUTPUT_FILE
cat scripts/run.py >> $OUTPUT_FILE

# import文の整形、loggerの削除、ファイルパスの補正 TODO: コメント行の削除
sed -e '/^import/d' $OUTPUT_FILE |
  sed -e '/^from/d' |
  sed -e '/logger/d' |
  sed -e '/^file_path = /d'|
  sed -e '/^warnings./d'|
  sed -e '/^CONFIG_FILE = /d'|
  sed -e '/^with open/d'|
  sed -e '/yml = /d'|
  sed -e '/self.logger/d'|
  sed -e '/^FEATURE_DIR_NAME = /d'|
  sed -e '/^MODEL_DIR_NAME = /d'|
  sed -e '/^    confirm/d'|
  sed -e '/^    exist_check/d'|
  sed -e '/to_pickle/d'|
  sed -e '/if save:/d'|
  sed -e "s/Runner(run_name, ModelLGB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)/Runner(run_name, ModelLGB, setting, model_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME, X_train, y_train, X_test)/" |
  sed -e "s/_pred = runner.run_predict_cv()/_pred = runner.run_predict_cv(is_kernel=True)/" |
  sed -e "s/os.path.join(file_path, '..\/data\/input/('..\/input\/data-science-bowl-2019/" |
  sed -e "s/os.path.join(file_path, '..\/data\/output\/submission.csv')/'submission.csv'/" > $OUTPUT_TMP_FILE
echo "$INITIAL_STATEMENTS" > scripts/initial_statements.txt
cat scripts/initial_statements.txt > $OUTPUT_FILE
cat $OUTPUT_TMP_FILE >> $OUTPUT_FILE
rm $OUTPUT_TMP_FILE
rm scripts/initial_statements.txt

# 実行文を置き換え
sed -e 's/fire.Fire(main)/main()/' $OUTPUT_FILE > $OUTPUT_TMP_FILE
cat $OUTPUT_TMP_FILE > $OUTPUT_FILE
rm $OUTPUT_TMP_FILE
