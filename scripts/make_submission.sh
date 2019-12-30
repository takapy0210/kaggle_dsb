#!/bin/sh

OUTPUT_FILE='scripts/for_script_submission.py'
OUTPUT_TMP_FILE='scripts/for_script_submission_tmp.py'
INITIAL_STATEMENTS="
from typing import List

import pandas as pd
import lightgbm as lgb
"

# kaggleに提出するスクリプトファイルを生成
rm $OUTPUT_FILE
touch $OUTPUT_FILE

# 空のファイルに追記
cat scripts/load_data.py >> $OUTPUT_FILE
cat scripts/create_feature.py >> $OUTPUT_FILE
cat scripts/staging.py >> $OUTPUT_FILE
cat scripts/training.py >> $OUTPUT_FILE
cat scripts/main.py >> $OUTPUT_FILE

# import文の整形、loggerの削除、ファイルパスの補正 TODO: コメント行の削除
sed -e '/^import/d' $OUTPUT_FILE |
  sed -e '/^from/d' |
  sed -e '/logger/d' |
  sed -e '/^file_path = /d'|
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
