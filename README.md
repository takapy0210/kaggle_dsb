# kaggle_dsb

## Pipeline

![pipeline](https://github.com/shyaginuma/kaggle_dsb/blob/master/IGQL-DSB%20Pipeline.svg)

## ローカルでの実行

PJのrootディレクトリにいると仮定すると、

```
python3 scripts/run.py
```

でデータ読み込み〜動作します。

### 引数について

#### --mode

- `prd` : オリジナルのcsvファイルを読み込む（デフォルト）

- `dev` : オリジナルのcsvファイルから100kのみ読み込む（検証モード）

- `pkl` : オリジナルのcsvファイルをpkl変換したデータを読み込む。ローカルで全件動作確認する場合はこちらが推奨

#### --create_features

- True : 特徴量生成処理を実行する（デフォルト）

- False : 特徴量生成処理を実行しない。すでに特徴量生成が終了しており、モデルのパラメータチューニングなどを行う際に使用する

#### --model_type

- `lbg` : LightGBMでの学習を行う（デフォルト）

- `cb` : CatBoostでの学習を行う

- `nn` : MLPでの学習を行う

- `all` : LightGBM、CatBoost、MLPで学習を行い、最後にブレンドしたsubを作成する

#### --is_kernel

- True : kaggleのカーネル実行

- False : ローカル実行（デフォルト）

### テスト実行

スクリプトの動作確認時などは以下のコマンドで高速で実行確認できます。

```
python3 scripts/run.py --mode='dev'
```

参考：[python-fire](https://github.com/google/python-fire)


### pklファイル（全データ）での読み込み実行

- 通常で実行するよりデータの読み込みが高速に行えます

```
python3 scripts/run.py --mode='pkl'
```

### csvファイル（全データ）での読み込み実行（デフォルトの動作）

```
python3 scripts/run.py --mode='prd'
```

## kaggleへのサブミット

```
sh scripts/make_submission.sh
```

を実行すると、kaggle notebook環境で実行可能なスクリプトが生成されます。（`for_script_submission.py`）
コピペで動くところまで確認済みです。

※コード構成にめちゃくちゃ依存しているので、コードいじっているうちに動かなくなる可能性はあります汗


## run.py関連の説明

### model.py

- loghtGBMやxgboost、scikit-learnの各モデルをラップしたクラス。学習や予測を行う。

- このクラスを継承してモデルスクリプトを作成することで、インターフェースの差分を吸収（e.g. `model_lgb.py`）

### runner.py

- CV含めて学習・予測の一連の流れを行うクラス。

### util.py

- ファイル入出力

- ログの出力・表示

- 計算結果の出力・表示