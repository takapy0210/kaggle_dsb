# kaggle_dsb

## Pipeline

![pipeline](https://github.com/shyaginuma/kaggle_dsb/blob/master/IGQL-DSB%20Pipeline.svg)

## ローカルでの実行

PJのrootディレクトリにいると仮定すると、

```
python3 scripts/main.py
```

でデータ読み込み〜動作します。

### テスト実行

スクリプトの動作確認時などは以下のコマンドで高速で実行確認できます。

```
python3 scripts/main.py --dev=True
```

参考：[python-fire](https://github.com/google/python-fire)
