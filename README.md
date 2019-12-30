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


### pklファイルでの読み込み実行

dev=Falseの時のみ有効です。（read_pickleはnrowsが指定できないため）

```
python3 scripts/main.py --dev=False --use_pkl=True
```

### kaggleへのサブミット

```
sh scripts/make_submission.sh
```

を実行すると、kaggle notebook環境で実行可能なスクリプトが生成されます。（`for_script_submission.py`）
コピペで動くところまで確認済みです。

※コード構成にめちゃくちゃ依存しているので、コードいじっているうちに動かなくなる可能性はあります汗
