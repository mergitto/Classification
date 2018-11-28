# 分類問題(Classification)
機械学習の一つの手法であるscikit-learnを使用して分類木・回帰木を作成する


## 使用方法
csvデータを読み込みdataframe型で各分類手法の初期化を行う。

**必須**: `add_dummy_score`, `drop_columns`, `set_X_and_y`
- add_dummy_score … 必要があれば、教師データを作成するための処理
- drop_columns … 不必要なデータをlistで渡してdataframeから削除する
- set_X_and_y … 説明変数(X), 目的変数(y)のセッター

