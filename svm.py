import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation as cv

def cross_validation(X, y):
    print("======= 交差検証 ======")
    svc = SVC()
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.3, random_state=0)
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    print("予測精度: ",score)
    print("======= k-分割交差検証 ======")
    scores = cross_val_score(svc, X_train, y_train, cv=10)
    print("分割時の予測精度: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def simple_svm(X, y):
    cross_validation(X, y)
    print("================単純な学習後のテスト==================") # 学習データとテストデータの分離
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    # 学習とテスト
    svc = SVC()
    svc.fit(X_train, y_train)
    print('Train score: {:.3f}, X shape: {}, y shape: {}'.format(svc.score(X_train, y_train), X_train.shape, y_train.shape))
    print('Test score: {:.3f}, X shape: {}, y shape: {}'.format(svc.score(X_test, y_test), X_test.shape, y_test.shape))
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_train, svc.predict(X_train))))
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc.predict(X_test))))

def std_svm(X, y):
    print("================標準化した後のテスト==================")
    # 学習データとテストデータの分離
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    # 標準化
    std_scl = StandardScaler()
    std_scl.fit(X_train)
    X_train = std_scl.transform(X_train)
    X_test = std_scl.transform(X_test)
    svc = SVC()
    svc.fit(X_train, y_train)
    print('Train score: {:.3f}'.format(svc.score(X_train, y_train)))
    print('Test score: {:.3f}'.format(svc.score(X_test, y_test)))
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_train, svc.predict(X_train))))
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc.predict(X_test))))

def grid_svm(X, y):
    print("================グリッドサーチを行った後のテスト==================")
    # 学習データとテストデータの分離
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    svc_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    svc_grid_search = GridSearchCV(SVC(), svc_param_grid, cv=10)
    svc_grid_search.fit(X_train, y_train)
    print('Train score: {:.3f}'.format(svc_grid_search.score(X_train, y_train)))
    print('Test score: {:.3f}'.format(svc_grid_search.score(X_test, y_test)))
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc_grid_search.predict(X_test))))
    print('Best parameters: {}'.format(svc_grid_search.best_params_))
    print('Best estimator: {}'.format(svc_grid_search.best_estimator_))

def drop_column(df):
    df = df.drop([
            "advice_divide_mecab", "bm25_average", "course_code", "created",
            "keywords", "modified", "score", "score_min_max", "search_word",
            "type_id", "shokushu_id",
            "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
            "search_word_wakati", "topic", "st_no"], axis=1)
    return df

def add_dummy_score(df):
    #df.loc[df["score_std"] > 0.26, "score_dummy"] = 3 # High
    #df.loc[df["score_std"] < -0.74, "score_dummy"] = 1 # Low
    #df.loc[(df["score_std"] <= 0.26) & (df["score_std"] >= -0.74), "score_dummy"] = 2 # Newtral
    df.loc[df["score_std"] >= -0.24, "score_dummy"] = 2 # High
    df.loc[df["score_std"] < -0.24, "score_dummy"] = 1 # Low
    return df

df = pd.read_csv('./questionnaire_evaluations_from_1031_to_1109_all.csv')
df = drop_column(df)
df = add_dummy_score(df)
df = df.drop("score_std", axis=1) # score_dummyはscore_stdで予測できてしまうので削除
df = df.drop("report_created_date", axis=1)
X = df.drop("score_dummy", axis=1) # score_dummy以外の列を抽出
y = df["score_dummy"] # score_dummyの列を1次元に展開

simple_svm(X, y)
std_svm(X, y)
#grid_svm(X, y)


df = pd.read_csv('./questionnaire_evaluations_preprocessed_all.csv')
df = drop_column(df)
df = add_dummy_score(df)
df = df.drop("score_std", axis=1) # score_dummyはscore_stdで予測できてしまうので削除
df = df.drop([
        "is_good", "evaluation_id", "report_no", "recommend_formula", "report_created_date", "bm25_sum", "recommend_level",
        "tfidf_top_average", "recommend_rank", "most_highest_similarity"
    ], axis=1) # 無駄なカラムの削除
X = df.drop("score_dummy", axis=1) # score_dummy以外の列を抽出
y = df["score_dummy"] # score_dummyの列を1次元に展開

print('\n\n')
simple_svm(X, y)
std_svm(X, y)
#grid_svm(X, y)


