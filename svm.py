import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation as cv

class Classification():
    def __init__(self, df):
        self.df = df
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()

    def set_X_and_y(self, objective_key=""):
        self.X = self.df.drop(objective_key, axis=1)
        self.y = self.df[objective_key].astype(int)

    def drop_columns(self, drop_list):
        self.df =  self.df.drop(drop_list, axis=1)

    def add_dummy_score(self):
        self.df.loc[self.df["score_std"] >= -0.24, "score_dummy"] = 0 # High
        self.df.loc[self.df["score_std"] < -0.24, "score_dummy"] = 1 # Low

    def cross_validation(self):
        from sklearn.model_selection import cross_val_score
        print("======= 交差検証 ======")
        svc = SVC()
        X_train, X_test, y_train, y_test = cv.train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)
        print("予測精度: ",score)
        print("======= k-分割交差検証 ======")
        scores = cross_val_score(svc, X_train, y_train, cv=10)
        print("分割時の予測精度: ", scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def simple_svm(self):
        self.cross_validation()
        print("================単純な学習後のテスト==================") # 学習データとテストデータの分離
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=0, test_size=0.3)
        # 学習とテスト
        svc = SVC()
        svc.fit(X_train, y_train)
        print('Train score: {:.3f}, X shape: {}, y shape: {}'.format(svc.score(X_train, y_train), X_train.shape, y_train.shape))
        print('Test score: {:.3f}, X shape: {}, y shape: {}'.format(svc.score(X_test, y_test), X_test.shape, y_test.shape))
        print('Confusion matrix:\n{}'.format(confusion_matrix(y_train, svc.predict(X_train))))
        print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc.predict(X_test))))

    def std_svm(self):
        print("================標準化した後のテスト==================")
        # 学習データとテストデータの分離
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=0, test_size=0.3)
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

    def grid_svm(self):
        print("================グリッドサーチを行った後のテスト==================")
        # 学習データとテストデータの分離
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=0, test_size=0.3)
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

