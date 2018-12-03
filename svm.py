import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dtreeviz.trees import *
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
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
        tmp_df = self.df.sort_values("score_std", ascending=False)
        df_size = len(self.df)
        high_rate = int(df_size * 0.6)
        threshold = tmp_df[:high_rate].iloc[-1].score_std
        print("正規化後の閾値: ", threshold)
        self.df.loc[self.df["score_std"] >= threshold, "score_dummy"] = 0 # High
        self.df.loc[self.df["score_std"] < threshold, "score_dummy"] = 1 # Low

    def std_X(self, X_train, X_test):
        # データの標準化処理
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        return X_train_std, X_test_std

    def drop_na(self, drop_na_list=[]):
        self.df = self.df.dropna(subset=drop_na_list)
        self.df = self.df.reset_index(drop=True)

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

    def train_test_data_split(self, random_state=1, test_size=0.3):
        from sklearn.model_selection import train_test_split
        return train_test_split(self.X, self.y,random_state=random_state, test_size=test_size)

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
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        svc_grid_search = GridSearchCV(SVC(), params, cv=10)
        svc_grid_search.fit(X_train_std, y_train)
        print('Train score: {:.3f}'.format(svc_grid_search.score(X_train_std, y_train)))
        print('Test score: {:.3f}'.format(svc_grid_search.score(X_test_std, y_test)))
        print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc_grid_search.predict(X_test_std))))
        print('Best parameters: {}'.format(svc_grid_search.best_params_))
        print('Best estimator: {}'.format(svc_grid_search.best_estimator_))

        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
        }
        svc_grid_search = GridSearchCV(LinearSVC(), params, cv=10)
        svc_grid_search.fit(X_train_std, y_train)
        print('Train score: {:.3f}'.format(svc_grid_search.score(X_train_std, y_train)))
        print('Test score: {:.3f}'.format(svc_grid_search.score(X_test_std, y_test)))
        print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc_grid_search.predict(X_test_std))))
        print('Best parameters: {}'.format(svc_grid_search.best_params_))
        print('Best estimator: {}'.format(svc_grid_search.best_estimator_))

        params = {
            'n_neighbors': range(1,11)
        }
        svc_grid_search = GridSearchCV(KNeighborsClassifier(), params, cv=10)
        svc_grid_search.fit(X_train_std, y_train)
        print('Train score: {:.3f}'.format(svc_grid_search.score(X_train_std, y_train)))
        print('Test score: {:.3f}'.format(svc_grid_search.score(X_test_std, y_test)))
        print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc_grid_search.predict(X_test_std))))
        print('Best parameters: {}'.format(svc_grid_search.best_params_))
        print('Best estimator: {}'.format(svc_grid_search.best_estimator_))

    def get_model(self, clf_name="svm", C=10, gamma=0.01, n_neighbors=2):
        if clf_name == "svm":
            clf = SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape=None, degree=3, gamma=gamma, kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
        elif clf_name == "linear_svc":
            clf = LinearSVC(C=C, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
        elif clf_name == "knn":
            clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=1, n_neighbors=n_neighbors, p=2,
                    weights='uniform')
        return clf

    def learning_curve_show(self, save_file_name, clf_name="svm", C=0.001, gamma=0.01, n_neighbors=2):
        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=1, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        from sklearn.model_selection import learning_curve
        from sklearn.pipeline import make_pipeline
        clf = self.get_model(clf_name=clf_name, C=C, gamma=gamma, n_neighbors=n_neighbors)
        pipe_lr = make_pipeline(StandardScaler(), clf)
        train_sizes, train_scores, test_scores = learning_curve(
                    estimator=pipe_lr,
                    X = X_train_std,
                    y = y_train,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    cv=10,
                    n_jobs=1 )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.figure()
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label="training accuracy")
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
        plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label="validation accuracy" )
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
        plt.grid()
        plt.xlabel("number of training samples")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.ylim([0.5, 1.0])
        plt.savefig(save_file_name)
        plt.close()

    def validation_curve_show(self, save_file_name, param_range=[], param_name="svc__C", clf_name="svm"):
        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=1, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        from sklearn.model_selection import validation_curve
        from sklearn.pipeline import make_pipeline
        clf = self.get_model(clf_name=clf_name)
        pipe_lr = make_pipeline(StandardScaler(), clf)
        train_scores, test_scores = validation_curve(
                estimator=pipe_lr,
                X=X_train_std, y=y_train,
                param_name=param_name, param_range=param_range, cv=10)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.figure()
        plt.plot(param_range, train_mean, color="blue", marker="o", markersize=5, label="training accuracy")
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
        plt.plot(param_range, test_mean, color="green", marker="o", markersize=5, label="test accuracy")
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
        plt.grid()
        plt.xscale('log')
        plt.legend(loc="lower right")
        plt.xlabel(param_name)
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(save_file_name)
        plt.close()


