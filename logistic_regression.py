import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from dtreeviz.trees import *
from sklearn.metrics import confusion_matrix
# 決定木のためのモジュール読み込み
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

class Tree():
    def __init__(self, df):
        self.df = df
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()

    def set_X_and_y(self, objective_key=""):
        self.X = self.df.drop(objective_key, axis=1)
        self.y = self.df[objective_key].astype(int)

    def train_test_data_split(self, random_state=1, test_size=0.3):
        from sklearn.model_selection import train_test_split
        return train_test_split(self.X, self.y,random_state=random_state, test_size=test_size)

    def std_X(self, X_train, X_test):
        # データの標準化処理
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        return X_train_std, X_test_std

    def random_forest(self, random_state=0, min_samples_leaf=3, max_depth=2):
        from sklearn.ensemble import RandomForestClassifier
        X_train, X_test, y_train, y_test = self.train_test_data_split(random_state=random_state, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        forest = RandomForestClassifier(
                min_samples_leaf=min_samples_leaf,
                random_state=random_state+1,
                max_depth=max_depth)
        forest.fit(X_train_std, y_train)
        score = {
                'train': metrics.accuracy_score(y_train, forest.predict(X_train_std)) ,
                'test': metrics.accuracy_score(y_test, forest.predict(X_test_std))
            }
        print(score)

    def cross_validation(self, max_depth=2, save_file_name="decision_tree_image/tmp.pdf"):
        from sklearn.model_selection import cross_val_score
        print("    ======= 交差検証 ======")
        clf = DecisionTreeClassifier(max_depth = max_depth)
        score = cross_val_score(estimator = clf, X = self.X, y = self.y, cv = 5)
        plt.figure()
        viz = dtreeviz(clf, self.X, self.y, target_name='score_dummy', feature_names=list(self.X.keys()), class_names=["high", "low"])
        viz.save("decision_tree_image/cross_decision_tree_{}.pdf".format(max_depth))
        plt.close()

        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=max_depth, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        clf.fit(X_train_std, y_train)
        print("    [max_depth, score_mean, train_predict, test_predict]")
        print("   ", [max_depth, score.mean(), metrics.accuracy_score(y_train, clf.predict(X_train_std)), metrics.accuracy_score(y_test, clf.predict(X_test_std))])
        print("    ============ end =============")

    def grid_search(self):
        from sklearn.model_selection import GridSearchCV
        print("    ======= グリッドサーチ ======")
        params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
              'criterion': ['gini', 'entropy']}
        clf = GridSearchCV(DecisionTreeClassifier(), params, cv = 10)
        clf.fit(X = self.X, y = self.y)
        print("   ",clf.best_estimator_)
        print("   best_score: ",clf.best_score_)
        print("   ",clf.best_params_)
        print("    ============ end =============")

    def learning_curve_show(self, save_file_name, max_depth=2):
        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=1, test_size=0.3)
        from sklearn.model_selection import learning_curve
        from sklearn.pipeline import make_pipeline
        pipe_lr = make_pipeline(StandardScaler(),
                        DecisionTreeClassifier(
                        class_weight=None, criterion='gini', max_depth=max_depth,
                        max_features=None, max_leaf_nodes=None,
                        min_impurity_split=1e-07, min_samples_leaf=1,
                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                        presort=False, random_state=None, splitter='best'))
        train_sizes, train_scores, test_scores = learning_curve(
                    estimator=pipe_lr,
                    X = X_train,
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

    def decision_tree_classifier(self, save_file_name="", max_depth=2):
        # 決定木による学習
        print("============== 決定木(max_depth={}) ===============".format(max_depth))

        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=0, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)

        treeModel = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        treeModel.fit(X_train_std, y_train)
        print("訓練データ(data_size:{})： ".format(len(X_train_std)), metrics.accuracy_score(y_train, treeModel.predict(X_train_std)))
        print('訓練データ：Confusion matrix:\n{}'.format(confusion_matrix(y_train, treeModel.predict(X_train_std))))
        print("テストデータ(data_size:{})： ".format(len(X_test_std)), metrics.accuracy_score(y_test, treeModel.predict(X_test_std)))
        print('テストデータ：Confusion matrix:\n{}'.format(confusion_matrix(y_test, treeModel.predict(X_test_std))))

        print("決定木による各項目における重要度")
        importance = pd.DataFrame({ '変数':self.X.columns, '重要度':treeModel.feature_importances_})
        print(importance)
        treeModel.fit(self.X, self.y)
        predicted = pd.DataFrame({'TreePredicted':treeModel.predict(self.X)})
        plt.figure()
        viz = dtreeviz(treeModel, self.X, self.y, target_name='score_dummy', feature_names=list(self.X.keys()), class_names=["high", "low"])
        viz.save(save_file_name)
        plt.close()
        print("============== end ===============")

    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        print("============== 回帰木 ===============")
        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=0, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)

        logisticModel = LogisticRegression()
        logisticModel.fit(X_train_std, y_train)
        predicted = pd.DataFrame({'LogisPredicted':logisticModel.predict(X_test_std)})
        print("訓練データ:", metrics.accuracy_score(y_train, logisticModel.predict(X_train_std)))
        print("testデータ:", metrics.accuracy_score(y_test, logisticModel.predict(X_test_std)))

        print('\n')
        print("============== end ===============")

    def drop_columns(self, drop_list):
        self.df =  self.df.drop(drop_list, axis=1)

    def add_dummy_score(self):
        #df.loc[df["score_std"] > 0.26, "score_dummy"] = 0 # High
        #df.loc[df["score_std"] < -0.74, "score_dummy"] = 1 # Low
        #df.loc[(df["score_std"] <= 0.26) & (df["score_std"] >= -0.74), "score_dummy"] = 2 # Newtral
        self.df.loc[self.df["score_std"] >= -0.24, "score_dummy"] = 0 # High
        self.df.loc[self.df["score_std"] < -0.24, "score_dummy"] = 1 # Low
        return self.df

