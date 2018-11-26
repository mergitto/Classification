import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from dtreeviz.trees import *
from sklearn.metrics import confusion_matrix
# 決定木のためのモジュール読み込み
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Tree():
    def __init__(self, X, y):
        self.X = X
        self.y = y

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

    def cross_validation(self):
        print("    ======= 交差検証 ======")
        print("    [max_depth, score_mean]")
        for max_depth in [2, 3, 4, 5, 6, 7, 8, 9]:
            clf = DecisionTreeClassifier(max_depth = max_depth)
            score = cross_val_score(estimator = clf, X = self.X, y = self.y, cv = 5)
            print("   ",[max_depth, score.mean()])
            viz = dtreeviz(clf, self.X, self.y, target_name='score_dummy', feature_names=list(self.X.keys()), class_names=["high", "low"])
            viz.save("decision_tree_image/cross_decision_tree_{}.pdf".format(max_depth))
        print("    ============ end =============")

    def grid_search(self):
        print("    ======= グリッドサーチ ======")
        params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
              'criterion': ['gini', 'entropy']}
        clf = GridSearchCV(DecisionTreeClassifier(), params, cv = 10)
        clf.fit(X = self.X, y = self.y)
        print("   ",clf.best_estimator_)
        print("   best_score: ",clf.best_score_)
        print("   ",clf.best_params_)
        print("    ============ end =============")

    def learning_curve_show(self, save_file_name):
        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=1, test_size=0.3)
        from sklearn.model_selection import learning_curve
        from sklearn.pipeline import make_pipeline
        pipe_lr = make_pipeline(StandardScaler(),
                        DecisionTreeClassifier(
                        class_weight=None, criterion='gini', max_depth=3,
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

    def decision_tree_classifier(self, df, save_file_name=""):
        # 決定木による学習
        print("============== 決定木 ===============")

        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=0, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)

        treeModel = DecisionTreeClassifier(max_depth=3, random_state=0)
        treeModel.fit(X_train_std, y_train)
        print("訓練データ： ", metrics.accuracy_score(y_train, treeModel.predict(X_train_std)))
        print("テストデータ： ", metrics.accuracy_score(y_test, treeModel.predict(X_test_std)))
        print('テストデータ：Confusion matrix:\n{}'.format(confusion_matrix(y_test, treeModel.predict(X_test_std))))

        print("決定木による各項目における重要度")
        importance = pd.DataFrame({ '変数':self.X.columns, '重要度':treeModel.feature_importances_})
        print(importance)
        treeModel.fit(self.X, self.y)
        predicted = pd.DataFrame({'TreePredicted':treeModel.predict(self.X)})
        data_predicted = pd.concat([df, predicted], axis =1)
        if True:
            viz = dtreeviz(treeModel, self.X, self.y, target_name='score_dummy', feature_names=list(self.X.keys()), class_names=["high", "low"])
            viz.save(save_file_name)
        print("============== end ===============")
        return data_predicted

    def logistic_regression(self, df):
        from sklearn.linear_model import LogisticRegression
        print("============== 回帰木 ===============")
        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=0, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)

        logisticModel = LogisticRegression()
        logisticModel.fit(X_train_std, y_train)
        predicted = pd.DataFrame({'LogisPredicted':logisticModel.predict(X_test_std)})
        print("訓練データ:", metrics.accuracy_score(y_train, logisticModel.predict(X_train_std)))
        print("testデータ:", metrics.accuracy_score(y_test, logisticModel.predict(X_test_std)))
        data_predicted = pd.concat([df, predicted], axis =1)

        print('\n')
        if False:
            plt.scatter(data_predicted['word_length'],data_predicted['recommend_rank'], c=data_predicted['LogisPredicted'])
            plt.xlabel('bm25')
            plt.ylabel('recommend_rank')
            plt.show()
        print("============== end ===============")
        return data_predicted

def drop_column(df):
    df = df.drop([
            "advice_divide_mecab", "bm25_average", "course_code", "created",
            "keywords", "modified", "score", "score_min_max", "search_word",
            "search_word_wakati", "topic", "st_no"], axis=1)
    return df

def add_dummy_score(df):
    #df.loc[df["score_std"] > 0.26, "score_dummy"] = 0 # High
    #df.loc[df["score_std"] < -0.74, "score_dummy"] = 1 # Low
    #df.loc[(df["score_std"] <= 0.26) & (df["score_std"] >= -0.74), "score_dummy"] = 2 # Newtral
    df.loc[df["score_std"] >= -0.24, "score_dummy"] = 0 # High
    df.loc[df["score_std"] < -0.24, "score_dummy"] = 1 # Low
    return df

if True:
    df = pd.read_csv('./questionnaire_evaluations_from_1031_to_1109_all.csv')
    df = drop_column(df)
    df = add_dummy_score(df)
    df = df.drop("score_std", axis=1) # score_dummyはscore_stdで予測できてしまうので削除
    X = df.drop([
            "score_dummy", "report_created_date", "bm25_sum",
            "type_id", "shokushu_id"
            #"tfidf_top_average", "recommend_level",
        ], axis=1) # score_dummy以外の列を抽出
    print(X.keys())
    y = df["score_dummy"].astype(int) # score_dummyの列を抽出
    tree = Tree(X, y)
    tree.decision_tree_classifier(df, save_file_name="decision_tree_image/decision_tree_questionnaire.pdf")
    tree.cross_validation()
    tree.grid_search()
    tree.learning_curve_show(save_file_name="./decision_tree_image/qustion_learning_curve.pdf")
    tree.logistic_regression(df)

if False:
    df = pd.read_csv('./sum.csv')
    df["topic"] = 0
    df = drop_column(df)
    df = add_dummy_score(df)
    df = df.drop("score_std", axis=1) # score_dummyはscore_stdで予測できてしまうので削除
    X = df.drop(["score_dummy", "bm25_sum", "tfidf_top_average", "recommend_level", "report_created_date"], axis=1) # score_dummy以外の列を抽出
    y = df["score_dummy"].astype(int) # score_dummyの列を抽出
    df = decision_tree_classifier(X, y, df, save_file_name="sum.pdf", learning_curve_name="decision_tree_image/sum_curve.pdf")

    df = pd.read_csv('./sum_jsd.csv')
    df["topic"] = 0
    df = drop_column(df)
    df = add_dummy_score(df)
    df = df.drop("score_std", axis=1) # score_dummyはscore_stdで予測できてしまうので削除
    X = df.drop(["score_dummy", "bm25_sum", "tfidf_top_average", "recommend_level", "report_created_date"], axis=1) # score_dummy以外の列を抽出
    y = df["score_dummy"].astype(int) # score_dummyの列を抽出
    df = decision_tree_classifier(X, y, df, save_file_name="sum_jsd.pdf", learning_curve_name="decision_tree_image/sum_jsd_curve.pdf")

    df = pd.read_csv('./sum_jsd_reverse.csv')
    df = drop_column(df)
    df = add_dummy_score(df)
    df = df.drop("score_std", axis=1) # score_dummyはscore_stdで予測できてしまうので削除
    X = df.drop(["score_dummy", "bm25_sum", "tfidf_top_average", "recommend_level", "report_created_date"], axis=1) # score_dummy以外の列を抽出
    y = df["score_dummy"].astype(int) # score_dummyの列を抽出
    df = decision_tree_classifier(X, y, df, save_file_name="sum_jsd_reverse.pdf", learning_curve_name="decision_tree_image/sum_jsd_reverse_curve.pdf")

if False:
    df = pd.read_csv('./questionnaire_evaluations_preprocessed_all.csv')
    df = drop_column(df)
    df = add_dummy_score(df)
    df = df.drop("score_std", axis=1) # score_dummyはscore_stdで予測できてしまうので削除
    df = df.drop(["is_good", "evaluation_id", "report_no", "recommend_formula", "recommend_level"], axis=1) # 無駄なカラムの削除
    #X = df.drop("score_dummy", axis=1) # score_dummy以外の列を抽出
    X = df.drop([
            "score_dummy", "report_created_date", "bm25_sum",
            "type_id", "shokushu_id"
            #"tfidf_top_average", "recommend_level","most_highest_similarity"
        ], axis=1) # score_dummy以外の列を抽出
    y = df["score_dummy"].astype(int) # score_dummyの列を1次元に展開

    df = decision_tree_classifier(X, y, df, save_file_name="decision_tree_evaluations.pdf")
    #df = logistic_regression(X, y, df)

