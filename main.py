import pandas as pd
from logistic_regression import Tree, drop_column, add_dummy_score

df = pd.read_csv('./questionnaire_evaluations_from_1031_to_1109_all.csv')
df = drop_column(df)
df = add_dummy_score(df)
X = df.drop([
        "score_dummy", "report_created_date", "bm25_sum",
        "type_id", "shokushu_id", "score_std"
        #"tfidf_top_average", "recommend_level",
    ], axis=1) # score_dummy以外の列を抽出
print(X.keys())
y = df["score_dummy"].astype(int) # score_dummyの列を抽出
tree = Tree(X, y)
tree.cross_validation()
tree.grid_search()
tree.learning_curve_show(save_file_name="./decision_tree_image/qustion_learning_curve.pdf")
tree.decision_tree_classifier(save_file_name="decision_tree_image/decision_tree_questionnaire.pdf")
tree.logistic_regression()
