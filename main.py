import pandas as pd
from logistic_regression import Tree, preprocess

df = pd.read_csv('./questionnaire_evaluations_from_1031_to_1109_all.csv')
df = preprocess(df)
X = df.drop([
        "score_dummy", "report_created_date", "bm25_sum",
        "type_id", "shokushu_id", "score_std",
        "recommend_level", "tfidf_top_average",
        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
        #'identification_word_count', 'is_match_keywords',
        #'most_highest_similarity', 'recommend_rank', 'report_created_datetime',
        #'similarity_sum', 'tfidf_sum', 'word_length'
    ], axis=1) # score_dummy以外の列を抽出
#print(X.keys())
y = df["score_dummy"].astype(int) # score_dummyの列を抽出
tree = Tree(X, y)
for i in range(2, 10):
    tree.decision_tree_classifier(save_file_name="decision_tree_image/decision_tree_questionnaire_{}.pdf".format(i), max_depth=i)
    tree.learning_curve_show(save_file_name="./decision_tree_image/qustion_learning_curve_{}.pdf".format(i), max_depth=i)
    tree.cross_validation(save_file_name="decision_tree_image/cross_decision_tree_{}.pdf".format(i), max_depth=i)

tree.grid_search()
tree.logistic_regression()




print(X.keys())
y = df["score_dummy"].astype(int) # score_dummyの列を抽出
tree = Tree(X, y)
import pandas as pd
train_test = {}
for index_y, max_depth in enumerate(range(2, 10)):
    train_test[index_y] = []
    for index_x, i in enumerate(range(0,10)):
        train_test[index_y].append(tree.tmp_random_forest(min_samples_leaf=max_depth,random_state=i))

for t in train_test.values():
    train = 0
    test = 0
    size = len(t)
    for value in t:
        train += value["train"]
        test += value["test"]

    print("=======")
    print((train / size) - (test / size))

