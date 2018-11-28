import pandas as pd
from logistic_regression import Tree

df = pd.read_csv('./questionnaire_evaluations_from_1031_to_1109_all.csv')
drop_list = [
        "report_created_date", "bm25_sum",
        "type_id", "shokushu_id", "score_std",
        "recommend_level", "tfidf_top_average",
        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
        "advice_divide_mecab", "bm25_average", "course_code", "created",
        "keywords", "modified", "score", "score_min_max", "search_word",
        "search_word_wakati", "topic", "st_no",
        #'count_selection', 'diff_date', 'first_final_diff_days',
        #'identification_word_count', 'is_match_keywords',
        #'most_highest_similarity', 'recommend_rank', 'report_created_datetime',
        #'similarity_sum', 'tfidf_sum', 'word_length'
    ] # 不必要なカラム
#tree = Tree(df)
#tree.add_dummy_score()
#tree.drop_columns(drop_list)
#tree.set_X_and_y(objective_key="score_dummy")
#for i in range(2, 10):
#    tree.decision_tree_classifier(save_file_name="decision_tree_image/decision_tree_questionnaire_{}.pdf".format(i), max_depth=i)
#    tree.learning_curve_show(save_file_name="./decision_tree_image/qustion_learning_curve_{}.pdf".format(i), max_depth=i)
#    tree.cross_validation(save_file_name="decision_tree_image/cross_decision_tree_{}.pdf".format(i), max_depth=i)
#    tree.random_forest(min_samples_leaf=i, random_state=i, max_depth=i)
#tree.grid_search()
#tree.logistic_regression()

df = pd.read_csv('./questionnaire_evaluations_preprocessed_all.csv')
drop_list = [
        "report_created_date", "bm25_sum",
        "type_id", "shokushu_id", "score_std",
        "recommend_level", "tfidf_top_average",
        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
        "advice_divide_mecab", "bm25_average", "course_code", "created",
        "keywords", "modified", "score", "score_min_max", "search_word",
        "search_word_wakati", "topic", "st_no",
        "is_good", "evaluation_id", "recommend_formula",
        #'count_selection', 'diff_date', 'first_final_diff_days',
        #'identification_word_count', 'is_match_keywords',
        #'most_highest_similarity', 'recommend_rank', 'report_created_datetime',
        #'similarity_sum', 'tfidf_sum', 'word_length'
    ] # 不必要なカラム
tree = Tree(df)
tree.add_dummy_score()
tree.drop_columns(drop_list)
tree.set_X_and_y(objective_key="score_dummy")
print(tree.X.keys())
for i in range(2, 10):
    tree.random_forest(min_samples_leaf=i, random_state=i, max_depth=i)

