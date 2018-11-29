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
        "most_highest_similarity",
        #'count_selection', 'diff_date', 'first_final_diff_days',
        #'identification_word_count', 'is_match_keywords',
        #'recommend_rank', 'report_created_datetime',
        #'similarity_sum', 'tfidf_sum', 'word_length'
    ] # 不必要なカラム
tree = Tree(df)
tree.add_dummy_score()
tree.drop_columns(drop_list)
tree.set_X_and_y(objective_key="score_dummy")
max_depth = [2,3,4,5,6,7,8,9]
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ]
for i in range(0, 8):
    current_max_depth = max_depth[i]
    current_C = C[i]
    print(i+1, "回目の処理")
    tree.learning_curve_show(
            save_file_name="./decision_tree_image/qustion_learning_curve_{}_{}.pdf".format("decision",current_max_depth),
            max_depth=current_max_depth,
            clf_name="decision")
    tree.learning_curve_show(
            save_file_name="./decision_tree_image/qustion_learning_curve_{}_{}.pdf".format("regression",current_C),
            C=current_C,
            clf_name="regression")
    tree.learning_curve_show(
            save_file_name="./decision_tree_image/qustion_learning_curve_{}_{}.pdf".format("random_forest",current_max_depth),
            max_depth=current_max_depth,
            clf_name="random_forest")
    tree.decision_tree_classifier(save_file_name="decision_tree_image/decision_tree_questionnaire_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
    tree.cross_validation(save_file_name="decision_tree_image/cross_decision_tree_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
    tree.random_forest(random_state=i, max_depth=current_max_depth)
tree.grid_search()
tree.logistic_regression()


#df = pd.read_csv('./questionnaire_evaluations_preprocessed_all.csv')
#drop_list = [
#        "report_created_date", "bm25_sum",
#        "type_id", "shokushu_id", "score_std",
#        "recommend_level", "tfidf_top_average",
#        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
#        "advice_divide_mecab", "bm25_average", "course_code", "created",
#        "keywords", "modified", "score", "score_min_max", "search_word",
#        "search_word_wakati", "topic", "st_no",
#        "is_good", "evaluation_id", "recommend_formula",
#        "most_highest_similarity",
#        #'count_selection', 'diff_date', 'first_final_diff_days',
#        #'identification_word_count', 'is_match_keywords',
#        #'most_highest_similarity', 'recommend_rank', 'report_created_datetime',
#        #'similarity_sum', 'tfidf_sum', 'word_length'
#    ] # 不必要なカラム

