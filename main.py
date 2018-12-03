import pandas as pd
from logistic_regression import Tree

df = pd.read_csv('./questionnaire_all_evaluations_preprocessed_from_20181030.csv')
df = df.dropna(subset=["search_word", "st_no", "score"])
df = df.reset_index(drop=True)
drop_list = [
        "report_created_date", "bm25_sum",
        "type_id", "shokushu_id", "score_std",
        "recommend_level", "tfidf_top_average",
        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
        "advice_divide_mecab", "bm25_average", "course_code", "created",
        "keywords", "modified", "score", "score_min_max", "search_word",
        "search_word_wakati", "topic", "st_no",
        "most_highest_similarity", "is_good", "recommend_formula", "evaluation_id", "report_no",
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
    current_max_depth, current_C = max_depth[i], C[i]
    print(i+1, "回目の処理")
    tree.learning_curve_show(
            save_file_name="./tree_image_from_20181030/learning_curve_{}_{}.pdf".format("decision",current_max_depth),
            max_depth=current_max_depth, clf_name="decision")
    tree.learning_curve_show(
            save_file_name="./tree_image_from_20181030/learning_curve_{}_{}.pdf".format("regression",current_C),
            C=current_C, clf_name="regression")
    tree.learning_curve_show(
            save_file_name="./tree_image_from_20181030/learning_curve_{}_{}.pdf".format("random_forest",current_max_depth),
            max_depth=current_max_depth, clf_name="random_forest")
    tree.decision_tree_classifier(save_file_name="tree_image_from_20181030/decision_tree_questionnaire_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
    tree.cross_validation(save_file_name="tree_image_from_20181030/cross_decision_tree_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
    tree.random_forest(random_state=i, max_depth=current_max_depth)
# validation_curveの描画
tree.validation_curve_show( "./tree_image_from_20181030/validation_curve_{}_C.pdf".format("regression"),
        param_range=C, param_name="logisticregression__C", clf_name="regression")
tree.validation_curve_show( "./tree_image_from_20181030/validation_curve_{}_max_depth.pdf".format("decision"),
        param_range=max_depth, param_name="decisiontreeclassifier__max_depth", clf_name="decision")
tree.validation_curve_show( "./tree_image_from_20181030/validation_curve_{}_max_depth.pdf".format("random_forest"),
        param_range=max_depth, param_name="randomforestclassifier__max_depth", clf_name="random_forest")
tree.grid_search()
tree.logistic_regression()


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
    current_max_depth, current_C = max_depth[i], C[i]
    print(i+1, "回目の処理")
    tree.learning_curve_show(
            save_file_name="./tree_image/learning_curve_{}_{}.pdf".format("decision",current_max_depth),
            max_depth=current_max_depth, clf_name="decision")
    tree.learning_curve_show(
            save_file_name="./tree_image/learning_curve_{}_{}.pdf".format("regression",current_C),
            C=current_C, clf_name="regression")
    tree.learning_curve_show(
            save_file_name="./tree_image/learning_curve_{}_{}.pdf".format("random_forest",current_max_depth),
            max_depth=current_max_depth, clf_name="random_forest")
    tree.decision_tree_classifier(save_file_name="tree_image/decision_tree_questionnaire_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
    tree.cross_validation(save_file_name="tree_image/cross_decision_tree_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
    tree.random_forest(random_state=i, max_depth=current_max_depth)
# validation_curveの描画
tree.validation_curve_show( "./tree_image/validation_curve_{}_C.pdf".format("regression"),
        param_range=C, param_name="logisticregression__C", clf_name="regression")
tree.validation_curve_show( "./tree_image/validation_curve_{}_max_depth.pdf".format("decision"),
        param_range=max_depth, param_name="decisiontreeclassifier__max_depth", clf_name="decision")
tree.validation_curve_show( "./tree_image/validation_curve_{}_max_depth.pdf".format("random_forest"),
        param_range=max_depth, param_name="randomforestclassifier__max_depth", clf_name="random_forest")

tree.grid_search()
tree.logistic_regression()


#df = pd.read_csv('./questionnaire_evaluations_preprocessed_all.csv')
#drop_list = [
#        "report_created_date", "bm25_sum",
#        "type_id", "shokushu_id", "score_std", "recommend_level", "tfidf_top_average",
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
#tree = Tree(df)
#tree.add_dummy_score()
#tree.drop_columns(drop_list)
#tree.set_X_and_y(objective_key="score_dummy")
#
#for i in range(0, 8):
#    current_max_depth = max_depth[i]
#    current_C = C[i]
#    print(i+1, "回目の処理")
#    tree.learning_curve_show(
#            save_file_name="./tree_image_all/qustion_learning_curve_{}_{}.pdf".format("decision",current_max_depth),
#            max_depth=current_max_depth,
#            clf_name="decision")
#    tree.learning_curve_show(
#            save_file_name="./tree_image_all/qustion_learning_curve_{}_{}.pdf".format("regression",current_C),
#            C=current_C,
#            clf_name="regression")
#    tree.learning_curve_show(
#            save_file_name="./tree_image_all/qustion_learning_curve_{}_{}.pdf".format("random_forest",current_max_depth),
#            max_depth=current_max_depth,
#            clf_name="random_forest")
#    tree.decision_tree_classifier(save_file_name="tree_image_all/decision_tree_questionnaire_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
#    tree.cross_validation(save_file_name="tree_image_all/cross_decision_tree_{}.pdf".format(current_max_depth), max_depth=current_max_depth)
#    tree.random_forest(random_state=i, max_depth=current_max_depth)
#
## validation_curveの描画
#tree.validation_curve_show( "./tree_image_all/validation_curve_{}_C.pdf".format("regression"),
#        param_range=C, param_name="logisticregression__C", clf_name="regression")
#tree.validation_curve_show( "./tree_image_all/validation_curve_{}_max_depth.pdf".format("decision"),
#        param_range=max_depth, param_name="decisiontreeclassifier__max_depth", clf_name="decision")
#tree.validation_curve_show( "./tree_image_all/validation_curve_{}_max_depth.pdf".format("random_forest"),
#        param_range=max_depth, param_name="randomforestclassifier__max_depth", clf_name="random_forest")
#
#tree.grid_search()
#tree.logistic_regression()


