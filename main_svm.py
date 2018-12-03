import pandas as pd
from svm import Classification

df = pd.read_csv('./questionnaire_evaluations_from_1031_to_1109_all.csv')
classification = Classification(df)
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
classification.add_dummy_score()
classification.drop_columns(drop_list)
classification.set_X_and_y(objective_key="score_dummy")
classification.simple_svm()
classification.std_svm()
classification.grid_svm()

C =     [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
n_neighbors = [1,2,3,4,5,6,7,8]

for i in range(0, 8):
    print("C: ", C[i], "gamma: ", gamma[i])
    classification.learning_curve_show(
            save_file_name="./classifier_image/learning_curve_{}_C{}_G{}.pdf".format("svm", C[i], gamma[i]),
            clf_name="svm", C=C[i], gamma=gamma[i])
    classification.learning_curve_show(
            save_file_name="./classifier_image/learning_curve_{}_n_nei{}.pdf".format("knn", n_neighbors[i]),
            clf_name="knn", n_neighbors=n_neighbors[i])
classification.learning_curve_show( save_file_name="./classifier_image/learning_curve_{}_C_small_G_small.pdf".format("svm"), clf_name="svm", C=0.001, gamma=0.001)
classification.learning_curve_show( save_file_name="./classifier_image/learning_curve_{}_C_small_G_large.pdf".format("svm"), clf_name="svm", C=0.001, gamma=1000)
classification.learning_curve_show( save_file_name="./classifier_image/learning_curve_{}_C_large_G_small.pdf".format("svm"), clf_name="svm", C=1000, gamma=0.001)
classification.learning_curve_show( save_file_name="./classifier_image/learning_curve_{}_C_large_G_large.pdf".format("svm"), clf_name="svm", C=1000, gamma=1000)

classification.validation_curve_show(save_file_name="./classifier_image/validation_curve_svc_C.pdf",
        param_range=C, param_name="svc__C", clf_name="svm")
classification.validation_curve_show(save_file_name="./classifier_image/validation_curve_svc_gamma.pdf",
        param_range=gamma, param_name="svc__gamma", clf_name="svm")
classification.validation_curve_show(save_file_name="./classifier_image/validation_curve_knn_n_nei.pdf",
        param_range=n_neighbors, param_name="kneighborsclassifier__n_neighbors", clf_name="knn")


df = pd.read_csv('./questionnaire_all_evaluations_preprocessed_from_20181030.csv')
classification = Classification(df)
drop_na_list = ["score"]
classification.drop_na(drop_na_list=drop_na_list)
drop_list = [
        "report_created_date", "bm25_sum",
        "type_id", "shokushu_id", "score_std",
        "recommend_level", "tfidf_top_average",
        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
        "advice_divide_mecab", "bm25_average", "course_code", "created",
        "keywords", "modified", "score", "score_min_max", "search_word",
        "search_word_wakati", "topic", "st_no",
        "is_good", "evaluation_id", "recommend_formula",
        "most_highest_similarity",
        #'count_selection', 'diff_date', 'first_final_diff_days',
        #'identification_word_count', 'is_match_keywords',
        #'most_highest_similarity', 'recommend_rank', 'report_created_datetime',
        #'similarity_sum', 'tfidf_sum', 'word_length'
    ] # 不必要なカラム
classification.add_dummy_score()
classification.drop_columns(drop_list)
print('hoge')
classification.set_X_and_y(objective_key="score_dummy")
print('hoge')
classification.simple_svm()
classification.std_svm()
classification.grid_svm()

for i in range(0, 8):
    print("C: ", C[i], "gamma: ", gamma[i])
    classification.learning_curve_show(
            save_file_name="./classifier_image_from_20181030/learning_curve_{}_C{}_G{}.pdf".format("svm", C[i], gamma[i]),
            clf_name="svm", C=C[i], gamma=gamma[i])
    classification.learning_curve_show(
            save_file_name="./classifier_image_from_20181030/learning_curve_{}_n_nei{}.pdf".format("knn", n_neighbors[i]),
            clf_name="knn", n_neighbors=n_neighbors[i])
classification.learning_curve_show( save_file_name="./classifier_image_from_20181030/learning_curve_{}_C_small_G_small.pdf".format("svm"),
        clf_name="svm", C=0.001, gamma=0.001)
classification.learning_curve_show( save_file_name="./classifier_image_from_20181030/learning_curve_{}_C_small_G_large.pdf".format("svm"),
        clf_name="svm", C=0.001, gamma=1000)
classification.learning_curve_show( save_file_name="./classifier_image_from_20181030/learning_curve_{}_C_large_G_small.pdf".format("svm"),
        clf_name="svm", C=1000, gamma=0.001)
classification.learning_curve_show( save_file_name="./classifier_image_from_20181030/learning_curve_{}_C_large_G_large.pdf".format("svm"),
        clf_name="svm", C=1000, gamma=1000)

classification.validation_curve_show(save_file_name="./classifier_image_from_20181030/validation_curve_svc_C.pdf",
        param_range=C, param_name="svc__C", clf_name="svm")
classification.validation_curve_show(save_file_name="./classifier_image_from_20181030/validation_curve_svc_gamma.pdf",
        param_range=gamma, param_name="svc__gamma", clf_name="svm")
classification.validation_curve_show(save_file_name="./classifier_image_from_20181030/validation_curve_knn_n_nei.pdf",
        param_range=n_neighbors, param_name="kneighborsclassifier__n_neighbors", clf_name="knn")


#df = pd.read_csv('./questionnaire_evaluations_preprocessed_all.csv')
#classification = Classification(df)
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
#classification.add_dummy_score()
#classification.drop_columns(drop_list)
#classification.set_X_and_y(objective_key="score_dummy")
#classification.simple_svm()
#classification.std_svm()
#classification.grid_svm()
#
#for i in range(0, 8):
#    print("C: ", C[i], "gamma: ", gamma[i])
#    classification.learning_curve_show(
#            save_file_name="./classifier_image_all/learning_curve_{}_C{}_G{}.pdf".format("svm", C[i], gamma[i]),
#            clf_name="svm", C=C[i], gamma=gamma[i])
#    classification.learning_curve_show(
#            save_file_name="./classifier_image_all/learning_curve_{}_n_nei{}.pdf".format("knn", n_neighbors[i]),
#            clf_name="knn", n_neighbors=n_neighbors[i])
#classification.learning_curve_show( save_file_name="./classifier_image_all/learning_curve_{}_C_small_G_small.pdf".format("svm"),
#        clf_name="svm", C=0.001, gamma=0.001)
#classification.learning_curve_show( save_file_name="./classifier_image_all/learning_curve_{}_C_small_G_large.pdf".format("svm"),
#        clf_name="svm", C=0.001, gamma=1000)
#classification.learning_curve_show( save_file_name="./classifier_image_all/learning_curve_{}_C_large_G_small.pdf".format("svm"),
#        clf_name="svm", C=1000, gamma=0.001)
#classification.learning_curve_show( save_file_name="./classifier_image_all/learning_curve_{}_C_large_G_large.pdf".format("svm"),
#        clf_name="svm", C=1000, gamma=1000)
#
#classification.validation_curve_show(save_file_name="./classifier_image_all/validation_curve_svc_C.pdf",
#        param_range=C, param_name="svc__C", clf_name="svm")
#classification.validation_curve_show(save_file_name="./classifier_image_all/validation_curve_svc_gamma.pdf",
#        param_range=gamma, param_name="svc__gamma", clf_name="svm")
#classification.validation_curve_show(save_file_name="./classifier_image_all/validation_curve_knn_n_nei.pdf",
#        param_range=n_neighbors, param_name="kneighborsclassifier__n_neighbors", clf_name="knn")

