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

df = pd.read_csv('./questionnaire_evaluations_preprocessed_all.csv')
classification = Classification(df)
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
classification.set_X_and_y(objective_key="score_dummy")
classification.simple_svm()
classification.std_svm()
classification.grid_svm()

