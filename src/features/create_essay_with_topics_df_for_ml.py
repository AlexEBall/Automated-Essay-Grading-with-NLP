import pandas as pd
import numpy as np
from utils.helpers import lda_description_for_building, get_sample_essay, create_DF_from_tuple, process_topic_and_score_df

# Topic names from LDA analysis
# TODO: Refactor out into it's own space
topic_names = {0: 'looking_at_websites_for_info',
               1: 'doesnt_have_the_negative_exercise_effect',
               2: 'spend_time_looking_on_websites',
               3: 'games_and_information',
               4: 'bad_if_kids_spend_too_much_time'}

col_names = pd.DataFrame(columns=list(topic_names.values()))

# this is a bit time consuming - make the if statement True
# if you want to create the dataframe yourself.
if 0 == 1:

    essays_with_topic_scores = process_topic_and_score_df(col_names)

essays_with_topic_scores = essays_with_topic_scores.drop(['essay_set', 'essay', 'rater1_domain1', 'rater2_domain1', 'prompt', 'has_source_material', 'grade_7', 'grade_8', 'grade_10'], axis=1)
essays_with_topic_scores.to_csv('../data/processed/essays_with_topic_scores.csv')