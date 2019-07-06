import pandas as pd
import numpy as np
from utils.helpers import uniqueColumns, printEssaySetStats
from data.intermediate.essay_eicts.essay_dictionaries import essay_prompts, essay_gradeLevels, essay_sourceDependent

training_essay_set = pd.read_excel('../../data/raw/training_set_rel3.xlsx')

print('Total number of esseays in the dataset: {}'.format(len(training_essay_set)))

# if you want to see the unique column names, run this line
# uniqueColumns(training_essay_set)

# separate essays into distinct sets
set1 = training_essay_set[training_essay_set.essay_set == 1]
set2 = training_essay_set[training_essay_set.essay_set == 2]
set3 = training_essay_set[training_essay_set.essay_set == 3]
set4 = training_essay_set[training_essay_set.essay_set == 4]
set5 = training_essay_set[training_essay_set.essay_set == 5]
set6 = training_essay_set[training_essay_set.essay_set == 6]
set7 = training_essay_set[training_essay_set.essay_set == 7]
set8 = training_essay_set[training_essay_set.essay_set == 8]

# if you want to see stats from each essay set, run these lines
# printEssaySetStats(set1)
# printEssaySetStats(set2)
# printEssaySetStats(set3)
# printEssaySetStats(set4)
# printEssaySetStats(set5)
# printEssaySetStats(set6)
# printEssaySetStats(set7)
# printEssaySetStats(set8)

# map dictionaries to dataframe
training_essay_set['prompt'] = training_essay_set.essay_set.map(essay_prompts)
training_essay_set['grade_level'] = training_essay_set.essay_set.map(essay_gradeLevels)
training_essay_set['has_source_material'] = training_essay_set.essay_set.map(essay_sourceDependent)

# read in source essay matrial for certain essays
source3 = open('./data/sourceEssay3.txt')
source4 = open('./data/sourceEssay4.txt')
source5 = open('./data/sourceEssay5.txt')
source6 = open('./data/sourceEssay6.txt')

# create a source essay dictionary and map it to dataframe
essay_sourceText = {
    1: np.nan,
    2: np.nan,
    3: source3.read(),
    4: source4.read(),
    5: source5.read(),
    6: source6.read(),
    7: np.nan,
    8: np.nan,
}
training_essay_set['source_text'] = training_essay_set.essay_set.map(essay_sourceText)

# create one hot encoded columns for categorical data
gradeDF = pd.get_dummies(training_essay_set['grade_level'], prefix='grade')

# concat that to the dataframe and drop unnecessary column
training_essay_set = pd.concat([training_essay_set, gradeDF], axis=1)
training_essay_set.drop(['grade_level', 7, 8, 10], axis=1, inplace=True)

# save dataframe to csv
training_essay_set.to_csv('../../data/intermediate/prepped_essays_df.csv', index=False)