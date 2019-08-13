import enchant 
import pandas as pd
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer

essays = pd.read_csv('../../data/intermediate/prepped_essays_df.csv')

# ----------- Isolate essays from the 6th set ------------ #
essays = essays[essays['essay_set'] == 6]
essays.dropna(axis=1, how='all', inplace=True)

def getCorrectAndIncorrectSpelling(seriesOfEssays):
    """
    Returns a list of tuples:
    number of mispelled words, number of correctly spelled words 
    """
    _byRow = []

    for essay in seriesOfEssays:
        _individual = []
        for word in essay.split():

            _individual.append(d.check(word))

        _byRow.append((_individual.count(False), _individual.count(True)))

    return _byRow

# ------ Use Enchant to get the number of misspelt and correctly spelt words ------ #
spellings = getCorrectAndIncorrectSpelling(essays['essay'])

# create a list from the tuples
list1, list2 = zip(*spellings)

# assign them to the dataframe
essays = essays.assign(misspelt = list1) 
essays = essays.assign(correct = list2) 

# ----- Create a RegEx Tokenizer to better split actual words ----- #
tokenizer = RegexpTokenizer(r'\w+')

# Number of words in essay
def get_essay_lengths(regExTokenizer, df):
    """
    Function that gets the number of words in an
    essay, not including punctuation and assigns them to the df
    """
    length = []
    for essay in df['essay']:
        length.append(len(regExTokenizer.tokenize(essay)))
        
    return df.assign(length = length) 

essays = get_essay_lengths(tokenizer, essays)

# lexical diversity
def get_lexical_diversity(regExTokenizer, df):
    """
    Function that measures lexical diversity which is
    The ratio of total words to unique words
    Then assign that to the df
    """
    ld = []
    for essay in df['essay']:
        
        ld.append(round(len(regExTokenizer.tokenize(essay)) / float(len(set(regExTokenizer.tokenize(essay)))), 2))
    return df.assign(lexical_diversity = ld)

essays = get_lexical_diversity(tokenizer, essays)


