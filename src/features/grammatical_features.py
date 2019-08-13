import enchant 
import pandas as pd
import spacy
import re
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import urllib.request
from bs4 import BeautifulSoup

# Load spaCy model
nlp = spacy.load('en_core_web_md')

essays = pd.read_csv('../../data/intermediate/prepped_essays_df.csv')

# ----------- Isolate essays from the 6th set ------------ #
essays = essays[essays['essay_set'] == 6]
essays.dropna(axis=1, how='all', inplace=True)

# load the enchant dictionary
d = enchant.Dict("en_US")

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

# number of sentences
def get_number_of_sentences(df):
    """
    Function that returns the number of sentences
    in the document. Could be useful for run-on sentences
    Assign them to the df
    """
    
    _byRow_sents = []
    for essay in df['essay']:
        sents = []
        parsed_essay = nlp(essay)
        for num, sentence in enumerate(parsed_essay.sents):
            sents.append(sentence)
        
        _byRow_sents.append(len(sents))
        
    return df.assign(n_sentences = _byRow_sents)

essays = get_number_of_sentences(essays)

# Get Parts of Speech
def get_list_of_number_of_pos(df):
    """
    Function that parses the essay for each words POS
    Returns tuples containg for now, nouns, verbs, adverbs and adjectives
    """
    pos = []
    
    for essay in df['essay']:
        parsed_essay = nlp(essay)
        token_pos = [token.pos_ for token in parsed_essay]
        
        pos.append((token_pos.count('NOUN'), token_pos.count('VERB'), token_pos.count('ADV'), token_pos.count('ADJ')))
        
    return pos

pos_list = get_list_of_number_of_pos(essays)
nouns, verbs, adverbs, adjectives = zip(*pos_list)

# Assign them to the dataframe
essays = essays.assign(nouns = nouns)
essays = essays.assign(verbs = verbs)
essays = essays.assign(adverbs = adverbs)
essays = essays.assign(adjectives = adjectives)


# -------- Use Beautiful Soup to scrape for the top 500 SAT words ------- #
f = open('../../data/intermediate/out.txt','w')

url = "https://satvocabulary.us/"
page = urllib.request.urlopen(url)

soup = BeautifulSoup(page)
soup.unicode

tab = soup.find("table", {"class":"WORDLIST"})
# tab
rows = tab.find_all('tr')
rows

top500SATWords = []

for row in rows:
    data = row.find_all("td")
    
    top500SATWords.append(re.sub("[\(\[].*?[\)\]]", "", ((data[1].get_text()).split(', ')[0])))

# Remove the first element from the list because it's not actually a word
top500SATWords.pop(0)

# lemmatize the top500 SAT words
top500_joined = ' '.join(str(v) for v in top500SATWords)
token_lemma_top500 = [token.lemma_ for token in nlp(top500_joined)]

def calculate_number_of_top500_SAT_words(parsed_essay, list_of_word_to_check_against):
    """
    Function that calculates the number of top 500 SAT words in the passed in parsed_essay
    """

    list_of_word_to_check_against = sorted(list(set(list_of_word_to_check_against)))
    token_attributes = [(token.lemma_, token.is_stop, token.is_punct) for token in parsed_essay]
    
    df = pd.DataFrame(token_attributes, columns=['text', 'stop', 'punctuation'])
    df2 = df[df['stop'] != True]
    df3 = df2[df2['punctuation'] != True]
    
    low = df3['text'].tolist()
    low  = [word.lower() for word in low]
    
    count = sum(el in low for el in list_of_word_to_check_against)
    return count

def buildDFSAT(df):
    """
    Function that assigns the top500 SAT word occurences to the dataframe
    """
    n_top500 = []
    for essay in df['essay']:
        
        parsed_essay = nlp(essay)
        n_top500.append(calculate_number_of_top500_SAT_words(parsed_essay, token_lemma_top500))        
    
    essays = df.assign(n_top500 = n_top500) 
    return essays
        
essays = buildDFSAT(essays)