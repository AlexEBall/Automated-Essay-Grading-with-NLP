def uniqueColumns(df):
    """  
    pretty print unique columns   
    """
    for column in df.columns:
        print(column)

def printEssaySetStats(df):
    """  
    pretty print the essay stats, especially how many null values there are in each column  
    """

    print('Essay Set #{0} Length of dataset {1}'.format(df.essay_set.unique()[0], len(df)))
    print(df.isnull().sum(axis=0))
    print('\n')

def adding_stanford_nlp_groups_NER_to_stop_words(nlp):
    """
    helper funciton to add Stanford NLP Group NERs to spaCy stop words
    range of 0 - 15
    """

    for number in list(range(0, 16)):
        nlp.vocab['@ORGANIZATION' + str(number)].is_stop = True
        nlp.vocab['@PERSON' + str(number)].is_stop = True
        nlp.vocab['@CAPS' + str(number)].is_stop = True
        nlp.vocab['@LOCATION' + str(number)].is_stop = True
        nlp.vocab['@DATE' + str(number)].is_stop = True
        nlp.vocab['@TIME' + str(number)].is_stop = True
        nlp.vocab['@MONEY' + str(number)].is_stop = True
        nlp.vocab['@PERCENT' + str(number)].is_stop = True
        nlp.vocab['@MONTH' + str(number)].is_stop = True
        nlp.vocab['@EMAIL' + str(number)].is_stop = True
        nlp.vocab['@NUM' + str(number)].is_stop = True
        nlp.vocab['@DR' + str(number)].is_stop = True
        nlp.vocab['@CITY' + str(number)].is_stop = True
        nlp.vocab['@STATE' + str(number)].is_stop = True

def removing_stanford_nlp_groups_NER_from_stop_words(nlp):
    """
    helper funciton to remove Stanford NLP Group NERs from spaCy stop words
    range of 0 - 15
    """

    for number in list(range(0, 16)):
        nlp.vocab['@ORGANIZATION' + str(number)].is_stop = False
        nlp.vocab['@PERSON' + str(number)].is_stop = False
        nlp.vocab['@CAPS' + str(number)].is_stop = False
        nlp.vocab['@LOCATION' + str(number)].is_stop = False
        nlp.vocab['@DATE' + str(number)].is_stop = False
        nlp.vocab['@TIME' + str(number)].is_stop = False
        nlp.vocab['@MONEY' + str(number)].is_stop = False
        nlp.vocab['@PERCENT' + str(number)].is_stop = False
        nlp.vocab['@MONTH' + str(number)].is_stop = False
        nlp.vocab['@EMAIL' + str(number)].is_stop = False
        nlp.vocab['@NUM' + str(number)].is_stop = False
        nlp.vocab['@DR' + str(number)].is_stop = False
        nlp.vocab['@CITY' + str(number)].is_stop = False
        nlp.vocab['@STATE' + str(number)].is_stop = False

def punct_space_stop(token):
    """
    helper function to eliminate tokens
    that are pure punctuation, whitespace or stopwords
    """
    
    return token.is_punct or token.is_space or token.is_stop

def line_review(filename, codecs):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename, codecs, nlp):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename, codecs), batch_size=100, n_threads=4):
        
        for sent in parsed_review.sents:
            yield ' '.join([token.lemma_ for token in sent if not punct_space_stop(token)])
