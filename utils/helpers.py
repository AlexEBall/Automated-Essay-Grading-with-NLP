import os
import codecs
import spacy
import itertools as it
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Word2Vec

nlp = spacy.load('en_core_web_md')

intermediate_directory = os.path.join('../data/intermediate')
lda_model_filepath = os.path.join(intermediate_directory, 'lda_model_all')

# load the finished LDA model from disk
lda = LdaMulticore.load(lda_model_filepath)

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

def adding_stanford_nlp_groups_NER_to_stop_words():
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

def removing_stanford_nlp_groups_NER_from_stop_words():
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

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename,):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename), batch_size=100, n_threads=4):
        
        for sent in parsed_review.sents:
            yield ' '.join([token.lemma_ for token in sent if not punct_space_stop(token)])

def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    
    for essay in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(essay)


def explore_topic(topic_number, topn=5):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """

    print('{:20} {}'.format('term', 'frequency') + '\n')

    for term, frequency in lda.show_topic(topic_number, topn=5):
        print('{:20} {:.3f}'.format(term, round(frequency, 3)))


def get_sample_essay(essay_number):
    """
    retrieve a particular review index
    from the reviews file and return it
    """

    return list(it.islice(line_review(essay_set1_txt_filepath), essay_number, essay_number+1))[0]


def lda_description(essay_text, min_topic_freq=0.05):
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """

    # parse the essay text with spaCy
    parsed_essay = nlp(essay_text)

    # lemmatize the text and remove punctuation and whitespace
    unigram_essay = [token.lemma_ for token in parsed_essay
                     if not punct_space_stop(token)]

    # apply the first-order and secord-order phrase models
    bigram_essay = bigram_model[unigram_essay]
    trigram_essay = trigram_model[bigram_essay]

    # create a bag-of-words representation
    essay_bow = trigram_dictionary.doc2bow(trigram_essay)

    # create an LDA representation
    essay_lda = lda[essay_bow]

    # sort with the most highly related topics first
    essay_lda = sorted(essay_lda)

    for topic_number, freq in essay_lda:
        if freq < min_topic_freq:
            break

        # print the most highly related topic names and frequencies
        print('{:25} {}'.format(topic_names[topic_number], round(freq, 3)))

def lda_description_for_building(essay_text, min_topic_freq=0.05):
        
    # parse the essay text with spaCy
    parsed_essay = nlp(essay_text)

    # lemmatize the text and remove punctuation and whitespace
    unigram_essay = [token.lemma_ for token in parsed_essay
                     if not punct_space_stop(token)]

    # apply the first-order and secord-order phrase models
    bigram_essay = bigram_model[unigram_essay]
    trigram_essay = trigram_model[bigram_essay]

    # create a bag-of-words representation
    essay_bow = trigram_dictionary.doc2bow(trigram_essay)

    # create an LDA representation
    essay_lda = lda[essay_bow]

    # sort with the most highly related topics first
    essay_lda = sorted(essay_lda)

    for topic_number, freq in essay_lda:

        topics.append(topic_names[topic_number])
        freqs.append(round(freq, 3))
        
    # return topic and fre
    return list(zip(topics, freqs))

def get_related_terms(token, topn=5):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """

    for word, similarity in essay2vec_model.wv.most_similar(positive=[token], topn=topn):

        print('{:20} {}'.format(word, round(similarity, 3)))

def create_DF_from_tuple(tuple):
        
    return pd.DataFrame({k:v for k,*v in tuple})

# This function is a bit inefficient will try to refactor
def process_topic_and_score_df(topic_names):
    
    processed_df = pd.DataFrame(columns=['essay_id', 'essay_set', 'essay', 'rater1_domain1', 'rater2_domain1',
       'domain1_score', 'prompt', 'has_source_material', 'grade_7', 'grade_8',
       'grade_10', 'bad_if_kids_spend_too_much_time',
       'doesnt_have_the_negative_exercise_effect', 'games_and_information',
       'looking_at_websites_for_info', 'spend_time_looking_on_websites'])
    for n in range(len(essays)):
        tuple = lda_description_for_building(get_sample_essay(n))
        topic_scores = create_DF_from_tuple(tuple)
        processed_topic_scores = pd.concat([topic_scores, topic_names], sort=True).drop_duplicates().reset_index(drop=True).fillna(0)
        indexNamesArr = processed_topic_scores.index.values
        indexNamesArr[0] = n
        merged_dfs = pd.concat([essays.loc[[n]], processed_topic_scores], axis=1, sort=False)
        processed_df = processed_df.append(merged_dfs)

    return processed_df