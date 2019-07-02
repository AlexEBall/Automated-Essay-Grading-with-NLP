import os
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

from utils.helpers import trigram_bow_generator, explore_topic

import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle

intermediate_directory = os.path.join('../../data/intermediate')
trigram_essays_all_filepath = os.path.join(intermediate_directory, 'trigram_essays_all.txt')
trigram_dictionary_filepath = os.path.join(intermediate_directory, 'trigram_dict_all.dict')

# this is a bit time consuming - make the if statement True
# if you want to learn the dictionary yourself.
if 0 == 1:

    trigram_essays = LineSentence(trigram_essays_all_filepath)

    # learn the dictionary by iterating over all of the reviews
    trigram_dictionary = Dictionary(trigram_essays)
    
    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
    trigram_dictionary.compactify()

    trigram_dictionary.save(trigram_dictionary_filepath)
    
# load the finished dictionary from disk
trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)

trigram_bow_filepath = os.path.join(intermediate_directory, 'trigram_bow_corpus_all.mm')

# this is a bit time consuming - make the if statement True
# if you want to build the bag-of-words corpus yourself.
if 0 == 1:

    # generate bag-of-words representations for
    # all reviews and save them as a matrix
    MmCorpus.serialize(trigram_bow_filepath, trigram_bow_generator(trigram_essays_all_filepath))
    
# load the finished bag-of-words corpus from disk
trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

lda_model_filepath = os.path.join(intermediate_directory, 'lda_model_all')

# this is a bit time consuming - make the if statement True
# if you want to train the LDA model yourself.
if 0 == 1:

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # workers => sets the parallelism, and should be
        # set to your number of physical cores minus one
        lda = LdaMulticore(trigram_bow_corpus,
                           num_topics=5,
                           id2word=trigram_dictionary,
                           workers=3)
    
    lda.save(lda_model_filepath)
    
# load the finished LDA model from disk
lda = LdaMulticore.load(lda_model_filepath)

explore_topic(topic_number=0)

topic_names = {0: 'computers are useful for looking at websites for school',
               1: 'computers are useful to spend time online playing games',
               2: 'computers help kids learn about the world',
               3: 'computers are useful to look for information',
               4: 'computers are useful to access information at school'}

topic_names_filepath = os.path.join(intermediate_directory, 'topic_names.pkl')

with open(topic_names_filepath, 'wb') as f:
    pickle.dump(topic_names, f)

LDAvis_data_filepath = os.path.join(intermediate_directory, 'ldavis_prepared')

# this is a bit time consuming - make the if statement True
# if you want to execute data prep yourself.
if 0 == 1:

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus, trigram_dictionary)

    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)