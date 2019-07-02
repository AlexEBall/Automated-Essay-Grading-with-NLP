import os
import codecs
import spacy
import itertools as it
import pandas as pd
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

from utils.helpers import adding_stanford_nlp_groups_NER_to_stop_words, removing_stanford_nlp_groups_NER_from_stop_words, punct_space_stop, line_review, lemmatized_sentence_corpus
nlp = spacy.load('en_core_web_md')

essays = pd.read_csv('../../data/intermediate/prepped_essays_df.csv')

# ----------- ISOLATE JUST ESSAYS FROM 1ST SET ------------ #
essays = essays[essays['essay_set'] == 1]
essays.dropna(axis=1, how='all', inplace=True)

intermediate_directory = os.path.join('../../data/intermediate')

essay_set1_txt_filepath = os.path.join(intermediate_directory, 'essay_set1_text_all.txt')


# ----------- WRITE ALL ESSAYS TO A .TXT FILE ------------ #
if 0 == 1:
    essay_count = 0

    # create & open a new file in write mode
    with codecs.open(essay_set1_txt_filepath, 'w', encoding='utf_8') as essay_set1_txt_file:

        # loop through all essays in the dataframe
        for row in essays.itertuples():
            # write the essay as a line in the new file and escape newline characters in the original essays
            essay_set1_txt_file.write(row.essay.replace('\n', '\\n') + '\n')
            essay_count += 1

        print('Text from {:,} essays written to the new txt file.'.format(essay_count))

else:

    with codecs.open(essay_set1_txt_filepath, encoding='utf_8') as essay_set1_txt_file:
        for essay_count, line in enumerate(essay_set1_txt_file):
            pass

        print('Text from {:,} essays in the txt file.'.format(essay_count + 1))


# ----------- USING SPACY ON A SINGLE ESSAY ------------ #
# Run the commands below to look more into a specific essay

# test_essay = essays.iloc[0, 2]
# parsed_essay = nlp(test_essay)

# for num, sentence in enumerate(parsed_essay.sents):
#     print('Sentence {}:'.format(num + 1))
#     print(sentence)
#     print('')

# for num, entity in enumerate(parsed_essay.ents):
#     print('Entity {}:'.format(num + 1), entity, '-', entity.label_)
#     print('')

# token_text = [token.orth_ for token in parsed_essay]
# token_pos = [token.pos_ for token in parsed_essay]

# pd.DataFrame(zip(token_text, token_pos), columns=['token_text', 'part_of_speech'])

# token_lemma = [token.lemma_ for token in parsed_essay]
# token_shape = [token.shape_ for token in parsed_essay]

# pd.DataFrame(zip(token_text, token_lemma, token_shape), columns=['token_text', 'token_lemma', 'token_shape'])

# token_entity_type = [token.ent_type_ for token in parsed_essay]
# token_entity_iob = [token.ent_iob_ for token in parsed_essay]

# pd.DataFrame(zip(token_text, token_entity_type, token_entity_iob), columns=['token_text', 'entity_type', 'inside_outside_begin'])

# token_attributes = [(token.orth_,
#                      token.prob,
#                      token.is_stop,
#                      token.is_punct,
#                      token.is_space,
#                      token.like_num,
#                      token.is_oov)
#                     for token in parsed_essay]

# df = pd.DataFrame(token_attributes,
#                   columns=['text',
#                            'log_probability',
#                            'stop?',
#                            'punctuation?',
#                            'whitespace?',
#                            'number?',
#                            'out of vocab.?'])

# df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?'].applymap(lambda x: u'Yes' if x else u''))

# df


# ----------- ADD OR REMOVE STOP WORDS ------------ #
# removing_stanford_nlp_groups_NER_from_stop_words(nlp)
adding_stanford_nlp_groups_NER_to_stop_words()


# ----------- LOOKING AT UNIGRAMS ------------ #
unigram_sentences_filepath = os.path.join(intermediate_directory, 'unigram_sentences_all_essays.txt')
essays_set1_all_filepath = os.path.join(intermediate_directory, 'essay_set1_text_all.txt')

# this is a bit time consuming - make the if statement True
# if you want to execute data prep yourself.
if 0 == 1:

    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in lemmatized_sentence_corpus(essays_set1_all_filepath, codecs, nlp):
            f.write(sentence + '\n')

unigram_sentences = LineSentence(unigram_sentences_filepath)

# for unigram_sentence in it.islice(unigram_sentences, 19, 42):
#     print(' '.join(unigram_sentence))
#     print('')

# ----------- LOOKING AT BIGRAMS ------------ #
bigram_model_filepath = os.path.join(intermediate_directory, 'bigram_model_all')

# this is a bit time consuming - make the if statement True
# if you want to execute modeling yourself.
if 0 == 1:
    bigram_model = Phrases(unigram_sentences)

    bigram_model.save(bigram_model_filepath)

# load the finished model from disk
bigram_model = Phrases.load(bigram_model_filepath)

bigram_sentences_filepath = os.path.join(intermediate_directory, 'bigram_sentences_all.txt')

# this is a bit time consuming - make the if statement True
# if you want to execute data prep yourself.
if 0 == 1:

    with codecs.open(bigram_sentences_filepath, 'w', encoding='utf_8') as f:

        for unigram_sentence in unigram_sentences:
            bigram_sentence = ' '.join(bigram_model[unigram_sentence])

            f.write(bigram_sentence + '\n')

bigram_sentences = LineSentence(bigram_sentences_filepath)

# for bigram_sentence in it.islice(bigram_sentences, 19, 42):
#     print(' '.join(bigram_sentence))
#     print('')

# ----------- LOOKING AT TRIGRAMS ------------ #
trigram_model_filepath = os.path.join(intermediate_directory, 'trigram_model_all')

# this is a bit time consuming - make the if statement True
# if you want to execute modeling yourself.
if 0 == 1:
    trigram_model = Phrases(bigram_sentences)

    trigram_model.save(trigram_model_filepath)

# load the finished model from disk
trigram_model = Phrases.load(trigram_model_filepath)

trigram_sentences_filepath = os.path.join(intermediate_directory, 'trigram_sentences_all.txt')

if 0 == 1:

    with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:

        for bigram_sentence in bigram_sentences:
            trigram_sentence = ' '.join(trigram_model[bigram_sentence])

            f.write(trigram_sentence + '\n')

trigram_sentences = LineSentence(trigram_sentences_filepath)

# for trigram_sentence in it.islice(trigram_sentences, 205, 245):
#     print(' '.join(trigram_sentence))
#     print('')

# ----------- RUN THE TRIGRAMS PHRASE MODEL ON ALL ESSAYS ------------ #
trigram_essays_all_filepath = os.path.join(intermediate_directory, 'trigram_essays_all.txt')

# this is a bit time consuming - make the if statement True
# if you want to execute data prep yourself.
if 0 == 1:

    with codecs.open(trigram_essays_all_filepath, 'w', encoding='utf_8') as f:

        for parsed_essay in nlp.pipe(line_review(essays_set1_all_filepath), batch_size=100, n_threads=4):
            # lemmatize the text, removing punctuation and whitespace
            unigram_essays = [token.lemma_ for token in parsed_essay if not punct_space_stop(token)]

            # apply the first-order and second-order phrase models
            bigram_essays = bigram_model[unigram_essays]
            trigram_essays = trigram_model[bigram_essays]

            # write the transformed review as a line in the new file
            trigram_essays = ' '.join(trigram_essays)
            f.write(trigram_essays + '\n')

# print('Original:' + '\n')

# for essay in it.islice(line_review(essays_set1_all_filepath), 301, 302):
#     print(essay)

# print('----' + '\n')
# print('Transformed:' + '\n')

with codecs.open(trigram_essays_all_filepath, encoding='utf_8') as f:
    for essay in it.islice(f, 301, 302):
        print(essay)