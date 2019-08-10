import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.plotly as py
import cufflinks
import plotly.figure_factory as ff
from plotly.offline import iplot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

%matplotlib inline
sns.set_style("darkgrid")

essays = pd.read_csv('../../data/intermediate/prepped_essays_df.csv')

'''
Here you can toggle on/off different essay sets to see similar visualizaitons for each set
Take a look at the images folder in this directory to see what kind of images you can produce
''' 

essays1 = essays[essays['essay_set'] == 1]
# essays2 = essays[essays['essay_set'] == 2]
# essays3 = essays[essays['essay_set'] == 3]
# essays4 = essays[essays['essay_set'] == 4]
# essays5 = essays[essays['essay_set'] == 5]
# essays6 = essays[essays['essay_set'] == 6]
# essays7 = essays[essays['essay_set'] == 7]
# essays8 = essays[essays['essay_set'] == 8]

essays1.dropna(axis=1, how='all', inplace=True)

# This creates a histogram of score distribution
essays1['domain1_score'].iplot(
    kind='hist',
    xTitle='score',
    linecolor='black',
    yTitle='count',
    title='Essay Set 1 Scores Distribution')


essays1['length'] = essays1.essay.str.len()

# This creates a histogram of essay length
essays1['length'].iplot(
    kind='hist',
    bins=100,
    xTitle='essay length',
    linecolor='black',
    yTitle='count',
    title='Essay Set 1 Length Distribution')

def get_top_n_words(corpus, stopwords=False, n=None):
    if stopwords == True:
        vec = CountVectorizer(stop_words = 'english').fit(corpus)
    else:
        vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Generate a plotly histogram of unigrams with stopwords
common_words = get_top_n_words(essays1.essay, n=20)
    
df1 = pd.DataFrame(common_words, columns = ['essay' , 'count'])
df1.groupby('essay').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in essays before removing stop words')

# Generate a plotly histogram of unigrams without stopwords
common_words = get_top_n_words(essays1['essay'], stopwords=True, n=20)
    
df2 = pd.DataFrame(common_words, columns = ['essay' , 'count'])
df2.groupby('essay').sum()['count'].sort_values(ascending=False).iplot(
        kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in essays after removing stop words')


def get_top_n_bigram(corpus, stopwords=False, n=None):
    if stopwords == True:
        vec = CountVectorizer(ngram_range=(2, 2), stop_words = 'english').fit(corpus)
    else:
        vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Generate a plotly histogram of bigrams with stopwords
common_words = get_top_n_bigram(essays1['essay'], n=20)
    
df3 = pd.DataFrame(common_words, columns = ['essay' , 'count'])
df3.groupby('essay').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in essays before removing stop words')

# Generate a plotly histogram of bigrams without stopwords
common_words = get_top_n_bigram(essays1['essay'], stopwords=True, n=20)

df4 = pd.DataFrame(common_words, columns = ['essay' , 'count'])
df4.groupby('essay').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in essays after removing stop words')

def get_top_n_trigram(corpus, stopwords=False, n=None):
    if stopwords == True:
        vec = CountVectorizer(ngram_range=(3, 3), stop_words = 'english').fit(corpus)
    else:
        vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Generate a plotly histogram of trigrams with stopwords
common_words = get_top_n_trigram(essays1['essay'], n=20)
    
df5 = pd.DataFrame(common_words, columns = ['essay' , 'count'])
df5.groupby('essay').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in essays before removing stop words')

# Generate a plotly histogram of trigrams without stopwords
common_words = get_top_n_trigram(essays1['essay'], stopwords=True, n=20)
    
df6 = pd.DataFrame(common_words, columns = ['essay' , 'count'])
df6.groupby('essay').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in essays after removing stop words')