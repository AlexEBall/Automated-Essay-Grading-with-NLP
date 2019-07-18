import os
import pyLDAvis
import pyLDAvis.gensim
import pickle

# Run in Jupyter to see the visualization
intermediate_directory = os.path.join('../data/intermediate')
LDAvis_data_filepath = os.path.join(intermediate_directory, 'ldavis_prepared')

with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.display(LDAvis_prepared)