import os
import pyLDAvis
import pyLDAvis.gensim

# Run in Jupyter to see the visualization
intermediate_directory = os.path.join('../../data/intermediate')
LDAvis_data_filepath = os.path.join(intermediate_directory, 'ldavis_prepared')

pyLDAvis.display(LDAvis_prepared)
