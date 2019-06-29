## Topic Modeling with Latent Dirichlet Allocation (_LDA_)

*Topic modeling* is family of techniques that can be used to describe and summarize the documents in a corpus according to a set of latent "topics". For this demo, we'll be using [*Latent Dirichlet Allocation*](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) or LDA, a popular approach to topic modeling.

In many conventional NLP applications, documents are represented a mixture of the individual tokens (words and phrases) they contain. In other words, a document is represented as a *vector* of token counts. There are two layers in this model &mdash; documents and tokens &mdash; and the size or dimensionality of the document vectors is the number of tokens in the corpus vocabulary. This approach has a number of disadvantages:
* Document vectors tend to be large (one dimension for each token $\Rightarrow$ lots of dimensions)
* They also tend to be very sparse. Any given document only contains a small fraction of all tokens in the vocabulary, so most values in the document's token vector are 0.
* The dimensions are fully indepedent from each other &mdash; there's no sense of connection between related tokens, such as _knife_ and _fork_.

LDA injects a third layer into this conceptual model. Documents are represented as a mixture of a pre-defined number of *topics*, and the *topics* are represented as a mixture of the individual tokens in the vocabulary. The number of topics is a model hyperparameter selected by the practitioner. LDA makes a prior assumption that the (document, topic) and (topic, token) mixtures follow [*Dirichlet*](https://en.wikipedia.org/wiki/Dirichlet_distribution) probability distributions. This assumption encourages documents to consist mostly of a handful of topics, and topics to consist mostly of a modest set of the tokens.

### Describing text with LDA
Beyond data exploration, one of the key uses for an LDA model is providing a compact, quantitative description of
natural language text. Once an LDA model has been trained, it can be used to represent free text as a mixture of the
topics the model learned from the original corpus. This mixture can be interpreted as a probability distribution across
the topics, so the LDA representation of a paragraph of text might look like 50% _Topic A_, 20% _Topic B_, 20% _Topic
C_, and 10% _Topic D_.

To use an LDA model to generate a vector representation of new text, you'll need to apply any text preprocessing steps
you used on the model's training corpus to the new text, too. For our model, the preprocessing steps we used include:
1. Using spaCy to remove punctuation and lemmatize the text
1. Applying our first-order phrase model to join word pairs
1. Applying our second-order phrase model to join longer phrases
1. Removing stopwords
1. Creating a bag-of-words representation

Once you've applied these preprocessing steps to the new text, it's ready to pass directly to the model to create an LDA
representation. The `lda_description(...)` function will perform all these steps for us, including printing the
resulting topical description of the input text.