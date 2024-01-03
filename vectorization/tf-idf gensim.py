import gensim
from gensim import corpora
from pprint import pprint

text = ['I like studying computational linguistics',
        'Natural language processing is an interesting subfield of linguistics',
        'Do you like linguistics?']

tokens = [[token for token in sentence.split()] for sentence in text]

gensim_dictionary = corpora.Dictionary()
gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in tokens]

from gensim import models
import numpy as np

tfidf = models.TfidfModel(gensim_corpus, smartirs='ntc')

for sent in tfidf[gensim_corpus]:
    pprint([[gensim_dictionary[id], np.around(frequency, decimals=2)] for id, frequency in sent])
    
