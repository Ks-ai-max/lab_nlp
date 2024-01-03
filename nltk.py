# Tokenization
from nltk import word_tokenize, sent_tokenize
sent = str(input())
print(word_tokenize(sent))
print(sent_tokenize(sent))


# Removing stop-words
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
sent = str(input())
from nltk import word_tokenize
token = word_tokenize(sent)
cleaned_token = []
for word in token:
    if word not in stop_words:
        cleaned_token.append(word)
print(token)
print(cleaned_token)


# Stemming
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
words = ['processing', 'processor', 'procession', 'processed', 'inprocess']
stemmed = [stemmer.stem(word) for word in words]
print(stemmed)


# Lemmatization 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
word = "running"
pos_tag = wordnet.VERB
lemma = lemmatizer.lemmatize(word, pos=pos_tag)
print(lemma)


# POS tagging 
from nltk import pos_tag
from nltk import word_tokenize
sent = "NLTK is a powerful library for working with human language data."
tokens = word_tokenize(sent) 
pos_tags = pos_tag(tokens)
print(pos_tags)

