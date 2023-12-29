import wikipedia
import nltk

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

black_hole = wikipedia.page("Black hole")
artificial_intelligence = wikipedia.page("Artificial Intelligence")
analitical_psychology = wikipedia.page("Analytical psychology")
digital_humanities = wikipedia.page("Digital humanities")

corpus = [black_hole.content, artificial_intelligence.content, analitical_psychology.content, digital_humanities.content]


import re
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Убираем все специальные символы и ставим на их место пробел
        document = re.sub(r'\W', ' ', str(document))

        # Убираем все одиночные символы
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Убираем все одиночные символы в начале текста
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Заменяем несколько идущих подряд пробелов одним
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Убираем 'b', который появляется при очистке документа
        document = re.sub(r'^b\s+', '', document)

        # Преобразуем в нижний регистр
        document = document.lower()

        # Токенизация + лемматизация + удаление стоп-слов и слов, короче 5 символов
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word)  > 5]

        return tokens


processed_data = [];
for doc in corpus:
    tokens = preprocess_text(doc)
    processed_data.append(tokens)


from gensim import corpora

gensim_dictionary = corpora.Dictionary(processed_data)
gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in processed_data]


import pickle

pickle.dump(gensim_corpus, open('gensim_corpus_corpus.pkl', 'wb'))
gensim_dictionary.save('gensim_dictionary.gensim')


import gensim

lda_model = gensim.models.ldamodel.LdaModel(gensim_corpus, num_topics=4, id2word=gensim_dictionary, passes=20)
lda_model.save('gensim_model.gensim')


topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

print('\nPerplexity:', lda_model.log_perplexity(gensim_corpus))


