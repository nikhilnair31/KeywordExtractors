import spacy
import numpy as np
from scipy.sparse.csr import csr_matrix
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def get_text():
    text_file = open("Data/testing_text.txt","r+", encoding="utf8")
    copy_text_file = open("Data/copy_text.txt","r+", encoding="utf8")
    copy_text_file.truncate(0)
    copy_text_file.write( text_preprocessing(text_file.read()) )
    text_file.close()
    copy_text_file.close()

def text_preprocessing(text):
    final = []
    word_lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha()]
    for word in words:
        if nlp.vocab[word_lemmatizer.lemmatize(word)].is_stop == False:
            final.append(word)
    return ' '.join(final)

if(__name__ == '__main__'):
    global nlp 
    nlp = spacy.load("en_core_web_sm")
    get_text()

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english', sublinear_tf=True)
    tfidf_matrix =  tf.fit_transform( open("Data/copy_text.txt","r+", encoding="utf8") )
    feature_names = tf.get_feature_names()
    feature_index = tfidf_matrix[0,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[0, x] for x in feature_index])
    idk_dict = {}
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        idk_dict[w] = s
    print(f'sorted : { sorted(idk_dict.items(), key=lambda x: x[1], reverse=True)[:10] }\n')