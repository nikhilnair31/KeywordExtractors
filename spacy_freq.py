import spacy
from collections import Counter
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)            
    return result

def lemmatizer(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]

    for word in words:
        final.append(wordnet_lemmatizer.lemmatize(word))
    return ' '.join(final)

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    final = []
    text = open("Data/testing_text.txt","r", encoding="utf8").read()
    
    keywords = [ x[0] for x in Counter(get_hotwords(lemmatizer(text))).most_common(10) ]
    print(keywords)