import nltk
from rake_nltk import Metric, Rake
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

wordnet_lemmatizer = WordNetLemmatizer()
final = []
text = open("Data/testing_text.txt","r", encoding="utf8").read()

tokens = word_tokenize(text)
words = [word for word in tokens if word.isalpha()]

for word in words:
    final.append(wordnet_lemmatizer.lemmatize(word))
finaltext = ' '.join(final)
# The useLocation hook returns the location object that represents the current URL. You can think about it like a useState that returns a new location whenever the URL changes.This could be really useful e.g. in a situation where you would like to trigger a new “page view” event using your web analytics tool whenever a new page loads, as in the following example

r = Rake(min_length=1, max_length=3)
r.extract_keywords_from_text(finaltext)
print(r.get_ranked_phrases()[:10])