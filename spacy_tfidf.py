import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def textProcessing(doc):
    Nouns = []
    Noun_set = []
    trimmed_noun_set = []
    removing_duplicates = []
    arr = []
    vocab = []
    vocab_dict = {}

    doc = nlp(doc.lower())

    for possible_nouns in doc:
        if possible_nouns.pos_ in ["NOUN","PROPN"] :
            #print("1 : ", [possible_nouns , [child for child in possible_nouns.children]])
            Nouns.append([possible_nouns , [child for child in possible_nouns.children]])
    for i,j in Nouns:
        for k in j:
            #print("2 : ", [k,i])
            Noun_set.append([k,i])
    
    for i , j in Noun_set:
        if i.pos_ in ['PROPN','NOUN','ADJ']:
            #print("3 : ", [i ,j])
            trimmed_noun_set.append([i ,j])           
    
    for word in trimmed_noun_set:
        if word not in removing_duplicates:
            #print("word : ", word)
            removing_duplicates.append(word)   
    for i in removing_duplicates:
        strs = ''
        for j in i:
            strs += str(j)+" "
        arr.append(strs.strip())
    
    for word in Noun_set:
        string = ''
        for j in word:
            string+= str(j)+ " "
        vocab.append(string.strip())
    
    for word in vocab:
        vocab_dict[word] = 0
        #print("word : ", word, " | vocab_dict : ", vocab_dict[word])
        
    for word in arr:
        vocab_dict[word] += 1
        #print("word : ", word, " | vocab_dict : ", vocab_dict[word])

    return vocab_dict , arr

def computeTF(wordDict,bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

def computeIDF(doclist):
    import math 
    count = 0
    idfDict = {}
    for element in doclist:
        for j in element:
            count+=1
    N = count

    idfDict = dict.fromkeys(doclist[0].keys(),0)

    for doc in doclist:
        for word,val in doc.items():
            if val>0:
                idfDict[word]+= 1

    for word,val in idfDict.items():
        if val == 0:
            idfDict[word] = 0.0
        else:
            idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTfidf(tf,idf):
    tfidf = {}
    sorted_list = []
    for word , val in tf.items():
        tfidf[word] = val * idf[word]
    ranking_list  = sorted(tfidf.items(),reverse=True, key = lambda kv:(kv[1], kv[0]))[:10]
    for i, _ in ranking_list:
        sorted_list.append(i)
    return sorted_list

def lemmatizer(text):
    final = []
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    for word in words:
        final.append(wordnet_lemmatizer.lemmatize(word))
    return ' '.join(final)

nlp = spacy.load("en_core_web_sm")
text = open("Data/testing_text.txt","r", encoding="utf8").read()
vocab_dict , arr = textProcessing(lemmatizer(text))
# print(f'vocab_dict : { vocab_dict } \n\n arr : { arr } \n')
# print(f'sorted : { sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)[:10] }\n')
tf = computeTF(vocab_dict,arr)
idf = computeIDF([vocab_dict])
tfidf = computeTfidf(tf,idf)
print(f'tfidf : { tfidf } \n')