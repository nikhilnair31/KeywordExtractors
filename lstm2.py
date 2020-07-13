import os
import re
import math 
import itertools
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras import regularizers
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from keras.models import Sequential
# from keras_contrib.layers import CRF
from keras.models import Model, Input
from keras.layers import Dense, Flatten, Embedding, Bidirectional, LSTM, Dropout, TimeDistributed
from keras.models import load_model
from nltk.stem import WordNetLemmatizer 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
contractions = { "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have",
"couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
"hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",  "he'll": "he will",
"he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
"isn't": "is not", "it'd": "it would", "it'll": "it will","it's": "it is", "let's": "let us","ma'am": "madam", "mayn't": "may not",
"might've": "might have", "mightn't": "might not","must've": "must have", "mustn't": "must not","needn't": "need not", "oughtn't": "ought not",
"shan't": "shall not", "sha'n't": "shall not","she'd": "she would", "she'll": "she will","she's": "she is","should've": "should have",
"shouldn't": "should not","that'd": "that would","that's": "that is","there'd": "there had","there's": "there is","they'd": "they would",
"they'll": "they will","they're": "they are","they've": "they have","wasn't": "was not","we'd": "we would","we'll": "we will",
"we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what're": "what are","what's": "what is","what've": "what have",
"where'd": "where did","where's": "where is","who'll": "who will","who's": "who is","won't": "will not","wouldn't": "would not","you'd": "you would",
"you'll": "you will","you're": "you are"}

def clean_text(text, remove_stopwords = True):
    word_lemmatizer = WordNetLemmatizer()
    text = text.split()
    new_text = []
    for word in text:
        if word.isalpha() in contractions:
            new_text.append(word_lemmatizer.lemmatize(contractions[word]))
        else:
            new_text.append(word_lemmatizer.lemmatize(word))
    text = " ".join(new_text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return ' '.join( [w for w in text.split() if len(w)>1] )

def get_data():
    global text_features, final_labels, final_label_updated
    text_features = []
    folder_path_features = 'Data/Inspec/docsutf8/'
    for filename in os.listdir(folder_path_features):
        with open(os.path.join(folder_path_features, filename), 'r') as f:
            text_features.append( clean_text(f.read().lower(), remove_stopwords=True) )
    final_labels = []
    folder_path_labels  = 'Data/Inspec/keys/'
    for filenamekey in os.listdir(folder_path_labels):
        with open(os.path.join(folder_path_labels, filenamekey), 'r') as ff:
            final_labels.append( clean_text(ff.read().lower(), remove_stopwords=False) )
    
    final_label_updated = []
    for i in range(len(text_features)):
        sent = []
        str_to_replace = text_features[i]
        str_to_key = final_labels[i]
        split = str_to_replace.split()
        for word in split:
            if word in str_to_key:
                sent.append(1)
                #str_to_replace = re.sub(r'\b'+word+r'\b', 'yakp', str_to_replace)
            else:
                sent.append(0)
                #str_to_replace = re.sub(r'\b'+word+r'\b', 'nokp', str_to_replace)
        final_label_updated.append(sent+[0]*(input_max_len-len(sent)))
        #final_label_updated.append(idk.tolist())
    # print(f'[DEBUG] text_features : {text_features[:2]} \n\nfinal_labels : {final_labels[:2]}\n\nfinal_label_updated : {final_label_updated[:2]}\n')
    # print(f'[DEBUG] Max length of text_features : {len(max(text_features, key=len))}\tAvg length of text_features : { sum(map(len, text_features))/float(len(text_features)) }')
    # print(f'[DEBUG] Max length of final_labels : {len(max(final_labels, key=len))}\tAvg length of final_labels : { sum(map(len, final_labels))/float(len(final_labels)) }')
    # print(f'[DEBUG] Max length of final_label_updated : {len(max(final_label_updated, key=len))}\tAvg length of final_label_updated : { sum(map(len, final_label_updated))/float(len(final_label_updated)) }')

def text_processing():
    global t, word_index, vocab_size, encoded_features, padded_features, encoded_labels, padded_labels
    t = Tokenizer()
    
    t.fit_on_texts(text_features)
    encoded_features = t.texts_to_sequences(text_features)
    padded_features = pad_sequences(encoded_features, maxlen=input_max_len, padding='post')
    print(f'\n[DEBUG] encoded_features :\n{encoded_features[:2]}\nencoded_features.len : {len(encoded_features)}\nencoded_features[0].len : {len(encoded_features[0])}')
    print(f'\n[DEBUG] padded_features :\n{padded_features[:2]}\npadded_features.len : {len(padded_features)}\npadded_features[0].len : {len(padded_features[0])}')
    print(f'{padded_features.shape} | {padded_features[0].shape} | {padded_features[0:1].shape}')

    word_index = t.word_index
    print(f'\n[DEBUG] word_index : {dict(itertools.islice(word_index.items(), 10))}')
    vocab_size = len(t.word_index) + 1
    print(f'[DEBUG] vocab_size : {vocab_size}\n')

def glove_embedding_matrix():
    embeddings_index = dict()
    f = open(f'Data/gloVe/glove.6B.{embedding_size}d.txt', encoding='utf8')
    for line in f:
    	values = line.split()
    	coefs = asarray(values[1:], dtype='float32')
    	embeddings_index[values[0]] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, embedding_size))
    for word, i in t.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_test():
    print("[DEBUG] Model loaded sucessfully")
    model = load_model(model_name)
    test_pred = model.predict(padded_features[0:2], verbose=logging)
    for i in range(len(test_pred)):
        text = test_pred[i]
        print(f'\n[DEBUG] text_features :\n{text_features[i]}')
        print(f'[DEBUG] final_labels :\n{final_labels[i]}')
        print(f'[DEBUG] text :\n{text[:len(encoded_features[i])]}')
        print(f'[DEBUG] floor_pred :\n{[int(round(num)) for num in text][:len(encoded_features[i])]}')
        print(f'[DEBUG] floor_actual :\n{final_label_updated[i][:len(encoded_features[i])]}')
        kw_tuple = list(zip(list( text_features[i].split()), text[:len(encoded_features[i])] ))
        print(f'[DEBUG] sorted kw_tuple :\n{sorted(kw_tuple, key=lambda tup: tup[1], reverse=True)}\n')

def train():
    input = Input(shape=(input_max_len,))
    model = Embedding(vocab_size, embedding_size, weights=[glove_embedding_matrix()], input_length=input_max_len, trainable=False)(input)
    model = Bidirectional(LSTM(embedding_size,dropout=dropout,recurrent_dropout=recurrent_dropout, return_sequences=True))(model)
    model = Bidirectional(LSTM(2*embedding_size,dropout=dropout,recurrent_dropout=recurrent_dropout, return_sequences=True))(model)
    model = TimeDistributed(Dense(embedding_size,activation='sigmoid'))(model)
    model = Flatten()(model)
    model = Dense(input_max_len, activation='sigmoid')(model)
    out = model
    model = Model(input, out)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(padded_features, np.array(final_label_updated), validation_split = validation_split, epochs=epochs, batch_size=batch_size, verbose=logging, shuffle=True)
    model.save(model_name)
    metrics(history, model)

def metrics(history, model):
    loss, accuracy = model.evaluate(padded_features, np.array(final_label_updated), verbose=logging)
    print(f'[DEBUG] Accuracy : {accuracy*100}')
    print(f'[DEBUG] Loss : {loss*100}')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

global input_max_len, output_max_len, embedding_size, dropout, recurrent_dropout, epochs, batch_size, validation_split, model_name
logging = 1
epochs = 2
batch_size = 64
validation_split = 0.2
input_max_len = 650
output_max_len = 300
embedding_size = 100
dropout = 0.4
recurrent_dropout = 0.3
model_name = 'Model Version/lstm2_tdf3adsig.h5'
get_data()
text_processing()
train()
load_test()
#Uses bi-LSTM with gloVe