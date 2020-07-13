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
from keras.models import Model, Input
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.layers import Dense, Flatten, Embedding, Bidirectional, LSTM, Dropout, TimeDistributed
from nltk.stem import WordNetLemmatizer 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Layer
import keras.backend as K

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

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
            else:
                sent.append(0)
        final_label_updated.append(sent+[0]*(input_max_len-len(sent)))

    # plt.hist([len(sen) for sen in text_features], bins=10)
    # plt.show()
    # print(f'\n\ntext_features : {text_features[:2]} \nfinal_labels : {final_labels[:2]}\nfinal_label_updated : {final_label_updated[:2]}\n\n')
    # print(f'Max length of text_features : {len(max(text_features, key=len))}\nAvg length of text_features : { sum(map(len, text_features))/float(len(text_features)) }')
    # print(f'Max length of final_labels : {len(max(final_labels, key=len))}\nAvg length of final_labels : { sum(map(len, final_labels))/float(len(final_labels)) }')
    # print(f'Max length of final_label_updated : {len(max(final_label_updated, key=len))}\nAvg length of final_label_updated : { sum(map(len, final_label_updated))/float(len(final_label_updated)) }')

def text_processing():
    global t, word_index, vocab_size, encoded_features, padded_features, encoded_labels, padded_labels
    t = Tokenizer()
    t.fit_on_texts(text_features)
    encoded_features = t.texts_to_sequences(text_features)
    padded_features = pad_sequences(encoded_features, maxlen=input_max_len, padding='post')
    print(f'\npadded_features :\n{padded_features[:2]}\npadded_features.len : {len(padded_features)}\npadded_features[0].len : {len(padded_features[0])}')
    print(f'{padded_features[0].shape} | {padded_features[0:1].shape}') 

    word_index = t.word_index
    print(f'\nword_index : {dict(itertools.islice(word_index.items(), 10))}')
    vocab_size = len(t.word_index) + 1
    print(f'vocab_size : {vocab_size}\n')

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
    print("[INFO] Model loaded sucessfully\n")
    model = load_model(model_name,  custom_objects={'attention':attention})
    vec_to_pred = padded_features[0:1]
    predicted_val = model.predict(vec_to_pred, verbose=logging)

    for i in range(len(predicted_val)):
        text = predicted_val[i]
        print(f'\nog_text :\n{text_features[i]}')
        print(f'og_keywords :\n{final_labels[i]}')
        print(f'og_text embedding :\n{text[:len(encoded_features[i])]}')
        print(f'pred_val :\n{[int(round(num)) for num in text][:len(encoded_features[i])]}')
        print(f'actual_val :\n{final_label_updated[i][:len(encoded_features[i])]}')
        keyword_tuple = list(zip(list( text_features[i].split()), text[:len(encoded_features[i])] ))
        print(f'sorted keyword_tuple :\n{sorted(keyword_tuple, key=lambda tup: tup[1], reverse=True)}')

    y_act = list(itertools.chain.from_iterable(final_label_updated[0:1]))
    y_pred = list(itertools.chain.from_iterable(np.round_(predicted_val).astype(np.int64)))
    # print(f'\ny_act :\n{y_act}\ny_pred :\n{y_pred}')
    print(f'\nconfusion_matrix :\n{ confusion_matrix(y_act, y_pred) }')
    # print(f'\nclassification_report :\n{classification_report( y_act, y_pred, target_names=[0,1])}')

def train():
    input = Input(shape=(input_max_len,))
    model = Embedding(vocab_size, embedding_size, weights=[glove_embedding_matrix()], input_length=input_max_len, trainable=False)(input)
    model = Bidirectional(LSTM(embedding_size,dropout=dropout,recurrent_dropout=recurrent_dropout, return_sequences=True))(model)
    model = Bidirectional(LSTM(2*embedding_size,dropout=dropout,recurrent_dropout=recurrent_dropout, return_sequences=True))(model)
    model = attention()(model)
    model = Dense(input_max_len, activation='sigmoid')(model)
    # model = TimeDistributed(Dense(embedding_size,activation='sigmoid'))(model) #Needs attention to return_sequences=True
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
epochs = 3
batch_size = 32
validation_split = 0.05
input_max_len = 1000
output_max_len = 300
embedding_size = 100
dropout=0.4
recurrent_dropout=0.3
model_name = 'Model Version/lstm3_k3adsig.h5'
get_data()
text_processing()
train()
load_test()
#Uses attention with bi-LSTM and gloVe