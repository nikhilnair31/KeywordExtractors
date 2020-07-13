import os
import re
import nltk
import pandas as pd
import numpy as np
import keras as k
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from keras.models import load_model
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

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
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
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
    return text

def get_data():
    global final_features, final_labels
    final_features = []
    folder_path_features = 'C:/X/Dev/Python Projects/KeyWord-Extractor/Data/Inspec/docsutf8/'
    for filename in os.listdir(folder_path_features):
        with open(os.path.join(folder_path_features, filename), 'r') as f:
            final_features.append( f.read() )
    final_labels = []
    folder_path_labels  = 'C:/X/Dev/Python Projects/KeyWord-Extractor/Data/Inspec/keys/'
    for filenamekey in os.listdir(folder_path_labels):
        with open(os.path.join(folder_path_labels, filenamekey), 'r') as ff:
            final_labels.append( ff.read() )# clean_text(ff.read(), remove_stopwords=False) )
    print(f'final_features : {final_features[:3]} \n\n final_labels : {final_labels[:3]}\n\n')

get_data()
tokenizer = Tokenizer(num_words=25000, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(final_features)
word_index = tokenizer.word_index
train_sequences_features = tokenizer.texts_to_sequences(final_features)
print(f'\nMax of train_sequences_features : {(max(final_features, key=len))}\nAvg length of train_sequences_features : { sum(map(len, final_features))/float(len(final_features)) }')
train_sequences_features = pad_sequences(train_sequences_features, maxlen=300, padding='post', truncating='post')
print(f'Shape of data tensor : {train_sequences_features.shape}\nLength of train_sequences_features[0] : {len(train_sequences_features[0])}\n')
print(f'final_features[0] :\n{final_features[0]}\ntrain_sequences_features[0] :\n{train_sequences_features[0]}\n\n')
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_article(train_sequences_features[0]))
print('---')
print(final_features[0])

tokenizer.fit_on_texts(final_labels)
word_indexx = tokenizer.word_index
train_sequences_labels = tokenizer.texts_to_sequences(final_labels)
print(f'\nMax of train_sequences_labels : {(max(final_labels, key=len))}\nAvg length of train_sequences_labels : { sum(map(len, final_labels))/float(len(final_labels)) }')
train_sequences_labels = pad_sequences(train_sequences_labels, maxlen=300, padding='post', truncating='post')
print(f'Shape of data tensor : {train_sequences_labels.shape} \nLength of train_sequences_labels[0] : {len(train_sequences_labels[0])}\n')
print(f'final_labels[0] :\n{final_labels[0]}\train_sequences_labels[0] :\n{train_sequences_labels[0]}\n\n')
reverse_word_indexx = dict([(value, key) for (key, value) in word_indexx.items()])
def decode_article(text):
    return ' '.join([reverse_word_indexx.get(i, '?') for i in text])
print(decode_article(train_sequences_labels[0]))
print('---')
print(final_labels[0])

xFeature_train, xFeature_test, yLabel_train, yLabel_test = train_test_split(train_sequences_features, train_sequences_labels, test_size=0.1, random_state=0)
print(f'xFeature_train : {xFeature_train[:3]} \n\n xFeature_test : {xFeature_test[:3]}\n')
print(f'yLabel_train : {yLabel_train[:3]} \n\n yLabel_test : {yLabel_test[:3]}\n\n')
print(f'xFeature_train : {xFeature_train.shape} \t yLabel_train : {yLabel_train.shape}\n')
print(f'xFeature_test : {xFeature_test.shape} \t yLabel_test : {yLabel_test.shape}\n\n')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(25000, 300, input_length=xFeature_train.shape[1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.Dense(300, activation='relu')
])
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xFeature_train, yLabel_train, epochs=4, batch_size=64,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

model.save('Model Version/lstm.h5')
# model = tf.keras.models.load_model('Model Version/lstm.h5')
test_pred = model.predict(xFeature_test, verbose=1)
print(decode_article(xFeature_test[0]))
print(xFeature_test[0])
print(test_pred[0])
print(test_pred[0].shape)

#Uses bi-LSTM