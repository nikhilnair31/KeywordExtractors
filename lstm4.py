import numpy as np
import keras as k
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from math import nan
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_contrib.utils import save_load_utils
from keras.models import Model, Input, load_model, model_from_json 
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn_crfsuite.metrics import flat_classification_report  
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


class SentenceGetter(object):
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None  

def glove_embedding_matrix():
    embeddings_index = dict()
    f = open(f'Data/gloVe/glove.6B.100d.txt', encoding='utf8')
    for line in f:
    	values = line.split()
    	coefs = asarray(values[1:], dtype='float32')
    	embeddings_index[values[0]] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    embedding_matrix = zeros((n_words, 100))
    for word, i in word2idx.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out

def get_data():
    global df, data, sentences
    df = pd.read_csv("Data/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
    data = df.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos', 'next-next-shape', 'next-next-word', 'next-pos', 
        'next-shape', 'next-word', 'prev-iob', 'prev-lemma', 'prev-pos', 'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 
        'prev-prev-shape', 'prev-prev-word', 'prev-shape', 'shape', 'prev-word',"pos"],axis=1)
    data = df[['sentence_idx','word','tag']]
    data.replace(["B-geo", "B-tim", "B-org", "I-per", "B-per", "I-org", "B-gpe", "B-gpe", "I-geo", "I-tim", "B-art", "B-eve", "I-eve", "I-art", "I-gpe", "B-nat", "I-nat"], "YES_KW", inplace=True) 
    data.replace(["O", "unk"], "not_kw", inplace=True) 
    print('\n data.head :\n', data.head())
    print('\n tag count :\n', data['tag'].value_counts())

    getter = SentenceGetter(data)
    sentences = getter.sentences
    print(sentences[:2])

def text_processing():
    global maxlen, words, n_words, tags, n_tags
    maxlen = max([len(s) for s in sentences])
    print(f'\n maxlen : {maxlen}')

    words = list(set(data["word"].values))
    words.append("ENDPAD")
    n_words = len(words)
    print(f'\n words : {words[:25]}')
    print(f'\n n_words : {n_words}')

    tags = []
    for tag in set(data["tag"].values):
        if tag is nan or isinstance(tag, float):
            tags.append('unk')
        else:
            tags.append(tag)
    n_tags = len(tags)
    print(f'\n tags : {tags}')
    print(f'\n n_tags : {n_tags}')

def dict_stuff():
    global word2idx, tag2idx, idx2tag, X, y, X_train, X_test, y_train, y_test 
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {v: k for k, v in dict.items(tag2idx)}
    print(f'\n word2idx : {dict(itertools.islice(word2idx.items(), 25))}')
    print(f'\n tag2idx : {tag2idx}')
    print(f'\n idx2tag : {idx2tag}')

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=maxlen, sequences=X, padding="post",value=n_words - 1)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["not_kw"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f'\n X_test : {X_test[:1]}')
    print(f'\n y_test : {y_test[:1]}')

def model_stuff():
    global model
    word_embedding_size = 100
    input = Input(shape=(maxlen,))
    model = Embedding(input_dim=n_words, output_dim=word_embedding_size, weights=[glove_embedding_matrix()], input_length=maxlen, trainable=False)(input)
    model = Bidirectional(LSTM(units=word_embedding_size,return_sequences=True,dropout=0.5,recurrent_dropout=0.5,kernel_initializer=k.initializers.he_normal()))(model)
    model = Bidirectional(LSTM(units=word_embedding_size*2,return_sequences=True,dropout=0.5,recurrent_dropout=0.5,kernel_initializer=k.initializers.he_normal()))(model)
    model = TimeDistributed(Dense(n_tags, activation="relu"))(model)  
    crf = CRF(n_tags)
    out = crf(model)
    model = Model(input, out)
    # adam = k.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
    model.summary()

def train_model():
    global checkpoint, history
    filepath="Model Version/ner_{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, np.array(y_train), batch_size=256, epochs=5, validation_split=0.2, verbose=1, callbacks=callbacks_list)
    model.save("Model Version/ner_saw.hdf5")
    # model_json = model.to_json()
    # with open("Model Version/ner_saw.json", "w") as json_file:
    #     json_file.write(model_json)
    # save_load_utils.save_all_weights(model,"Model Version/ner_saw.hdf5")
    plot_history(history)

def load_model():
    # json_file = open('Model Version/ner_saw.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json, custom_objects={'CRF': CRF})
    # model = save_load_utils.load_all_weights(loaded_model, "Model Version/ner_saw.hdf5")
    import tensorflow as tf
    crf = CRF(n_tags)
    model = k.models.load_model("Model Version/ner_saw.hdf5", custom_objects={'CRF': crf, 'crf_loss': crf.loss_function, 'crf_viterbi_accuracy': crf.accuracy})
    print("Loaded model from disk")
    model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])

def plot_history(history):
    plt.style.use('ggplot')
    crf_viterbi_accuracy: = history.history['crf_viterbi_accuracy:']
    val_crf_viterbi_accuracy: = history.history['val_crf_viterbi_accuracy:']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(crf_viterbi_accuracy:) + 1)
    plt.plot(x, crf_viterbi_accuracy:, 'b', label='Training crf_viterbi_accuracy')
    plt.plot(x, val_crf_viterbi_accuracy:, 'r', label='Validation val_crf_viterbi_accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

#PREDICTING VALUES
def single_pred():
    i = 1
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    gt = np.argmax(y_test[i], axis=-1)
    print(gt)
    print("{:14}: ({:5}): {}".format("Word", "True", "Pred"))
    for idx, (w,pred) in enumerate(zip(X_test[i],p[0])):
        print("{:14}: ({:5}): {}".format(words[w],idx2tag[gt[idx]],tags[pred]))

def full_pred():
    TP = {}
    TN = {}
    FP = {}
    FN = {}
    for tag in tag2idx.keys():
        TP[tag] = 0
        TN[tag] = 0    
        FP[tag] = 0    
        FN[tag] = 0

    test_pred = model.predict(np.array(X_test), verbose=1)   
    pred_labels = pred2label(test_pred)
    test_labels = pred2label(y_test)
    print(f'\n pred_labels : {pred_labels[:2]}')
    print(f'\n test_labels : {test_labels[:2]}')

    def accumulate_score_by_tag(gt, pred):
        if gt == pred:
            TP[gt] += 1
        elif gt != 'not_kw' and pred == 'not_kw':
            FN[gt] +=1
        elif gt == 'not_kw' and pred != 'not_kw':
            FP[gt] += 1
        else:
            TN[gt] += 1 
    
    for i, sentence in enumerate(X_test):
        y_hat = np.argmax(test_pred[0], axis=-1)
        gt = np.argmax(y_test[0], axis=-1)
        for idx, (w,pred) in enumerate(zip(sentence,y_hat)):
            accumulate_score_by_tag(idx2tag[gt[idx]],tags[pred])
    for tag in tag2idx.keys():
        print(f'tag:{tag}')    
        print('\t TN:{:10}\tFP:{:10}'.format(TN[tag],FP[tag]))
        print('\t FN:{:10}\tTP:{:10}'.format(FN[tag],TP[tag]))  

#SCORING MODEL
def scoring():
    m = MultiLabelBinarizer().fit(y_test)
    print("F1-score: {:.1%}".format( f1_score(m.transform(y_test), m.transform(test_pred), average='macro') ))
    print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    report = flat_classification_report(y_pred=pred_labels, y_true=test_labels)
    print(report)


get_data()
text_processing()
dict_stuff()
model_stuff()
train_model()
load_model()
single_pred()
full_pred()
#scoring()