import keras as k
import pandas as pd
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


from google.colab import drive
drive.mount('/content/drive')

pip install git+https://www.github.com/keras-team/keras-contrib.git

data = pd.read_csv("drive/My Drive/ner_dataset.csv", encoding="latin1")

data = data.fillna(method="ffill")

data.tail(10)

words = list(set(data["Word"].values))
n_words = len(words); n_words

pos = list(set(data["POS"].values))
n_pos = len(pos); n_pos

pos

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)


sentences = getter.sentences


max_len = 75

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
idx2word = {i: w for w, i in word2idx.items()}
pos2idx = {t: i + 1 for i, t in enumerate(pos)}
pos2idx["PAD"] = 0
idx2pos = {i: w for w, i in pos2idx.items()}

from keras.preprocessing.sequence import pad_sequences
X_word = [[word2idx[w[0]] for w in s] for s in sentences]


X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')


y = [[pos2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, value=pos2idx["PAD"], padding='post', truncating='post')


from keras.utils import to_categorical

y = [to_categorical(i, num_classes=n_pos+1) for i in y]

print(y[0][0])

sentences[0]

pos2idx["NNS"]

from sklearn.model_selection import train_test_split
X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)

input = Input(shape=(max_len,))
word_embedding_size = 300
model = Embedding(input_dim=n_words+2, output_dim=word_embedding_size, input_length=max_len)(input)  # 300-dim embedding
model = Bidirectional(LSTM(units=word_embedding_size, 
                           return_sequences=True, 
                           dropout=0.2, 
                           recurrent_dropout=0.2, 
                           kernel_initializer=k.initializers.he_normal()))(model)
                             # variational biLSTM
model = Bidirectional(LSTM(units=2*word_embedding_size, 
                           return_sequences=True, 
                           dropout=0.2, 
                           recurrent_dropout=0.2, 
                           kernel_initializer=k.initializers.he_normal()))(model)
model = TimeDistributed(Dense(n_pos+1, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_pos+1)  # CRF layer
out = crf(model)  # output

adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

model = Model(input, out)

model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])

model.summary()

filepath="pos-bi-lstm-td-model-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_word_tr,
                    np.array(y_tr),
                    batch_size=128, epochs=10, validation_split=0.1, verbose=1,callbacks=callbacks_list)

from keras.models import load_model
model=load_model("pos-bi-lstm-td-model-0.99.h5", custom_objects={'CRF':CRF, 'crf_loss':crf.loss_function, 'crf_viterbi_accuracy':crf.accuracy })

!pip install seqeval

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

test_pred = model.predict(X_word_te, verbose=1)

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2pos[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))

test_pred.shape

i = 1000
p = np.argmax(test_pred[i], axis=-1)
true=np.argmax(y_te[i],axis=-1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_word_te[i], true, p):
    if w != 0:
        print("{:15}: {:5} {}".format(idx2word[w], idx2pos[t], idx2pos[pred]))

def text_splitting(a):
  a=a.split(".")
  #print(a)
  a.remove(a[len(a)-1])
  l=[]
  for i in range(len(a)):
    l.append(a[i])
    l.append(".")
  #print(l)
  g=[]
  for i in range(len(l)):
    if(l[i]=="."):
      g.append(".")
    
    else:
      c=[]
      l[i]=l[i].split(",")
      #print(l[i])
      if(len(l[i])==1):
        g=g+l[i][0].split(" ")
        #print("vishal")
      else:
        for j in range(len(l[i])-1):
          c.append(l[i][j])
          c.append(",")
          if(j==len(l[i])-2):
            c.append(l[i][j+1])
        #print(c)
        for k in range(len(c)):
          if c[k]==",":
            g.append(",")
          else:
            g=g+c[k].split(" ")
  for i in g:
    if i=='':
      g.remove(i)
  return g   



def pos_split(a):
  xword=[]
  for i in a:
    if i in words:
      xword.append(word2idx[i])
    else:
      xword.append(word2idx["UNK"])
  x_word=pad_sequences(maxlen=max_len, sequences=[xword], value=word2idx["PAD"], padding='post', truncating='post')
  y_pred_test_pos = model.predict(x_word)
  q_pos=np.argmax(y_pred_test_pos, axis=-1)
  list_of_pos=[]
  for i,j in enumerate(x_word[0]):
    if(j!=word2idx["PAD"]):
      list_of_pos.append(idx2pos[q_pos[0][i]])
  return list_of_pos

def keyword_extraction(list_of_pos):
  only_nn=[]
  for i in range(len(list_of_pos)):
    if("NN" in list_of_pos[i] or "JJ" in list_of_pos[i] ):
      only_nn.append(i);

  only_nn_final=[]
  i=0
  while(i<len(only_nn)):
    g=[]
    while(only_nn[i]+1 in only_nn and i<len(only_nn)):
      g.append(only_nn[i])
      g.append(only_nn[i]+1)
      i=i+1
    if(only_nn[i] not in g):
      only_nn_final.append([only_nn[i]])
    i=i+1
    if(len(g)!=0):
      only_nn_final.append(sorted(list(set(g))))


  for i in only_nn_final:
      if(len(i)==1 and list_of_pos[i[0]]=="NNS"):
        i.remove(i[0])
      if(len(i)==1 and list_of_pos[i[0]]=="JJ"):
        i.remove(i[0])
      if(len(i)==1 and list_of_pos[i[0]]=="JJR"):
        i.remove(i[0])
      if(len(i)==1 and list_of_pos[i[0]]=="JJS"):
        i.remove(i[0])
  
  keywords=[]
  for i in only_nn_final:
    keyword=""
    for j in i:
      keyword=keyword+a[j]+" "
    keywords.append(keyword)
  for i in range(len(keywords)):
    if(keywords[i]!=""):
      print(keywords[i])






#a=["Vishal","Vikram","Ray","was","born","in","1998","in","Delhi","in","Safdarjung","hospital",".","He","is","doing","a","task","of","keyword","extraction","in","Learnogether",".","He","is","fun","loving","."]
#a=["Albert","Einstein","was","a","great","scientist","."]
#a="Automated keyword extraction allows you to analyze as much data as you want. Yes you could read texts and identify key terms manually, but it would be extremely time-consuming."
# a="Python is an interpreted, high level, general purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace."
# a="Keyword extraction acts based on rules and predefined parameters. You don’t have to deal with inconsistencies, which are common when performing any text analysis manually."
# a="You can perform keyword extraction on social media posts, customer reviews, surveys, or customer support tickets in real-time, and get insights about what’s being said about your product as they happen."
a = "The useLocation hook returns the location object that represents the current URL. You can think about it like a useState that returns a new location whenever the URL changes.This could be really useful e.g. in a situation where you would like to trigger a new “page view” event using your web analytics tool whenever a new page loads, as in the following example:"
a=text_splitting(a)
keyword_extraction(pos_split(a))






