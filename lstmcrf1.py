import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from math import nan
import itertools
#import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class SentenceGetter:
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped] 
        self.maxsentlen = max([len(s) for s in self.sentences])
        self.padd = ('ENDPAD', 'not_kw')
        for sent in self.sentences:
            leng = self.maxsentlen-len(sent)
            sent.extend(itertools.repeat(self.padd, leng)) 

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

seed_value = 2020
np.random.seed(seed_value)
df = pd.read_csv("Data/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
data = df.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos', 'next-next-shape', 'next-next-word', 'next-pos', 
    'next-shape', 'next-word', 'prev-iob', 'prev-lemma', 'prev-pos', 'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 
    'prev-prev-shape', 'prev-prev-word', 'prev-shape', 'shape', 'prev-word',"pos"],axis=1)
data = df[['sentence_idx','word','tag']]
data.replace(["B-geo", "B-tim", "B-org", "I-per", "B-per", "I-org", "B-gpe", "B-gpe", "I-geo", "I-tim", "B-art", "B-eve", "I-eve", "I-art", "I-gpe", "B-nat", "I-nat"], "YES_KEYWORD", inplace=True) 
data.replace(["O", "unk"], "not_kw", inplace=True) 
print('data.head :\n{}\n'.format(data.head()) )
print('tag.value_counts :\n{}\n'.format(data['tag'].value_counts()) )

getter = SentenceGetter(data)
sentences = getter.sentences
maxlen = getter.maxsentlen
print(f'maxlen : {maxlen}\n')

words = list(set(data["word"].values))
words.append("ENDPAD")
n_words = len(words)
print(f'words : {words[:25]}')
print(f'n_words : {n_words}')

tags = []
for tag in set(data["tag"].values):
    if tag is nan or isinstance(tag, float):
        tags.append('unk')
    else:
        tags.append(tag)
n_tags = len(tags)
print(f'\n tags : {tags}')
print(f'n_tags : {n_tags}')

word2idx = {w: i for i, w in enumerate(words)}
idx2word = {v: k for k, v in dict.items(word2idx)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {v: k for k, v in dict.items(tag2idx)}
print(f'\n word2idx : {dict(itertools.islice(word2idx.items(), 25))}')
print(f'\n tag2idx : {tag2idx}')

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

X = [[word2idx[w[0]] for w in s] for s in sentences]
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = [to_categorical(i, num_classes=n_tags) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print(f'\n X_train : {X_train[:1]}')
print(f'\n y_train : {y_train[:1]}')

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y    
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        # return self.X[idx][0], self.y[idx], self.X[idx][1]
        return torch.as_tensor(np.array(self.X[idx][0])), self.y[idx], self.X[idx][1]

train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_test, y_test)
print(f'idk 0 :\n{torch.as_tensor(np.array(X_train[0][0]))} | {y[0][0]} | {X_train[0][1]}')

def train_model(model, epochs=5, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            print(f'x size : {x.shape}\nx : {x}\ny size : {y.shape}\ny : {y}\nl size : {l.shape}\nl : {l}\n')
            y_pred = model(x, l)
            print(f'y_pred size : {y_pred.shape}\ny_pred : {y_pred}\n')
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        # val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        # if i % 5 == 1:
        #     print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

# def validation_metrics (model, valid_dl):
#     model.eval()
#     correct = 0
#     total = 0
#     sum_loss = 0.0
#     sum_rmse = 0.0
#     for x, y, l in valid_dl:
#         x = x.long()
#         y = y.long()
#         y_hat = model(x, l)
#         loss = F.cross_entropy(y_hat, y)
#         pred = torch.max(y_hat, 1)[1]
#         correct += (pred == y).float().sum()
#         total += y.shape[0]
#         sum_loss += loss.item()*y.shape[0]
#         sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
#     return sum_loss/total, correct/total, sum_rmse/total

batch_size = 256
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)

class LSTM_fixed_len(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 3)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, l):
        x = self.embeddings(x).view(256, 1, 1)
        x = self.dropout(x)
        print(f'1 x size : {x.shape}\n1 x : {x}\n')
        lstm_out, (ht, ct) = self.lstm(x)
        x = F.sigmoid(x)
        return self.linear(x)
        #return self.linear(ht[-1])

model_fixed =  LSTM_fixed_len(n_words, 50, 50)
train_model(model_fixed, epochs=5, lr=0.01)