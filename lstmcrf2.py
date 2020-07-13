import spacy
import random
import torch   
import torch.nn as nn
from torchtext import data 
from spacy.tokenizer import Tokenizer 

def spacy_tokenize(x):
    return [tok.text for tok in tokenizer(x)]

SEED = 2019
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
TEXT = data.Field(tokenize = spacy_tokenize, tokenizer_language="en",batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float, batch_first=True)

fields = [(None, None), ('text',TEXT), ('label', LABEL)]
training_data = data.TabularDataset(path = 'Data/quora.csv', format = 'csv', fields = fields, skip_header = True)
train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))
print(f'training_data : {vars(training_data.examples[0])}')
print(f'train_data : {vars(train_data.examples[0])}')
print(f'valid_data : {vars(valid_data.examples[0])}')

TEXT.build_vocab(train_data, min_freq=3, vectors = "glove.6B.100d")  
LABEL.build_vocab(train_data)
print("Size of TEXT vocabulary:",len(TEXT.vocab))
print("Size of LABEL vocabulary:",len(LABEL.vocab))
print(TEXT.vocab.freqs.most_common(10))  
# print(TEXT.vocab.stoi)   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
train_iterator, valid_iterator = data.BucketIterator.splits( (train_data, valid_data),  batch_size = 64,
    sort_key = lambda x: len(x.text), sort_within_batch=True, device = device)

class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,bidirectional, dropout):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.Sigmoid()
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        outputs=self.act(dense_outputs)
        return outputs

size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers,bidirectional = True, dropout = dropout)
print(model)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)   
print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
print(pretrained_embeddings.shape)

import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()
def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
    
model = model.to(device)
criterion = criterion.to(device)
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()  
    for batch in iterator:
        optimizer.zero_grad()   
        text, text_lengths = batch.text  
        predictions = model(text, text_lengths).squeeze()
        loss = criterion(predictions, batch.label)  
        acc = binary_accuracy(predictions, batch.label)   
        loss.backward()       
        optimizer.step()     
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 5
path='Models/lstmcrf2ner.pt'
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), path)
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%\n')

model.load_state_dict(torch.load(path));
model.eval();

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()  

print(predict(model, "Are there any sports that you don't like?"))
print(predict(model, "Why Indian girls go crazy about marrying Shri. Rahul Gandhi ji?"))