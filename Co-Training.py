#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch.optim as optim
import math
from sklearn.feature_extraction.text import TfidfTransformer
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset,DataLoader


# In[2]:


column_names = ['type','title','text']
unlabel_df = pd.read_csv('./udn_for_mct.tsv',sep='\t',names=column_names)


# In[3]:


unlabel_data = unlabel_df['text'].values


# In[4]:


column_names = ['type','title','text']
df = pd.read_csv('./all_after_mapping.tsv',sep='\t',names=column_names)


# In[5]:


tokenlizeword = np.load('tokenlizeword0225_nopunct.npy',allow_pickle=True)


# In[6]:


wmodel = Word2Vec(tokenlizeword, size=300, window=5, min_count=0)
wmodel.save("word2vec.model")


# In[7]:


labels = df['type'].values
print(type(labels))
# labels = np.array(labels)


# In[8]:


max_size = 512
x_lstm = []
for k in range(tokenlizeword.shape[0]):
  # every article have max_size * 300 embedding matrix
    embedding_matrix = np.zeros((max_size,300))
    for i in range(len(tokenlizeword[k])):
        if(i>=max_size):
            break
        embedding_matrix[i] = wmodel[tokenlizeword[k][i]]
    x_lstm.append(embedding_matrix)


# In[9]:


x_lstm = np.array(x_lstm,dtype='float32')


# In[ ]:


lstm_train_x = x_lstm[10000:]
lstm_train_y = labels[10000:]

lstm_test_x = x_lstm[5000:10000]
test_y = labels[5000:10000]


# In[ ]:


embedding_dim = 300
n_hidden = 128 # number of hidden units in one cell
num_classes = 7  
BATCH_SIZE = 8
epochs = 20
class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
#         return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]
        return context, soft_attn_weights # context : [batch_size, n_hidden * num_directions(=2)]


    def forward(self, X):
#         input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        input = X
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]
#         print(input)
        
        hidden_state = Variable(torch.zeros(1*2, BATCH_SIZE, n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, BATCH_SIZE, n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        hidden_state = hidden_state.double()
        cell_state = cell_state.double()
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
#         print(cell_state)
#         print(hidden_state)
#         print(hidden_state)
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
#         print('attn_output.shape',attn_output.shape)
#         print('attention.shape',attention.shape)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]


# In[ ]:


class Lstmdataset(Dataset):
    def __init__(self, x,y):
        self.x = torch.from_numpy(x)
#         self.x = torch.DoubleTensor(x)
#         self.x = self.x.double()
        self.y = torch.from_numpy(y)
#         self.y = torch.DoubleTensor(y)
#         self.y = self.y.double()
#         print(type(self.x))
#         print(type(self.y))

        self.len = x.shape[0]
    def __getitem__(self, index):
#         print(index)
        x = self.x[index]
        y = self.y[index]
#         print(type(x))
#         print(type(y))
        return x , y

    def __len__(self):
        return self.len
    
lstm_trainset = Lstmdataset(lstm_train_x,lstm_train_y)
lstm_trainloader = DataLoader(lstm_trainset,batch_size=BATCH_SIZE,drop_last=True)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)


# In[ ]:


print(type(tokenlizeword[0]))
count = 0
for k in range(tokenlizeword.shape[0]):
    tokenlizeword[k] = np.array(tokenlizeword[k])
    count+=tokenlizeword[k].shape[0]
print(count/tokenlizeword.shape[0])
print(type(tokenlizeword[0]))


# In[ ]:


li = []
for k in range(tokenlizeword.shape[0]):
    li.append(' '.join(tokenlizeword[k]))
li = np.array(li)
li.shape


# In[ ]:


vectorizer = CountVectorizer(max_features=512)
X = vectorizer.fit_transform(li)
word = vectorizer.get_feature_names()
# print(word)
# print(X.toarray())
transformer = TfidfTransformer()
print(transformer)
tfidf = transformer.fit_transform(X)
x = tfidf.toarray()
print(x.shape)


# In[ ]:


tfidf_test_x = x[5000:10000]
tfidf_train_x = x[10000:]
tfidf_train_y = labels[10000:]


# In[ ]:


print('tfidf_train_x shape:',tfidf_train_x.shape)
print('tfidf_train_y shape:',tfidf_train_y.shape)
print('tfidf_test_x shape:',tfidf_test_x.shape)
print('test_y shape:',test_y.shape)
print('lstm_train_x shape:',lstm_train_x.shape)
print('lstm_train_y shape:',lstm_train_y.shape)
print('lstm_test_x shape:',lstm_test_x.shape)
print('test_y shape:',test_y.shape)


# In[ ]:


class TFIDFdataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.len = x.shape[0]
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x,y
    def __len__(self):
        return self.len


# In[ ]:


num_classes = 7
input_dim  = x.shape[1]
class TFIDFmodel(nn.Module):
    def __init__(self):
        super(TFIDFmodel, self).__init__()
        self.fc1 = nn.Linear(input_dim,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# In[ ]:


unlabel_token = np.load('tokenlizeword0226_udn_for_mct.npy',allow_pickle=True)


# In[ ]:


wmodel_unlabel = Word2Vec(unlabel_token ,size=300, window=5, min_count=0)


# In[ ]:


max_size = 512
unlabel_for_lstm = []
for k in range(unlabel_token.shape[0]):
  # every article have max_size * 300 embedding matrix
    embedding_matrix = np.zeros((max_size,300))
    for i in range(len(unlabel_token[k])):
        if(i>=max_size):
            break
        embedding_matrix[i] = wmodel_unlabel[unlabel_token[k][i]]
    unlabel_for_lstm.append(embedding_matrix)
unlabel_for_lstm = np.array(unlabel_for_lstm,dtype='float32')
print(unlabel_for_lstm.shape)


# In[ ]:


print(type(unlabel_token[0]))
count = 0
for k in range(unlabel_token.shape[0]):
    unlabel_token[k] = np.array(unlabel_token[k])
    count+=unlabel_token[k].shape[0]
print(count/unlabel_token.shape[0])
print(type(unlabel_token[0]))
li = []
for k in range(unlabel_token.shape[0]):
    li.append(' '.join(unlabel_token[k]))
li = np.array(li)
print(li.shape)
vectorizer = CountVectorizer(max_features=512)
X = vectorizer.fit_transform(li)
word = vectorizer.get_feature_names()
# print(word)
# print(X.toarray())
transformer = TfidfTransformer()
print(transformer)
tfidf = transformer.fit_transform(X)
unlabel_for_tfidf = tfidf.toarray()
print(unlabel_for_tfidf.shape)


# In[ ]:


def train_lstm_model(dataloader):
    lstm_model = BiLSTM_Attention()
    lstm_model = lstm_model.double()
    lstm_model = lstm_model.to(device)
    lstm_model.train()
    criterion = nn.CrossEntropyLoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        running_loss = 0
        for data in dataloader:
            x,y = [t.to(device) for t in data]
            x = x.double()
            y = y.double()
            lstm_optimizer.zero_grad()
            output, attention = lstm_model(x)
            y = y.long()
            loss = criterion(output, y)
            loss.backward()
            lstm_optimizer.step()
            running_loss += loss.item()
        print('Epoch:',epoch+1,'loss=',running_loss)
    return lstm_model


# In[ ]:


def train_tfidf_model(dataloader):
    tfidf_model = TFIDFmodel()
    tfidf_model = tfidf_model.float()
    tfidf_model = tfidf_model.to(device)
    tfidf_model.train()
    criterion = nn.CrossEntropyLoss()
    tfidf_optimizer = optim.Adam(tfidf_model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        running_loss = 0
        for data in dataloader:
            x,y = [t.to(device) for t in data]
            x = x.float()
            y = y.float()
            tfidf_optimizer.zero_grad()
            output = tfidf_model(x)
            y = y.long()
            loss = criterion(output, y)
            running_loss += loss.item()
            loss.backward()
            tfidf_optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(running_loss))
    return tfidf_model


# In[ ]:


class TFIDF_unlabel_dataset(Dataset):
    def __init__(self,x):
        self.x = x
        self.len = x.shape[0]
    def __getitem__(self, index):
        x = self.x[index]
        return x
    def __len__(self):
        return self.len

class Lstm_unlabel_dataset(Dataset):
    def __init__(self, x):
#         print(type(x))
        self.x = torch.from_numpy(x)
#         print(type(self.x))
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        x = self.x[index]
        return x
    def __len__(self):
        return self.len

def create_tfidf_unlabel_dataloader_dataset(x):
#     BATCH_SIZE = 16
    unlabel_tfidf_trainset = TFIDF_unlabel_dataset(x)
    unlabel_tfidf_trainloader = DataLoader(unlabel_tfidf_trainset,batch_size=BATCH_SIZE,drop_last=True)
    return unlabel_tfidf_trainset,unlabel_tfidf_trainloader

def create_lstm_unlabel_dataloader_dataset(x):
#     BATCH_SIZE = 16
    unlabel_lstm_trainset = Lstm_unlabel_dataset(x)
    unlabel_lstm_trainloader = DataLoader(unlabel_lstm_trainset,batch_size = BATCH_SIZE,drop_last=True)
    return unlabel_lstm_trainset , unlabel_lstm_trainloader
    


# In[ ]:


def predict_model_lstm(model,dataloader):
    predictions = None
    predictions_withoutmax = None
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            
#             print(data.shape)
            # 將所有 tensors 移到 GPU 上
            print(type(data))
            if next(model.parameters()).is_cuda:
                x = data.to(device)
            x = x.double()
            outputs, state  = model(x)
            after_softmax = F.softmax(outputs, dim=1)
            _, pred = torch.max(after_softmax, 1)

            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
                
            if predictions_withoutmax is None:
                predictions_withoutmax = after_softmax
            else:
                predictions_withoutmax = torch.cat((predictions_withoutmax,after_softmax))
    return predictions_withoutmax
    


# In[ ]:


def predict_model_lstm_testing(model,dataloader):
    predictions = None
    predictions_withoutmax = None
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                x,y = [t.to("cuda:0") for t in data if t is not None]
            x = x.double()
            outputs , state  = model(x)
            after_softmax = F.softmax(outputs, dim=1)
            _, pred = torch.max(after_softmax, 1)

            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
                
            if predictions_withoutmax is None:
                predictions_withoutmax = after_softmax
            else:
                predictions_withoutmax = torch.cat((predictions_withoutmax,after_softmax))
    return predictions_withoutmax
    


# In[ ]:


def predict_model_tfidf(model,dataloader):
    predictions = None
    predictions_withoutmax = None
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                x = data.to(device)
            x = x.float()
            outputs = model(x)
            after_softmax = F.softmax(outputs, dim=1)
            _, pred = torch.max(after_softmax, 1)

            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
                
            if predictions_withoutmax is None:
                predictions_withoutmax = after_softmax
            else:
                predictions_withoutmax = torch.cat((predictions_withoutmax,after_softmax))
    return predictions_withoutmax
    


# In[ ]:


def predict_model_tfidf_testing(model,dataloader):
    predictions = None
    predictions_withoutmax = None
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                x,y = [t.to("cuda:0") for t in data if t is not None]

            x = x.float()
            outputs = model(x)
            after_softmax = F.softmax(outputs, dim=1)
            _, pred = torch.max(after_softmax, 1)

            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
                
            if predictions_withoutmax is None:
                predictions_withoutmax = after_softmax
            else:
                predictions_withoutmax = torch.cat((predictions_withoutmax,after_softmax))
    return predictions_withoutmax
    


# In[ ]:


def pick_high_confidence_data(lstm_result,tfidf_result):
    baseline = 0.7
    print(lstm_result.shape)
    print(tfidf_result.shape)
    count = 0
    li = []
    y = []
    print(lstm_result.shape[0])
    for i in range(lstm_result.shape[0]):
        _lstm = lstm_result[i]
        _tfidf = tfidf_result[i]
#         if lstm max value's index equals to tfidf's
        lstm_val , lstm_index = torch.max(_lstm, 0)
        tfidf_val , tfidf_index = torch.max(_tfidf, 0)
        if lstm_index.item() == tfidf_index.item():
            count+=1
            if lstm_val.item()>=baseline and tfidf_val.item()>=baseline:
                li.append(i)
                y.append(lstm_index.item())
    print(count)
    return np.array(li),np.array(y)

    
    


# In[ ]:


# def co_training():
# #     Define some parameter
# #     BATCH_SIZE = 16
    
for i in range(50):
    if i==0:
    #     init lstm label data
        label_lstm_x =  lstm_train_x
        label_lstm_y = lstm_train_y
#         print(type(label_lstm_x))
#         print('before label_lstm.shape:',label_lstm_x.shape,label_lstm_y.shape)

    #     init tfidf label data
        label_tfidf_x = tfidf_train_x
        label_tfidf_y = tfidf_train_y
#         print(type(label_tfidf_x))
#         print('before label_tfidf.shape:',label_tfidf_x.shape,label_tfidf_y.shape)


    #     init lstm unlabel data
        unlabel_lstm_x = unlabel_for_lstm
    #     init tfidf unlabel data
        unlabel_tfidf_x = unlabel_for_tfidf
#         print(type(unlabel_lstm_x))
#         print(type(unlabel_tfidf_x))
#         print('before unlabel_shape:',unlabel_lstm_x.shape,unlabel_tfidf_x.shape)
    
#     create lstm label trainset and trainloader
    lstm_trainset = Lstmdataset(label_lstm_x , label_lstm_y)
    lstm_trainloader = DataLoader(lstm_trainset,batch_size=BATCH_SIZE,drop_last=True)
    
#     create tfidf label trainset and trainloader
    tfidf_trainset = TFIDFdataset(label_tfidf_x,label_tfidf_y)
    tfidf_trainloader = DataLoader(tfidf_trainset,batch_size=BATCH_SIZE,drop_last=True)
    
#     create unlabel trainset and trainloader
    unlabel_lstm_trainset, unlabel_lstm_trainloader = create_lstm_unlabel_dataloader_dataset(unlabel_lstm_x)
    unlabel_tfidf_trainset,unlabel_tfidf_trainloader = create_tfidf_unlabel_dataloader_dataset(unlabel_tfidf_x)
    
#     start co-training 

#   some judgement here


#   using data to train lstm and tfidf model
    lstm_model = train_lstm_model(lstm_trainloader)
    tfidf_model = train_tfidf_model(tfidf_trainloader)
    
#     get predict result from model
    lstm_predict_result = predict_model_lstm(lstm_model, unlabel_lstm_trainloader)
    tfidf_predict_result = predict_model_tfidf(tfidf_model, unlabel_tfidf_trainloader)
#     choose which unlabel data should be moved to label data

    idx , y = pick_high_confidence_data(lstm_predict_result,tfidf_predict_result)
    
# #     update unlabel data

    unlabel_be_chosen_tfidf = np.take(unlabel_tfidf_x, idx, 0) 
    unlabel_be_chosen_lstm = np.take(unlabel_lstm_x, idx, 0) 
    
    unlabel_tfidf_x = np.delete(unlabel_tfidf_x, idx, axis=0)
    unlabel_lstm_x = np.delete(unlabel_lstm_x, idx, axis=0)
    
    print(unlabel_be_chosen_lstm.shape,type(unlabel_be_chosen_lstm))
    print(unlabel_be_chosen_tfidf.shape,type(unlabel_be_chosen_tfidf))

    
    
    label_lstm_x =  np.concatenate((label_lstm_x,unlabel_be_chosen_lstm))
    label_lstm_y = np.concatenate((label_lstm_y , y))
    
    label_tfidf_x =  np.concatenate((label_tfidf_x , unlabel_be_chosen_tfidf))
    label_tfidf_y = np.concatenate((label_tfidf_y , y))
    
    print('after combine label_lstm.shape:',label_lstm_x.shape,label_lstm_y.shape)
    print('after combine label_tfidf.shape:',label_tfidf_x.shape,label_tfidf_y.shape)
    print('after unlabel_shape:',unlabel_lstm_x.shape,unlabel_tfidf_x.shape)
    
    torch.save(lstm_model, 'lstm_model_cotraining.pkl')
    torch.save(tfidf_model, 'tfidf_model_cotraining.pkl')
    
    
    indices = np.arange(label_lstm_x.shape[0])
    np.random.shuffle(indices)

    label_lstm_x = label_lstm_x[indices]
    label_lstm_y = label_lstm_y[indices]
    
    label_tfidf_x = label_tfidf_x[indices]
    label_tfidf_y = label_tfidf_y[indices]
    


# In[ ]:


lstm_model = torch.load('lstm_model_cotraining.pkl')
lstm_model.eval()


# In[ ]:


tfidf_model = torch.load('tfidf_model_cotraining.pkl')
tfidf_model.eval()


# In[ ]:


lstm_test_x = x_lstm[5000:10000]
lstm_test_y = labels[5000:10000]

lstm_testset = Lstmdataset(lstm_test_x , test_y)
lstm_testloader = DataLoader(lstm_testset,batch_size=BATCH_SIZE,drop_last=True)
    
#     create tfidf label trainset and trainloader
tfidf_testset = TFIDFdataset(tfidf_test_x, test_y)
tfidf_testloader = DataLoader(tfidf_testset,batch_size=BATCH_SIZE,drop_last=True)


# In[ ]:


lstm_predict_result = predict_model_lstm_testing(lstm_model , lstm_testloader)
tfidf_predict_result = predict_model_tfidf_testing(tfidf_model, tfidf_testloader)


# In[ ]:


final_predict = lstm_predict_result + tfidf_predict_result


# In[ ]:


_ , ans =torch.max(final_predict, 1)


# In[ ]:


ans = torch.Tensor.cpu(ans).numpy()


# In[ ]:


(ans ==test_y).sum()/5000


# In[ ]:




