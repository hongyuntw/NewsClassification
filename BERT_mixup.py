#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer
# from IPython.display import clear_output
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from transformers import BertForPreTraining
from transformers import BertForSequenceClassification
from transformers import BertModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt                   # For graphics
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from numpy import unique,where
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import OPTICS
import collections
import torch.optim as optim
from torch.distributions import Beta
import torch.nn as nn


# In[3]:


column_names = ['type','title','text']
dftrain = pd.read_csv('./data_after_sep/train.tsv',sep='\t',names=column_names)
dftest = pd.read_csv('./data_after_sep/test.tsv',sep='\t',names=column_names)
dfdev = pd.read_csv('./data_after_sep/dev.tsv',sep='\t',names=column_names)
testans = dftest['type'].values
testans = np.array(testans)


# In[4]:


from transformers import BertForPreTraining
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertForPreTraining.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('./chinese_wwm_pytorch/')
# model = BertForPreTraining.from_pretrained('./chinese_wwm_pytorch/')
# model = BertModel.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('./chinese_wwm_pytorch/')


# In[5]:


df_unlabel = pd.read_csv('./udn_for_mct.tsv',sep='\t',names=column_names)
df_all = pd.read_csv('./all_after_mapping.tsv',sep='\t',names=column_names)
li = [df_unlabel,df_all]
df_combine = pd.concat(li)


# In[6]:


class LabelDataset(Dataset):
    def __init__(self, input_dict, y ):
        self.input_ids = input_dict['input_ids']
        self.token_type_ids = input_dict['token_type_ids']
        self.attention_mask = input_dict['attention_mask']
        self.y = y
    def __getitem__(self,idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        y = self.y[idx]
        return inputid , tokentype , attentionmask, y
    def __len__(self):
        return len(self.input_ids)
    
class UnlabelDataset(Dataset):
    def __init__(self, input_dict):
        self.input_ids = input_dict['input_ids']
        self.token_type_ids = input_dict['token_type_ids']
        self.attention_mask = input_dict['attention_mask']
    def __getitem__(self,idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        return inputid , tokentype , attentionmask, 
    def __len__(self):
        return len(self.input_ids)
    


# In[7]:


# unlabel_texts = np.array(df_unlabel['text'].tolist())


# unlabel_dict = tokenizer.batch_encode_plus(unlabel_texts, 
#                                          add_special_tokens=True,
#                                          max_length=512,
#                                          return_special_tokens_masks=True,
#                                          pad_to_max_length=True,
#                                          return_tensors='pt')
# label_texts = np.array(df_all['text'].tolist())
# label_dict = tokenizer.batch_encode_plus(label_texts, 
#                                          add_special_tokens=True,
#                                          max_length=512,
#                                          return_special_tokens_masks=True,
#                                          pad_to_max_length=True,
#                                          return_tensors='pt')

# test_texts = np.array(dftest['text'].tolist())
# test_dict = tokenizer.batch_encode_plus(test_texts, 
#                                          add_special_tokens=True,
#                                          max_length=512,
#                                          return_special_tokens_masks=True,
#                                          pad_to_max_length=True,
#                                          return_tensors='pt')


# In[8]:


# BATCH_SIZE = 4

# unlabel_dataset = UnlabelDataset(unlabel_dict)
# unlabel_dataloader = DataLoader(unlabel_dataset, batch_size=BATCH_SIZE)
# label_dataset = UnlabelDataset(label_dict)
# label_dataloader = DataLoader(label_dataset, batch_size=BATCH_SIZE)

# test_dataset = UnlabelDataset(test_dict)
# test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE)


# # label_y  = df_all['type'].values
# # label_dataset = LabelDataset(label_dict,label_y)
# # label_dataloader = DataLoader(label_dataset, batch_size=BATCH_SIZE)


# In[9]:


class Classifier(nn.Module):  
    def __init__(self, num_labels, embedding_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_size, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,num_labels)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


# In[10]:


# torch.cuda.empty_cache()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device:", device)
# model = model.to(device)
# model.eval()
# label_768_outputs = []
# unlabel_768_outputs = []
# test_768_outputs = []
# for data in label_dataloader:
#     tokens_tensors, segments_tensors,masks_tensors = [t.to(device) for t in data]
#     with torch.no_grad():
#         outputs = model(input_ids=tokens_tensors, 
#                     token_type_ids=segments_tensors, 
#                     attention_mask=masks_tensors)
#         label_768_outputs.append(outputs[1].to('cpu'))
        
        
# for data in unlabel_dataloader:
#     tokens_tensors, segments_tensors,masks_tensors = [t.to(device) for t in data]
#     with torch.no_grad():
#         outputs = model(input_ids=tokens_tensors, 
#                     token_type_ids=segments_tensors, 
#                     attention_mask=masks_tensors)
#         unlabel_768_outputs.append(outputs[1].to('cpu'))
        
# for data in test_dataloader:
#     tokens_tensors, segments_tensors,masks_tensors = [t.to(device) for t in data]
#     with torch.no_grad():
#         outputs = model(input_ids=tokens_tensors, 
#                     token_type_ids=segments_tensors, 
#                     attention_mask=masks_tensors)
#         test_768_outputs.append(outputs[1].to('cpu'))


# In[11]:


# label_outputs_N_768 =[]
# for i in range(len(label_768_outputs)):
#     for k in range (len(label_768_outputs[i])):
#         label_outputs_N_768.append(label_768_outputs[i][k].numpy())
# label_outputs_N_768 = np.array(label_outputs_N_768)


# unlabel_outputs_N_768 =[]
# for i in range(len(unlabel_768_outputs)):
#     for k in range (len(unlabel_768_outputs[i])):
#         unlabel_outputs_N_768.append(unlabel_768_outputs[i][k].numpy())
# unlabel_outputs_N_768 = np.array(unlabel_outputs_N_768)


# test_outputs_N_768 = []
# for i in range(len(test_768_outputs)):
#     for k in range (len(test_768_outputs[i])):
#         test_outputs_N_768.append(test_768_outputs[i][k].numpy())
# test_outputs_N_768 = np.array(test_outputs_N_768)


# print(label_outputs_N_768.shape)
# print(unlabel_outputs_N_768.shape)
# print(test_outputs_N_768.shape)

# np.save('label_outputs_N_768',label_outputs_N_768)
# np.save('unlabel_outputs_N_768',unlabel_outputs_N_768)
# np.save('test_outputs_N_768',test_outputs_N_768)


# In[12]:


label_outputs_N_768 = np.load('label_outputs_N_768.npy')
unlabel_outputs_N_768 = np.load('unlabel_outputs_N_768.npy')
test_outputs_N_768 = np.load('test_outputs_N_768.npy')
print(label_outputs_N_768.shape)
print(unlabel_outputs_N_768.shape)
print(test_outputs_N_768.shape)


# In[13]:


class Label_768Dataset(Dataset):
    def __init__(self, x, y ):
        self.x = x
        self.y = y
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
    def __len__(self):
        return len(self.x)
    
label_y  = df_all['type'].values  
label_768_dataset = Label_768Dataset(label_outputs_N_768,label_y)
# Batch_size set 100 in paper
label_768_dataloader = DataLoader(label_768_dataset, batch_size=100)


test_y = dftest['type'].values
test_768_dataset = Label_768Dataset(test_outputs_N_768,test_y)
test_768_dataloader = DataLoader(test_768_dataset, batch_size=100)




# In[42]:


# # this is for one model

# classifier = Classifier(7,768)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device:", device)
# classifier = classifier.to(device)
# classifier.train()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

# # paper choose alpha from[0.1,0.2,0.5,1.0]
# alpha = 1
# T = 100

# for i in range(T):
#     running_loss = 0
#     for k,data in enumerate(label_768_dataloader,0):
# #         print(k)
# #         label data part [to get supervised loss]
#         x,y = [t.to(device) for t in data]
#         optimizer.zero_grad()
#         outputs = classifier(x)
#         supervised_loss = criterion(outputs,y)
# #         choose two unlabel data, uj uk is 768 dim
#         index = np.random.choice(unlabel_outputs_N_768.shape[0], 2, replace=False)  
#         uj = unlabel_outputs_N_768[index[0]]
#         uk = unlabel_outputs_N_768[index[1]]

        
#         uj = torch.from_numpy(uj)
#         uk = torch.from_numpy(uk)
#         uj = uj.to(device)
#         uk = uk.to(device)

# #         calculate fake label for uj and uk
#         y_uj = classifier(uj)
#         y_uj = torch.max(y_uj,0)[1]
        
#         y_uk = classifier(uk)
#         y_uk = torch.max(y_uk,0)[1]
        
# #         sample a lambda
#         m = Beta(torch.FloatTensor([alpha]), torch.FloatTensor([alpha]))
#         _lambda = m.sample().to(device)
    
# #     mixup
#         um = _lambda*uj + (1-_lambda)*uk
#         y_um = (_lambda*y_uj) + ((1-_lambda)*y_uk)
        
#         with torch.no_grad():
#             outputs = classifier(um)
#             outputs = torch.max(outputs,0)[1]
#             unsupervised_loss = (outputs-y_um)**2
            
# #         calculate all loss L = Ls + w(t)*Lus
            
#         lus = Variable(torch.Tensor([(i+1)*unsupervised_loss]),requires_grad=True)
#         lus = lus.to(device)
#         loss = supervised_loss + lus
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
# #         if k % 100 == 99:    # print every 2000 mini-batches
#     print('[%d] loss: %.3f' % (i + 1, running_loss))
        


# In[ ]:


# this is for teacher student model

shadow = {}
# def init_params(student_classifier,teacher_classifier):
def init_params():

#     copy params to shadow
    for name, param in student_classifier.named_parameters():
        shadow[name] = param.data.clone()
#    copy shadow params to teacher model
    for name, param in teacher_classifier.named_parameters():
        param.data = shadow[name]

    
#     def update_params(student_classifier,teacher_classifier,alpha):
# 
def update_params():
#     cal new params for teacher and save in shadow
    for name, param in student_classifier.named_parameters():
        new_average = (1.0 - alpha) * param.data + alpha * shadow[name]
        shadow[name] = new_average.clone()
        
        #    copy shadow params to teacher model      
    for name, param in teacher_classifier.named_parameters():
        param.data = shadow[name]




student_classifier = Classifier(7,768)
# teacher_classifier = student_classifier
teacher_classifier = Classifier(7,768)

# paper choose alpha from[0.1,0.2,0.5,1.0]
alpha = 0.8
T = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
student_classifier = student_classifier.to(device)
teacher_classifier = teacher_classifier.to(device)
student_classifier.train()
teacher_classifier.train()

# init teacher model
# init_params(student_classifier,teacher_classifier)
init_params()



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_classifier.parameters(), lr=1e-5)



for i in range(T):
    running_loss = 0
    for k,data in enumerate(label_768_dataloader,0):
#         print(k)
#         label data part [to get supervised loss]
        x,y = [t.to(device) for t in data]
        optimizer.zero_grad()
        outputs = student_classifier(x)
        supervised_loss = criterion(outputs,y)
        
#         choose two unlabel data, uj uk is 768 dim
        index = np.random.choice(unlabel_outputs_N_768.shape[0], 2, replace=False)  
        uj = unlabel_outputs_N_768[index[0]]
        uk = unlabel_outputs_N_768[index[1]]
        uj = torch.from_numpy(uj).to(device)
        uk = torch.from_numpy(uk).to(device)
#         uj = uj.to(device)
#         uk = uk.to(device)
        
        

#         calculate fake label for uj and uk
        y_uj = teacher_classifier(uj)
        y_uj = torch.max(y_uj,0)[1]
        y_uk = teacher_classifier(uk)
        y_uk = torch.max(y_uk,0)[1]
        
#         sample a lambda
        m = Beta(torch.FloatTensor([alpha]), torch.FloatTensor([alpha]))
        _lambda = m.sample().to(device)
    
#     mixup
        um = _lambda*uj + (1-_lambda)*uk
        y_um = (_lambda*y_uj) + ((1-_lambda)*y_uk)
#         y_um.requires_grad = True
        y_um = Variable(torch.FloatTensor(y_um),requires_grad=True)
        
        outputs = student_classifier(um)
        outputs = torch.max(outputs,0)[1].float()
        outputs.requires_grad = True
#         outputs = Variable(torch.FloatTensor(torch.max(outputs,0)[1]),requires_grad=True)
        unsupervised_loss = (outputs-y_um)**2
#         unsupervised_loss.requires_grad= True
            
#         calculate all loss L = Ls + w(t)*Lus
        lus = Variable(torch.Tensor([(i+1)*unsupervised_loss]),requires_grad=True)
        lus = lus.to(device)
        loss = supervised_loss + lus
        loss.backward()
        optimizer.step()
#         update teacher model
#         update_params(student_classifier,teacher_classifier,alpha)
        update_params()
        
        running_loss += loss.item()
#         if k % 100 == 99:    # print every 2000 mini-batches
    print('[%d] loss: %.3f' % (i + 1, running_loss))
        


# In[20]:


correct = 0
total = 0
with torch.no_grad():
    for data in test_768_dataloader:
        x,y = [t.to(device) for t in data]
        outputs = student_classifier(x)
        pred = torch.max(outputs,1)[1]
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = correct / total
    print('test data acc: %.3f' %(acc))


# In[21]:


# outputs.size()


# In[ ]:




