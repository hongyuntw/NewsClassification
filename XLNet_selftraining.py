#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import pandas as pd
from transformers import XLNetTokenizer
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import XLNetForSequenceClassification


# In[2]:


class TrainDataset(Dataset):
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

class TestDataset(Dataset):
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
    


# In[3]:


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    predictions_withoutmax = None
    correct = 0
    total = 0
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            after_softmax = F.softmax(logits.data, dim=1)
            _, pred = torch.max(after_softmax, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
                
            if predictions_withoutmax is None:
                predictions_withoutmax = after_softmax
            else:
                predictions_withoutmax = torch.cat((predictions_withoutmax,after_softmax))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions_withoutmax


def pick_high_confidence_data(result):
    baseline = 0.86
    print("======== picking hign confidence data ========")
    print(result.shape)
    count = 0
    li = []
    y = []
    for i in range(result.shape[0]):
        _res = result[i]
#         if lstm max value's index equals to tfidf's
        _val , _index = torch.max(_res, 0)
        if _val.item()>=baseline:
                li.append(i)
                y.append(_index.item())
    return np.array(li) , np.array(y)


# In[4]:


column_names = ['type','title','text']
# unlabel data
df_unlabel = pd.read_csv('./udn_for_mct.tsv',sep='\t',names=column_names)
# label data
df_all = pd.read_csv('./all_after_mapping.tsv',sep='\t',names=column_names)


NUM_LABELS = 7

tokenizer = XLNetTokenizer.from_pretrained('./chinese_xlnet_mid_pytorch/')
model  = XLNetForSequenceClassification.from_pretrained('./chinese_xlnet_mid_pytorch/',num_labels=NUM_LABELS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
BATCH_SIZE = 2


# In[ ]:




i = 0
while(1):
    if i==0:
        traintexts = np.array(df_all['text'].tolist())
        train_input_dict = tokenizer.batch_encode_plus(traintexts, 
                                         add_special_tokens=True,
                                         max_length=512,
                                         return_special_tokens_masks=True,
                                         pad_to_max_length=True,
                                         return_tensors='pt')
        train_y = np.array(df_all['type'].values)

        unlabeltexts = np.array(df_unlabel['text'].tolist())
        unlabel_input_dict = tokenizer.batch_encode_plus(unlabeltexts, 
                                         add_special_tokens=True,
                                         max_length=512,
                                         return_special_tokens_masks=True,
                                         pad_to_max_length=True,
                                         return_tensors='pt')

    else:
        train_input_dict = tokenizer.batch_encode_plus(traintexts, 
                                         add_special_tokens=True,
                                         max_length=512,
                                         return_special_tokens_masks=True,
                                         pad_to_max_length=True,
                                         return_tensors='pt')
        unlabel_input_dict = tokenizer.batch_encode_plus(unlabeltexts, 
                                         add_special_tokens=True,
                                         max_length=512,
                                         return_special_tokens_masks=True,
                                         pad_to_max_length=True,
                                         return_tensors='pt')
        
        
    trainset = TrainDataset(train_input_dict,train_y)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 2  # 幸運數字
    for epoch in range(EPOCHS):
        step = 0
        running_loss = 0.0
        for data in trainloader:
            tokens_tensors, segments_tensors,             masks_tensors, labels = [t.to(device) for t in data]
            outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors, 
                                labels=labels)
            optimizer.zero_grad()

            loss = outputs[0]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        torch.save(model, 'XLNet_selftraining_' + str(i) + "_" + str(epoch) + '.pkl')
        print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))
    
    unlabelset = TestDataset(unlabel_input_dict)
    unlabelloader = DataLoader(unlabelset, batch_size=32)
    
    ans_matrix = get_predictions(model, unlabelloader, False)
    
    idx , y = pick_high_confidence_data(ans_matrix)
    
    
    

    unlabel_be_chosen = np.take(unlabeltexts, idx, 0)
    unlabeltexts = np.delete(unlabeltexts,idx,axis=0)
    traintexts = np.concatenate((traintexts,unlabel_be_chosen))
    train_y = np.concatenate((train_y,y))

    torch.save(model, 'XLNet_selftraining_' + str(i) +'.pkl')

    
    if(unlabeltexts.shape[0]<100):
        break
    i += 1


# In[8]:


get_ipython().system('sudo jupyter nbconvert --to script XLNet_selftraining.ipynb')


# In[ ]:



get_ipython().system('')


# In[ ]:




