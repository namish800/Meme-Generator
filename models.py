#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


class Encoder(nn.Module):
    
    def __init__(self,encoded_image_size = 14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        
        resnet = torchvision.models.resnet101(pretrained=True)
        
        modules = list(resnet.children())[:-2]
        
        self.resnet = nn.Sequential(*modules)
        self.adaptie_pool = nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))
        
    def forward(self,images):
        out = self.resnet(images)
        out = self.adaptie_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out
        


# In[3]:


class Attention(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, enc_img, hidden_state):
        enc_att = self.encoder_att(enc_img)
        dec_att = self.decoder_att(hidden_state).unsqueeze(1)
        a = self.full_att(self.ReLU()(enc_att + dec_att)).squeeze(2)
        alpha = self.softmax(a)
        
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        return context,alpha     


# In[5]:


class Decoder(nn.Module):
    def __init__(self, num_embedding, embedding_dim, n_layers, hidden_dim, attention_dim, encoder_dim=2048, dropout=0.5):
        super(Decoder,self).__init__()
        
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        
        self.attention = Attention(encoder_dim=encoder_dim, decoder_dim = hidden_dim, attention_dim=attention_dim)
        
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTMCell((embedding_dim+encoder_dim), hidden_dim, n_layers)
        self.init_h = nn.Linear(encoder_dim,hidden_dim)
        self.init_c = nn.Linear(encoder_dim,hidden_dim)
        self.beta = nn.Linear(hidden_dim, encoder_dim)#Not sure why we need it.
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_dim, num_embedding)
        self.init_weights()
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)
    
    def load_glove(self,embeddings,require_grad=True):
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = require_grad
    
    def init_hidden_state(self, a):
        mean_a = a.mean(dim=1)
        h = self.init_h(mean_a)
        c = self.init_c(mean_a)
        return h,c
    
    def forward(self,enc_out,enc_captions,lengths):
        batch_size = enc_out.size(0)
        enc_dim = enc_out.size(-1)
        num_embedding = self.num_embedding
        
        enc_out = enc_out.view(batch_size, -1, enc_dim)
        num_pixels = enc_out.size(1)
        
        caption_lengths, ind = lengths.squeeze(1).sort(dim=0,descending=True)
        enc_out = enc_out[ind]
        enc_captions = enc_captions[ind]
        
        embeddings = self.embedding(enc_captions)
        h,c = self.init_hidden_state(enc_out)
        decode_lengths = (caption_lengths - 1).tolist()
        
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l>t for l in decode_lengths])
            attention_weighed_encoding, alpha = self.attention(enc_out[:batch_size_t],h[:batch_size_t])
            
            gate = self.sigmoid(self.beta(h[:batch_size_t]))
            attention_weighed_encoding = gate*attention_weighed_encoding
            
            h,c = self.lstm(torch.cat([embeddings[:batch_size_t,t,:], attention_weighed_encoding], dim=1),
                           h[:batch_size_t], c[:batch_size_t])
            
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t,t,:] = preds
            alphas[:batch_size_t,t,:] = alpha
        return predictions,alphas,enc_captions,decode_lengths,ind
    

        


# In[ ]:




