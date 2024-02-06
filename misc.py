#!/usr/bin/env python
# coding: utf-8

# In[3]:


from torch import nn
from torch import optim
from torch.autograd import Variable
import torch

    
def one_hot(labelTensor, batchSize, n_l, use_cuda=False):
    oneHot = - torch.ones(batchSize * n_l).view(batchSize, n_l)
    for i, j in enumerate(labelTensor):
        oneHot[i, int(j.item())] = 1  # Convert j to int to use as an index
    if use_cuda:
        return Variable(oneHot).cuda()
    else:
        return Variable(oneHot)



def TV_LOSS(imgTensor,img_size=128):
    x = (imgTensor[:,:,1:,:]-imgTensor[:,:,:img_size-1,:])**2
    y = (imgTensor[:,:,:,1:]-imgTensor[:,:,:,:img_size-1])**2

    out = (x.mean(dim=2)+y.mean(dim=3)).mean()
    return out


# In[ ]:




