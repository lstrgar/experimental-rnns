import torch
import torch.nn
import numpy as np

def descend(param,grad,lr=1e-4):
    new_param = param - lr*grad
    return torch.tensor(new_param,requires_grad=True)
    

def sinMSE(y,t):
    target = torch.sin  
    val = target(torch.tensor(t))
    
    diff = y - val
    mse = diff.pow(2)
    
    return mse, val
    
