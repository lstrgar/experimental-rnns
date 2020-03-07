import numpy as np
https://github.com/cknd/pyESNimport torch
import torch.nnT
from torch.distributions import normal
from torch.distributions import uniform

import matplotlib.pyplot as plt

'''
Naive reservoir model with 'n' neurons, 'p' fractional
connectivity, initial weight distribution from Normal(0, sig).
'''
class OriginalReservoir():
    def __init__(self,n=100,p=0.2,sig=0.1,bias=True,nl=np.tanh):
        self.n = n
        self.p = p
        self.sig = sig

        self.v = np.zeros(self.n) ## State Vector
        self.w = np.zeros([self.n,self.n]) ## Weight Matrix
        if bias: ## Network Bias
            self.b = np.random.randn(self.n)
        else:
            self.b = np.zeros(self.n)
        self.nl = nl ## Non-Linear Activation

        ## Populate Weight Matrix
        for i in range(self.n):
            for j in range(self.n):
                uni_draw = np.random.uniform()
                if uni_draw < self.p:
                    self.w[i,j] = np.random.normal(loc=0,scale=self.sig)


    '''
    Update state vector, given current state vector,
    weight matrix, and bias.
    '''
    def forward(self):
        z = np.matmul(self.w,self.v) + self.b
        y = self.nl(z)
        self.v = y
        return y



import torch, torch.nn as nn
from torch.distributions import normal
from torch.distributions import uniform

'''
PyTorch based reservoir model with 'n' neurons, 'p' fractional
connectivity, initial weight distribution from Normal(0, sig).
'''
class CUDAvoir(nn.Module):
    def __init__(self,n,p,sig,o=0.0,bias=True):
        super(CUDAvoir,self).__init__()

        self.n = torch.tensor(n)
        self.p = torch.tensor(p)
        self.sig = torch.tensor(sig)

        self.v = torch.zeros(self.n) ## Recurrent Layer State Vector
        self.w = torch.zeros(self.n,self.n) ## Recurrent Layer Weight Matrix

        self.ol = nn.Linear(self.n, 1, bias=False) ## Linear Output Layer
        self.o = torch.tensor([o]) ## Initalize Output Neuron
        self.fb = nn.Linear(1, self.n, bias=False) ## Linear Feedback Layer

        if bias: ## Recurrent Layer Bias
            self.b = torch.FloatTensor(n).uniform_(0,1)
        else:
            self.b = torch.zeros(self.n)
        
        ## Populate Recurrent Layer Weight Matrix
        norm = normal.Normal(loc=0,scale=self.sig)
        uni = uniform.Uniform(0,1) 
        for i in range(self.n):
            for j in range(self.n):
                uni_draw = uni.sample()
                if uni_draw < self.p:
                    self.w[i,j] = norm.sample()
    
    '''
    Update state vector and output vector given 
    current state vector and weight matrix, feedback
    layer, output, and bias
    '''
    def forward(self):
        z = torch.matmul(self.w,self.v) + self.fb(self.o) + self.b
        nl = nn.Tanh()
        y = nl(z)
        self.v = y
        self.o = self.ol(y)
        return y

        
class Reservoir():
    def __init__(self,n,p,g,init_pattern='random',bias=True,feedback=True,fb_scale=0.01,seed=1):
        torch.manual_seed(seed)
        
    
        
        if init_pattern == 'random':
            self.v = torch.randn(n)
        elif init_pattern == 'single':
            self.v = torch.zeros(n)
            self.v[0]=torch.tensor(1)
        else:
            self.v = torch.zeros(n)
        
        
            
        w = torch.zeros(n,n) ## Recurrent Layer Weight Matrix

        self.readout_w = torch.randn([n,1],requires_grad=True) ## Linear Output Layer
        
        self.y = torch.tensor([0.]) ## Initalize Output Neuron
        
        if bias: ## Recurrent Layer Bias
            self.b = torch.randn(n)
        else:
            self.b = torch.zeros(n)
            
        if feedback:
            self.fb_w = torch.randn([1,n]) * torch.tensor(fb_scale)
        else:
            self.fb_w = torch.zeros([1,n])
       
        
        ## Populate Recurrent Layer Weight Matrix
        scale_factor = g / np.sqrt(n)
        self.scale_factor = scale_factor
        
        norm = normal.Normal(loc=0,scale=scale_factor)
        uni = uniform.Uniform(0,1) 
                
        for i in range(n):
            for j in range(n):
                uni_draw = uni.sample()
                if uni_draw < p:
                    w[i,j] = norm.sample()
                    
        self.w = w
        
    
    def forward(self):
      
        z = torch.matmul(self.w,self.v) + self.b + torch.matmul(self.y,self.fb_w)
        
            
        nl = nn.Tanh()
        v = nl(z)
        y = torch.matmul(v,self.readout_w)
#         y = nl(y)
        
        self.v = v
        self.y = y
        
        return y
    
    def run(self,steps=200,plot=True,return_data=False):
        # Empty arrays for data collegtion          
            
        vs = []
        ys = []
        
        
        
        vs.append(self.v.detach().numpy())
        ys.append(self.y.detach().numpy())
        for t in range(steps):
            y = self.forward()
            v = self.v.detach().numpy()
            vs.append(v)
            ys.append(y.detach().numpy())
            
            
        vs = np.asarray(vs)
        
        if plot:
            plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.imshow(vs.T,cmap='viridis')
            
            plt.show()
            plt.plot(ys)
            
        if return_data:
            return vs, ys
        



