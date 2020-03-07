import numpy as np, torch

''' 
Force learning implemented for pytorch model.
Initialized with target function 'targ', time interval
'dt', constant parameter alpha 'a' used for p matrix
initialization, and dimension of the model 'n' 
'''
class Force:
    def __init__(self, targ, dt, a, n):
        self.targ = targ
        self.dt = dt
        self.a = a
        self.n = n
        self.p = (1/self.a) * torch.eye(self.n)
            

    def update_p(self, v):
        p = self.p
        num = torch.matmul(p, v)
        num = torch.matmul(num, v) * p
        den = torch.matmul(v, p)
        den = torch.matmul(den, v)
        den = 1 + den
        self.p = p - (num / den)
    

    def rls_update(self, olw, v, o, s):
        self.update_p(v)
        t = torch.tensor(self.dt * s)
        e = o - (4*self.targ(t))
        return olw - torch.matmul(e * self.p, v)
    

