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


'''
Backpropagation through time algorithm. Receives
's' number of steps to run, 'ip' input sequence,
'op' output sequence, 'targ' target output function
(relevant if ip+op are empty), 
'''
class BPTT:
    def __init__(self, s, ip=[], op[], targ):



class TBPTT:
    def __init__(self,)
