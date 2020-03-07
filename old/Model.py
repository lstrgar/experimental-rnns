import numpy as np
from copy import deepcopy

class Model():
    def __init__(self, dt=0.01, num_neurons = 200, sparsity = 0.3, 
            input_size = 0,output_size = 1,
            g_res = None):

        ### Class definitions
        self.num_neurons = num_neurons
        self.sparsity = sparsity
        self.input_size = input_size
        self.output_size = output_size
        self.dt = dt

        ### Scaling
        if g_res == None:
            self.w_res_scale = 1 / np.sqrt(self.num_neurons*sparsity)
        else:
            self.w_res_scale = g_res / np.sqrt(self.num_neurons*sparsity)

        self.w_rec_scale = self.w_res_scale / 10
        self.w_out_scale = 1 / np.sqrt(num_neurons)
        self.w_in_scale = 1


        ### Init weights and biases
        self.w_res = np.random.randn(self.num_neurons,self.num_neurons) * self.w_res_scale

        # Implement sparsity below
        self.w_res *= (np.random.rand(self.w_res.shape[0],
                                  self.w_res.shape[1]) < self.sparsity).astype(float)

        self.w_out = np.random.randn(self.num_neurons,self.output_size) * self.w_out_scale

        self.w_rec = np.random.randn(self.output_size,self.num_neurons) * self.w_rec_scale
        
        if self.input_size == 0:
            self.w_in = None
        else:
            self.w_in = np.random.randn(self.input_size,self.num_neurons) * self.w_in_scale 

        ### Initial state
        self.x_res = np.random.randn(num_neurons)
        
        self.r_res = np.expand_dims(np.tanh(self.x_res), axis = 1)
        
        self.y = np.zeros(self.output_size)
        
        ### Timing
        self.t = 0
        self.dt = self.dt
        
        ### History
        self.x_res_hist = []
        self.r_res_hist = []
        self.y_hist = []
        
    def reset_history(self):
        self.x_res_hist = []
        self.r_res_hist = []
        self.y_hist = []
        
    def step(self,feedback=False,act_in=None):
        if act_in is not None:
            feedback = True
            
        if feedback and act_in is None:
            self.x_res += self.dt * (-self.x_res + self.w_res @ np.tanh(self.x_res) + self.w_rec[0,:] * self.y[0])
            
        elif not feedback and act_in is None: 
            self.x_res += self.dt * (-self.x_res + self.w_res @ np.tanh(self.x_res))
            
        if feedback and act_in is not None:
            self.x_res += self.dt * (-self.x_res + self.w_res @ np.tanh(self.x_res) + self.w_rec[0,:] * self.y[0] + act_in.T @ self.w_in )
            
        x_res_checkpoint = deepcopy(self.x_res)
        self.x_res_hist.append(np.asarray(x_res_checkpoint))
        
        self.r_res = np.expand_dims(np.tanh(self.x_res), axis=1)
        self.r_res_hist.append(np.asarray(self.r_res))

        self.y = self.w_out.T @ self.r_res
        self.y_hist.append(np.asarray(self.y))
        
        self.t += self.dt
        

class Optimizer():
    def __init__(self,num_neurons=50,alpha=3,dt=0.01):
        self.P = 1/alpha * np.eye(num_neurons)
        self.alpha = alpha
        self.eta = 2e-3
        self.gamma = 2
        self.dt = dt
        
    def update(self,weights,state,target):
        error = (weights.T @ state - target)
        weights -= self.eta * error * state
        self.eta += self.dt * self.eta * ( -self.eta + np.abs(error)**self.gamma )
        return weights, error

def Target(x,targ_type='sine',mod = None):
    #The function we're trying to fit
    if targ_type == 'sine' and mod is None:
        scale = 2*np.pi/10
        return np.sin(scale*x)
    
    if targ_type == 'sine' and mod is not None:
        scale = 2*np.pi/10
        return np.sin(scale*x) * mod
    
    if targ_type == 'cosine' and mod is None:
        scale = 2*np.pi/10
        return np.cos(scale*x)
    
    if targ_type == 'mod_sine' and mod is None:
        scale = 2*np.pi/10
        return np.sin(scale*x)*x

        