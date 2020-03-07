import numpy as np
import torch
import matplotlib.pyplot as plt

class ESN:
    def __init__(self,n=200,p=0.1,n_in=1,n_out=1,g=1,seed=1,fb_scale=1,in_scale=1,out_scale=1,noise_scale=0.1,record=True):
        # Faithful reproduction of Jaeger and Hass 2004 for extendable gradient-descent in PyTorch
        self.n = n
        self.p = p
        self.n_in = n_in
        self.n_out = n_out
        self.g = g
        self.fb_scale = fb_scale
        self.record = record

        self.fb_scale = fb_scale
        self.in_scale = in_scale
        self.out_scale = out_scale
        self.noise_scale = noise_scale

        # Nonlinearity
        self.nl = torch.nn.Tanh()

        # Ensure reproducibility
        #torch.manual_seed(seed)

        # Initialize network parameters
        self.init_network()

        if self.record:
            self.x_agg = []
            self.y_agg = []
            self.x_agg.append(self.x)
            self.y_agg.append(self.y)

    def init_network(self):

        print('Initializing network with {} hidden neurons, spectral radius of {}, {} input neurons, {} output neurons'.format(self.n,self.g,self.n_in,self.n_out))

        # Reservoir weight matrix
        w_res = torch.empty(self.n,self.n).uniform_(-1,1)
        w_res[torch.rand(w_res.shape) > self.p] = 0

        # Scale reservoir weights by spectral radius (g)
        max_eig = np.max(np.abs(np.linalg.eigvals(w_res.detach().numpy())))
        print('Maximum eigenvalue of initialized weight matrix: {}'.format(max_eig))
        self.w_res = w_res * (self.g / max_eig)
        self.true_g = np.max(np.abs(np.linalg.eigvals(self.w_res.detach().numpy())))
        print('Maximum eigenvalue of scaled weight matrix: {}'.format(self.true_g))

        # Input and output weights
        if self.n_in > 0:
            self.w_in = torch.empty(self.n_in,self.n).uniform_(-1,1) * self.in_scale
        if self.n_out > 0:
            self.w_out = torch.empty(self.n,self.n_out).uniform_(-1,1)
            self.w_fb = torch.empty(self.n_out,self.n).uniform_(-1,1) * self.fb_scale

        # Initialize reservoir state vector
        self.x = torch.rand(self.n)

        # Starting output value of 0
        self.y = torch.zeros(self.n_in)

    def forward(self, u = torch.tensor([0.])):
        if self.record:
            self.y_agg.append(self.y.detach().numpy())
            self.x_agg.append(self.x.detach().numpy())
            
        res_x = torch.matmul(self.w_res,self.x)
        in_x = torch.matmul(u,self.w_in)
        fb_x = torch.matmul(self.y,self.w_fb)
        
   
        z = res_x + in_x + fb_x + (self.noise_scale * torch.randn(self.n))
        
    
        self.x = self.nl(z)
        self.y = self.nl(torch.matmul(self.x,self.w_out))

        
        

    def plot_history(self,start=0,end=-1,**kwargs):
        
        
        fig,ax = plt.subplots(2,1,sharex=True)
        
        keys = list(kwargs.keys())
        if 'title' in keys:
            fig.suptitle(kwargs['title'])
        
        fig.set_figheight(5)
        fig.set_figwidth(15)

        xs = np.vstack(self.x_agg)
        ys = np.asarray(self.y_agg)

        ax[0].imshow(xs.T[:,start:end],cmap='inferno')
        ax[1].plot(ys[start:end])

        plt.show()

    def run(self,steps=100):
        for step in range(steps):
            self.forward()

if __name__ == '__main__':

    dt = 0.1
    warmup_steps = 100
    train_steps = 2000

    # Initialize network
    net = ESN(n=1000,p=.01,g=.8,in_scale=0,fb_scale=0.1)

    # Let the network warmup
    net.run(warmup_steps)

    # Start forcing output to be teacher signal
    for train_step in range(train_steps):
        net.y = torch.sin(torch.tensor([train_step*dt]))
        net.forward()


    # Train readout weights on that epoch
    epochs = 40
    lr = 0.01
    states = torch.tensor(net.x_agg[(warmup_steps):(warmup_steps+train_steps)])
    targets = torch.tensor(net.y_agg[(warmup_steps):(warmup_steps+train_steps)])

    w_out = torch.tensor(net.w_out,requires_grad=True)

    for epoch in range(epochs):
        w_out.grad = None

        net_out = torch.matmul(states,w_out)
        diff = net_out - targets
        sq_err = diff.pow(2)

        mse = torch.sum(sq_err) / train_steps
        print(mse)
        mse.backward()

        w_out = torch.tensor(w_out - lr * w_out.grad,requires_grad=True)

    net.w_out = w_out
    net.run()
    net.plot_history()



