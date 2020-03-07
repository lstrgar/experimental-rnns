import torch, numpy as np, torch.nn as nn, matplotlib.pyplot as plt
from models import Reservoir, CUDAvoir
from optimizers import Force


'''
Runs naive reservoir model 'steps' number of
timesteps. Returns entire state vector history
when 'record' is set to True.
'''
def run_naive_net(net,steps,record=False):
    state_vecs = []
    for s in range(steps):
        y = net.forward()
        if record:
            state_vecs.append(y)
        else:
            pass
    if record:
        return np.asarray(state_vecs)


'''
Runs pytorch-based reservoir network 'steps'
number of timesteps. Returns entire state and 
output vector history when 'record' is set to True.
'''
def run_torch_net(net,steps,record=False):
    state_vecs = []
    outputs = []
    sin = torch.sin
    amp = 4
    func = lambda a : amp*sin(a)
    f = Force(targ=func,dt=0.1,a=3,n=net.n)
    for s in range(steps):
        y = net.forward()
        if record:
            state_vecs.append(y)
            outputs.append(net.o)
        else:
            pass
        net.ol.weight = nn.Parameter(f.rls_update(net.ol.weight,net.v,net.o,s))

    if record:
        state_vecs = torch.cat(state_vecs).detach().numpy()
        state_vecs = np.array_split(state_vecs, len(state_vecs)/net.n)
        outputs = torch.cat(outputs).detach().numpy()
        return (np.asarray(state_vecs), outputs)


if __name__ == '__main__':
    n = 1600
    p = 0.2
    sig = 0.3
    steps = 1000

    #res = Reservoir(n=n,p=p,sig=sig,bias=True)
    #svs = run_network(res,steps,record=True)

    cudares = CUDAvoir(n=n,p=p,sig=sig,o=0.0,bias=True)
    svs,ops = run_cudares(cudares,steps,record=True)

    plt.plot(ops)
    plt.show()

    # plt.imshow(svs.T)
    # plt.show()
