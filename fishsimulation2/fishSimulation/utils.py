import torch

def sigmoid_func(x,a,b):
    assert a<b
    return (a*torch.exp(-x)+b)/(1+torch.exp(-x))

def inv_sigmoid_func(x,a,b):
    assert len(x[x<=a])==0 and len(x[x>=b])==0
    #d = (b-a)/1000
    #x[x<=a] = a+d #截断
    #x[x>=b] = b-d #截断
    return torch.log((x-a)/(b-x))