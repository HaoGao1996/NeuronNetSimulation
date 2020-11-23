from EnKF import EnsembleKalmanFilter as enfk
import torch

def fx(x, F):
    return F @ x

def hx(x):
    return torch.tensor([x[0]])

x = torch.tensor([5, 3])
P = torch.eye(2) * 100
F = torch.tensor([[1., 1.], [0., 1.]])
H = torch.tensor([1., 0.])
N = 100

enfk0 = enfk(x=x, P=P, N=N)

print(enfk0)