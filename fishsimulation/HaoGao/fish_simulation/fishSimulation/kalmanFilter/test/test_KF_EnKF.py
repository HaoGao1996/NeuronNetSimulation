from fishSimulation.kalmanFilter.EnKF import EnsembleKalmanFilter
from fishSimulation.kalmanFilter.KF import KalmanFilter
import torch
import matplotlib.pyplot as plt

F = torch.tensor([[1., 1.], [0., 1.]])
H = torch.tensor([[1., 0.]])


def fxu(x):
    return F @ x

def hx(x):
    return torch.tensor([x[0]])

T = 100

x = torch.tensor([5., 3.])
P = torch.eye(2) * 100
N = 100
enkf = EnsembleKalmanFilter(x=x, P=P, dim_x=2, dim_z=1, hx=hx, fxu=fxu, N=N)
std_noise = 10
enkf.R *= std_noise**2
enkf.Q *= 0.002



kf = KalmanFilter(dim_x=2, dim_z=1, F=F, H=H)
kf.R *= std_noise**2
kf.Q *= 0.002

zt = []
kf_results = []
enkf_results = []

for t in range(T):
    z = t + torch.randn(1)[0].tolist()*std_noise
    zt.append(z)

    kf.predict()
    kf.update(z=z)

    enkf.predict()
    enkf.update(z=z)

    kf_results.append(kf.x[0].tolist())
    enkf_results.append(enkf.x[0].tolist())

fig = plt.figure(figsize=(10, 5))
plt.plot(zt, 'go', label='measurement', alpha=0.3)
plt.plot(kf_results, label='KF', c='b', lw=2, alpha=0.5)
plt.plot(enkf_results, label='EnKF', c='r', lw=1, alpha=0.5)
plt.plot(torch.arange(100), label='true', c='y', lw=1, alpha=0.5)
plt.legend(loc='best');
plt.show()