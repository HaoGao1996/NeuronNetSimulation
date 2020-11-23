# from fishSimulation.kalmanFilter.EnKF import EnsembleKalmanFilter
from fishSimulation.kalmanFilter.KF import KalmanFilter
import torch
import matplotlib.pyplot as plt

# def fx(x, F):
#     return F @ x
#
# def hx(x):
#     return torch.tensor([x[0]])

x = torch.tensor([[5, 3]])
P = torch.eye(2) * 100
F = torch.tensor([[1., 1.], [0., 1.]])
H = torch.tensor([[1., 0.]])
N = 100
T = 100

std_noise = 10
kf = KalmanFilter(dim_x=2, dim_z=1, F=F, H=H)
kf.R *= std_noise**2
kf.Q *= 0.002

zt = []
kf_result = []

for t in range(T):
    z = t + torch.randn(1)[0].tolist()*std_noise
    zt.append(z)

    kf.predict()
    kf.update(z=z)

    kf_result.append(kf.x[0].tolist())

fig = plt.figure(figsize=(10, 5))
plt.plot(zt, 'go', label='measurement', alpha=0.3)
plt.plot(kf_result, label='KF', c='b', lw=2)
plt.legend(loc='best');
plt.show()