from fishSimulation.models.fishCalciumEnKF import FishCalciumEnKF
import fishSimulation.utils.generator as gen
import torch
from fishSimulation.data.fetcher import get_data_rd
from os.path import join
import matplotlib.pyplot as plt

t = 1000  #ms
K = 4
N = 1

w = torch.rand((K, N))
pro = gen.gen_property()
dir = get_data_rd()

sp_input = torch.load(join(dir, '20201208_spike_input_single_1ms.pkl'))
ca_output = torch.load(join(dir, '20201208_calcium_output_single_10ms.pkl'))

fc_enkf = FishCalciumEnKF(node_property=pro, w_uij=w, delta_tb=1,
                          alpha=10, lam=(1.1, -0.15), bl=1, delta_tc=10,
                          N=100, Q=torch.tensor([[0.000005, 0], [0, 0.002]]), R=torch.eye(1)*0.5,
                          sp_input=sp_input)
fc_enkf.Q *= 0.001
fc_enkf.R *= 0.5

fc_enkf.init_sampling()

results = []
<<<<<<< HEAD
T = 1000
for i in range(T):
    print(i)
    fc_enkf.run_enkf(ca_output[i])
    results.append(fc_enkf.get_x_all())

results = torch.stack(results, dim=0)


print(results.shape)
t = torch.arange(T)

fig, axes = plt.subplots(2, 1)
axes[0].scatter(t, results[:, :, 0].mean(1))
axes[0].plot([0, T], [10 / 500, 10 / 500], 'r', linewidth=2)
axes[1].scatter(t, results[:, :, 1].mean(1))
axes[1].plot([0, T], [1.5 / 60, 1.5 / 60], 'r', linewidth=2)
=======
for i in range(100):
    # print(i)
    fc_enkf.run_enkf(ca_output[i])
    results.append(fc_enkf.get_x_all())

results = torch.stack(results, dim=0)


print(results.shape)
t = torch.arange(100)

fig, axes = plt.subplots(2, 1)
axes[0].scatter(t, results[:, :, 0].mean(1))
axes[1].scatter(t, results[:, :, 1].mean(1))
>>>>>>> f8b86a3250fbea8c6da88a56e1eb29d41b66e324
plt.show()
