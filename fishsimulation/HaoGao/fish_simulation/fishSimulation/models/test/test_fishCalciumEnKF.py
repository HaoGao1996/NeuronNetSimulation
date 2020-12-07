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
sp_input = torch.load(join(dir, '20201130_spike_input_single_1ms.pkl'))
ca_output = torch.load(join(dir, '20201130_calcium_output_single_10ms.pkl'))

fc_enkf = FishCalciumEnKF(node_property=pro, w_uij=w, delta_tb=1,
                          alpha=10, lam=(1.1, -0.15), bl=0, delta_tc=10,
                          N=100, P=None, Q=None, R=None,
                          sp_input=sp_input)
fc_enkf.Q *= 0.001
fc_enkf.R *= 1

fc_enkf.init_sampling()

results = []
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
plt.show()
