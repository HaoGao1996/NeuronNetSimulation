import matplotlib.pyplot as plt
from fishSimulation.data.fetcher import get_data_rd
import torch
from os.path import join
import fishSimulation.utils.generator as gen
from fishSimulation.models.fishCalciumEnKF import FishCalciumEnKF

## parameters
# size=(10000, 4)
# f=10, num=10,
# ratio=(0.8, 0.5)
# alpha=10
# lam=(1.1, -0.15)
# bl=0
# delta_tc=10
# std=1
# gen_property()

dir = get_data_rd()

sp_input = torch.load(join(dir, '20201130_spike_input_single_1ms.pkl'))
ca_output = torch.load(join(dir, '20201130_calcium_output_single_10ms.pkl'))

# fig, axes = plt.subplots(2, 1)
# axes[0].plot(sp_input.squeeze())
# axes[1].plot(ca_output)
# plt.show()

pro = gen.gen_property()
fc = FishCalciumEnKF(node_property=pro, w_uij=torch.ones(4, 1), delta_tb=1,
                     alpha=10, lam=(1.1, -0.15), bl=0, delta_tc=10,
                     N=100, P=None, Q=None, R=None,
                     sp_input=sp_input)

fc.P *= 0.00001
fc.Q *= 0.001
fc.R *= 2

fc.init_sampling_blocks()

print(fc)

