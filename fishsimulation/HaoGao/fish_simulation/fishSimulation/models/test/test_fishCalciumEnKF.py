from fishSimulation.models.fishCalciumEnKF import FishCalciumEnKF
import fishSimulation.utils.generator as gen
import torch
from fishSimulation.data.fetcher import get_data_rd
from os.path import join

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
fc_enkf.P *= 0.00001
fc_enkf.Q *= 0.001
fc_enkf.R *= 1

fc_enkf.init_sampling()

for i in range(1000):
    print(i)
    print(fc_enkf.run_enkf(ca_output[i]))

print(fc_enkf)