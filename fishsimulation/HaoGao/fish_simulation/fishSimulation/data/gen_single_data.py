from fishSimulation.utils import generator as gen
import matplotlib.pyplot as plt
from fishSimulation.data.fetcher import get_data_rd
import torch
from os.path import join

dir = get_data_rd()


ca_output, sp_input = gen.rand_lif_calcium_single(size=(10000, 4), f=20, num=10, ratio=(0.8, 0.5),
                                     alpha=10, lam=(1.1, -0.15), bl=0, delta_tc=10, std=1)

torch.save(sp_input, join(dir, '20201130_spike_input_single_1ms.pkl'))
torch.save(ca_output, join(dir, '20201130_calcium_output_single_10ms.pkl'))

fig, axes = plt.subplots(2, 1)
axes[0].plot(sp_input.squeeze())
axes[1].plot(ca_output)
plt.savefig(join(dir, '20201130_single.tiff'))
plt.show()

