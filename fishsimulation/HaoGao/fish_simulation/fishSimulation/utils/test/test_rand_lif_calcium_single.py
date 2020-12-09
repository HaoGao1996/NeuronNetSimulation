from fishSimulation.utils import generator as gen
import matplotlib.pyplot as plt

ca_output, sp_input = gen.rand_lif_calcium_single(size=(10000, 4), f=10, num=10, ratio=(0.8, 0.5),
                                     alpha=10, lam=(1.1, -0.15), bl=1, delta_tc=10, std=2)


plt.plot(ca_output)
plt.show()