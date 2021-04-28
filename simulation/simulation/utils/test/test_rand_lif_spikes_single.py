from fishSimulation.utils import generator as gen
import matplotlib.pyplot as plt

sp = gen.rand_lif_spikes_single(size=(10000, 4), f=15, delta_tb=1, num=10, ratio=(0.8, 0.5))
print(sp.size)

plt.plot(sp)
plt.show()

