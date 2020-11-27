from fishSimulation.utils import generator as gen
import matplotlib.pyplot as plt

c = gen.rand_lif_calcium(size=(1000, 4, 1), f=10, delta_t=1, num=10, ratio=(0.8, 0.5), a=10, lam=(1.1, -0.15), b=1, std=1)

plt.plot(c)
plt.show()