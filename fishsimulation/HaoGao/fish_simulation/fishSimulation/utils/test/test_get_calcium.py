from fishSimulation.utils import generator as gen
import matplotlib.pyplot as plt

t = 1000
sp = gen.rand_spikes(f=10, size=t)
c = gen.get_calcium(sp)

spn = gen.rand_spikes(f=10, size=t)
cn = gen.get_calcium(spn, std=1)

plt.plot(c)
plt.plot(cn)
plt.show()
