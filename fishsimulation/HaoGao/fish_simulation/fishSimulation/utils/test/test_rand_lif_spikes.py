from fishSimulation.utils import generator as gen

sp = gen.rand_lif_spikes(size=(1000, 4, 1), f=10, delta_t=1, num=10, ratio=(0.8, 0.5))

print(sp.size())
