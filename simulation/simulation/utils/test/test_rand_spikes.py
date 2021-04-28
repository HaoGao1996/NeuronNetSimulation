from fishSimulation.utils.generator import rand_spikes
import matplotlib.pyplot as plt

sp1 = rand_spikes(f=10, size=1000)

# plot firing rate
print(sp1.sum())

sp2 = rand_spikes(f=10, size=(1000, 10))

# plot firing rate
print(sp2.sum(axis=0))



