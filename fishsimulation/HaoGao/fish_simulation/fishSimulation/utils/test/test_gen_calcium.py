from fishSimulation.utils.generator import gen_calcium
import matplotlib.pyplot as plt

c = gen_calcium(t=1000, f=10)

plt.plot(c)
plt.show()
