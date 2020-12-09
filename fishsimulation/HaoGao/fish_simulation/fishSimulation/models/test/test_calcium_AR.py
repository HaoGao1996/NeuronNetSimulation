from fishSimulation.utils.generator import rand_spikes
from fishSimulation.models.calcium import CalciumAR
import matplotlib.pyplot as plt

t = 1000  #ms
sp = rand_spikes(f=10, size=t)
ca = CalciumAR(delta_t=1)

x_out = []
y_out = []
for i in range(t):
    ca.update(sp[i])
    x_out.append(i)
    y_out.append(ca.flu[0].tolist())

plt.subplots()
plt.plot(x_out, y_out)  #画图
plt.show()
