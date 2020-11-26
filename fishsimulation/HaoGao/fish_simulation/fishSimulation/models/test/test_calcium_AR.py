from fishSimulation.utils.generator import rand_spikes
from fishSimulation.models.calcium import CalciumAR
import torch
import matplotlib.pyplot as plt

t = 1000 #ms
sp = rand_spikes(f=20, size=t)
ca = CalciumAR()

x_out = [0]
y_out = [ca.y[0].tolist()]
for i in range(t):
    ca.update(sp[i])
    x_out.append(i+1)
    y_out.append(ca.y[0].tolist())

plt.subplots()
plt.plot(x_out, y_out)  #画图
plt.show()