from fishSimulation.utils.generator import gen_property, gen_spikes
from fishSimulation.neuron.block import block
import torch
import matplotlib.pyplot as plt

t = 1000 #ms
sp = gen_spikes(20)
pro = gen_property()
b = block(pro, delta_t=1)

x_out = []
y_out = []
for i in range(t):
    b.update(sp[i])
    x_out.append(b.t[0].tolist())
    y_out.append(b.V_i[0].tolist())

plt.subplots()
plt.plot(x_out, y_out)  #画图
plt.show()