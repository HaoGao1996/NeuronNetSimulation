from fishSimulation.utils.generator import gen_property
from fishSimulation.neuron.block import block
import torch
import matplotlib.pyplot as plt

t = 100 #ms
sp = torch.poisson(torch.ones(t)*1)
pro = gen_property()
b = block(pro, delta_t=10)

x_out = []
y_out = []
for i in range(t):
    x, y = b.update(sp[i])
    x_out.append(x)
    y_out.append(y)

plt.subplots()
plt.plot(x_out,y_out)  #画图
plt.show()#loss函数图