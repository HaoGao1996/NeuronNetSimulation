from fishSimulation.utils.generator import gen_property, rand_chemical
from fishSimulation.models.block import Block
import torch
import matplotlib.pyplot as plt

t = 1000 #ms
sp = rand_chemical(f=10, size=(10, t), E_ratio=0.8)
pro = gen_property()
b = Block(pro, delta_t=1)

x_out = []
y_out = []
for i in range(t):
    b.update(sp[i])
    x_out.append(b.t[0].tolist())
    y_out.append(b.V_i[0].tolist())

plt.subplots()
plt.plot(x_out, y_out)  #画图
plt.show()