import fishSimulation.utils.generator as gen
from fishSimulation.models.block import Block
import torch
import matplotlib.pyplot as plt

t = 10000  #ms
K = 4
N = 1

w = torch.rand((K, N))
pro = gen.gen_property()
b = Block(pro, w_uij=w, delta_t=1)

sp = gen.rand_input_single(f=20, size=(t, K), num=10, ratio=(0.8, 0.5))

x_out = []
y_out = []
for i in range(t):
    b.update(sp[i])
    x_out.append(b.t)
    y_out.append(b.V_i[0].tolist())

plt.subplots()
plt.plot(x_out, y_out)  #画图
plt.show()