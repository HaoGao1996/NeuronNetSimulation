import fishSimulation.utils.generator as gen
from fishSimulation.models.fishCalcium import FishCalcium
import torch
import matplotlib.pyplot as plt

t = 10000  #ms
K = 4
N = 1

w = torch.rand((K, N))
pro = gen.gen_property()
sp = gen.rand_input_single(f=20, size=(t, K), num=10, ratio=(0.8, 0.5))

fc = FishCalcium(pro, w_uij=w, delta_tb=1, sp_input=sp, delta_tc=50)

x1_out = []
y1_out = []
x2_out = []
y2_out = []
for i in range(t):
    fc.update()
    x1_out.append(fc.t)
    y1_out.append(fc.block.V_i[0].tolist())
    if fc.t % fc.calciumAR.delta_t:
        x2_out.append(fc.t)
        y2_out.append(fc.calciumAR.flu[0].tolist())

fig, axes = plt.subplots(2, 1)
axes[0].plot(x1_out, y1_out)
axes[1].plot(x2_out, y2_out)
plt.show()