from fishSimulation.models.fishCalcium import FishCalcium as fc
import fishSimulation.utils.generator as gen
import torch

t = 10000  #ms
K = 4
N = 1

w = torch.rand((K, N))
pro = gen.gen_property()
b = fc(node_property=pro, w_uij=w, alpha=10, lam=(1.3, -0.5), bl=0, delta_tb=1, delta_tc=2)
b.update_block(1)

print(b)
